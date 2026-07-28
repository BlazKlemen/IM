import argparse
import copy
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

PREPROCESS_CACHE_DIR = Path(__file__).resolve().parent / ".preprocess_cache"
# Bump this any time preprocess_point_cloud() or anything it calls (normal
# estimation) changes behavior for the same inputs - the cache key only
# covers *parameters*, not code version, so a bug fix with no parameter
# change would otherwise keep serving the old, wrong cached result
# indefinitely instead of recomputing.
# v6: switched simulated target-scan generation from hidden_point_removal
# (point-based occlusion heuristic) to raycasting against the actual CAD
# mesh (simulate_camera_scan_raycast) - a v5 cache entry was computed with
# entirely different target points and must never be served after this
# change, even though target_cache_key's own string didn't change shape.
PREPROCESS_CACHE_VERSION = 6

# funkcija za vizualizacijo rezultatov registracije
def draw_registration_result(source: o3d.geometry.PointCloud,   # izvorni oblak točk
                             target: o3d.geometry.PointCloud,   # ciljni oblak točk
                             transformation: np.ndarray,      # transformacijska matrika
                             window_name: str = "Alignment") -> None:   # ime okna za vizualizacijo
    source_temp = copy.deepcopy(source)     # kopiramo izvorni oblak točk, da ne spremenimo originala
    target_temp = copy.deepcopy(target)     # kopiramo ciljni oblak točk, da ne spremenimo originala
    source_temp.paint_uniform_color([1.0, 0.706, 0.0])
    target_temp.paint_uniform_color([0.0, 0.651, 0.929])
    source_temp.transform(transformation)   # uporabimo transformacijsko matriko na izvorni oblak točk
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)  # nariše


# nalaganje cad modela
def load_cad_model(mesh_path: Path, n_points: int = 200000, use_cache: bool = True) -> o3d.geometry.PointCloud:  # pot do modela + število točk
    # Poisson-disk sampling on a fine STL is the slow part of every run, and
    # it's not reproducible on its own - Open3D's sampler has its own
    # internal RNG that np.random.seed() doesn't control, so point counts
    # drift run to run. Caching the sampled cloud to disk (keyed by n_points,
    # invalidated if the STL changes) fixes both: instant reload after the
    # first run, and identical points every time after that.
    # NOTE on the default n_points=200000: this is sampled over the *whole*
    # housing before crop_top_region keeps ~35% and simulated occlusion keeps
    # a further fraction, so the realistically-visible/relevant region ends up
    # roughly 1/15th of this (~13k points at 200k). That density (sub-mm
    # effective point spacing over a ~150mm part) is what a real structured-
    # light / laser-triangulation industrial scanner actually delivers - the
    # old 20000 default left target at only ~1300 points, unrealistically
    # sparse and noise-fragile. Poisson-disk at 200k costs a few minutes the
    # *first* run, then the cache makes every subsequent run instant.
    cache_path = mesh_path.with_name(f"{mesh_path.stem}.sampled_{n_points}pts.ply")   # pot do cache datotetke, da ni treba vsakič vzorčit CAD fila
    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= mesh_path.stat().st_mtime:
        pcd = o3d.io.read_point_cloud(str(cache_path))  # preverimo, če cache datoteka obstaja in je novejša od STL datoteke, če je tako, jo preberemo in vrnemo oblak točk
        if not pcd.is_empty():
            pcd.paint_uniform_color([1.0, 0.706, 0.0])
            return pcd

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))    # preberemo STL datoteko in ustvarimo mrežo trikotnikov
    if mesh.is_empty():     # preverimo, če je mreža prazna (če STL datoteka ne obstaja ali je poškodovana)
        raise FileNotFoundError(f"CAD mesh not found: {mesh_path}")
    mesh.compute_vertex_normals()   # izračunamo normale za vsako točko v mreži, kar je potrebno za registracijo
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)  # vzorčenje točk iz mreže s Poisson disk metodo, da dobimo oblak točk

    if use_cache:
        o3d.io.write_point_cloud(str(cache_path), pcd)

    pcd.paint_uniform_color([1.0, 0.706, 0.0])
    return pcd


def load_cad_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Loads the CAD mesh as a triangle mesh (not sampled to points) -
    needed by simulate_camera_scan_raycast, which raycasts directly against
    the mesh surface rather than an already-discretized point cloud, so
    occlusion is exact (a ray either hits the nearest triangle or it
    doesn't) instead of approximated from a finite point sample."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise FileNotFoundError(f"CAD mesh not found: {mesh_path}")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# Simulirani transformacijski matrika, ki se uporablja za generiranje simuliranega skeniranja iz CAD modela
SIMULATED_TRANSFORM = np.array([
    [0.866, -0.500, 0.0, 10.0],     # Da dobimo rotacijo zmnožimo matrike rotacij okoli posameznih osi
    [0.500, 0.866, 0.0, 20.0],      # Zadnji stolpec je translacija v x, y in z smereh
    [0.0, 0.0, 1.0, 170.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Stara, na točkah temelječa simulacija (hidden_point_removal) - ohranjena
# samo za primerjavo/referenco. Privzeto se za simulirano skeniranje zdaj
# uporablja simulate_camera_scan_raycast (glej spodaj), ki namesto point-
# based okluzijske hevristike (kateri radius parameter je bilo treba ročno
# uglaševati glede na razdaljo kamere, in ki je pri realističnih ~1m
# razdaljah izgubljala večino resnično vidnih točk - glej PREPROCESS_CACHE_VERSION
# komentar zgoraj) izvaja pravo ray-triangle presekanje na dejanski CAD mreži:
# okluzija je geometrijsko točna (žarek zadane najbližji trikotnik ali ne),
# FOV/ločljivost sta prava parametra kamere (ne implicitna/nemodelirana), in
# na voljo je pravi kot vpadanja žarka za vsako točko (za realistične izpade
# pri robnih/strmih kotih, ki jih realni triangulacijski senzorji dejansko imajo).
def simulate_camera_scan_legacy_hpr(source: o3d.geometry.PointCloud,           # izvorni oblak točk
                         transform: np.ndarray | None = None,       # transformacijska matrika
                         noise_std: float = 0.1,                    # standardni odklon Gaussovega šuma, ki se doda točkam, za realizem
                         camera_location: Optional[np.ndarray] = None,   # lokacija kamere, ki se uporablja za odstranjevanje skritih točk - če None, se izračuna dinamično nad transformiranim oblakom (glej spodaj)
                         up_axis: int = 2,                          # os, ki predstavlja "gor" - kamera je vedno nad delom vzdolž te osi, nikoli spodaj
                         flip_up_direction: bool = False,           # če True, je kamera na nasprotni (-up_axis) strani - glej crop_top_region()
                         radius: Optional[float] = None,            # radij za odstranjevanje skritih točk (če ni podan, se izračuna glede na velikost oblaka točk)
                         disable_occlusion: bool = True) -> o3d.geometry.PointCloud:   # če je True, se preskoči odstranjevanje skritih točk, da se ohrani celoten oblak točk, za testiranje vpliva delne vidljivosti na registracijo
    if transform is None:       # če transformacija ni podana, uporabimo simulirano transformacijo
        transform = SIMULATED_TRANSFORM
    target = copy.deepcopy(source)      # ne spreminjamo originala
    target.transform(transform)         # uporabimo transformacijsko matriko na izvorni oblak točk, da dobimo ciljni oblak točk

    if camera_location is None:
        # A fixed (0,0,0) camera location only makes sense if the transformed
        # object happens to end up "above" the origin - SIMULATED_TRANSFORM
        # translates it away from the origin, which puts a (0,0,0) camera on
        # the wrong side, so hidden_point_removal reveals the bottom-facing
        # surface instead of the top. The real robot's camera is always
        # positioned above the part, regardless of where the part actually
        # sits - so derive the camera location from target's own
        # (post-transform) position, same mechanism already used for normal
        # orientation.
        camera_location = top_camera_location(target, up_axis=up_axis, flip_up_direction=flip_up_direction)

    if disable_occlusion:
        # Če damo disable_occlusion=True, preskočimo odstranjevanje skritih točk in samo dodamo šum, da ohranimo celoten oblak točk.
        # S tem lahko testiramo, ali je delna vidljivost (occlusion) tisti dejavnik, ki povzroča odstopanje dobljene transformacije od referenčne.
        points = np.asarray(target.points)
        points += np.random.normal(scale=noise_std, size=points.shape)
        target.points = o3d.utility.Vector3dVector(points)
        return target

    if radius is None:
        # Če radij ni podan, ga izračunamo glede na velikost oblaka točk.
        # Uporabimo normo razlike med največjo in najmanjšo mejo oblaka točk, da dobimo premer, nato pa ga pomnožimo s faktorjem (100.0), da dobimo radij.
        # To je standardni pristop za določanje radija, ki zajema celoten oblak točk, da se odstrani skrite točke.
        diameter = np.linalg.norm(
            np.asarray(target.get_max_bound()) - np.asarray(target.get_min_bound()))
        radius = diameter * 100.0

    _, visible_indices = target.hidden_point_removal(camera_location, radius)  # ta funkcija zračuna, katere točke so vidne
    target = target.select_by_index(visible_indices)  # izberemo samo vidne točke, da dobimo oblak točk, ki je videti iz kamere
    points = np.asarray(target.points)  # te tri vrstice so namenjene za dodajanje šuma
    points += np.random.normal(scale=noise_std, size=points.shape)
    target.points = o3d.utility.Vector3dVector(points)
    return target


def build_camera_intrinsic(width_px: int, height_px: int, fov_deg: float) -> np.ndarray:
    """Standardni pinhole intrinsic matrix za simetričen horizontalni FOV.

    fov_deg naj pride iz datasheeta kandidatne kamere (pogosto podan
    neposredno kot "Field of View (H)"), ali pa se izračuna iz znane
    delovne razdalje in specificirane pokritosti (footprint) pri tisti
    razdalji:
        fov_deg = 2 * degrees(atan((footprint_mm / 2) / working_distance_mm))
    """
    fx = fy = (width_px / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx, cy = width_px / 2.0, height_px / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def build_camera_extrinsic(camera_location: np.ndarray, forward_direction: np.ndarray) -> np.ndarray:
    """4x4 world-to-camera extrinsic matrika (OpenCV konvencija: X desno, Y
    dol, Z naprej) za kamero na camera_location, ki gleda vzdolž
    forward_direction (v world koordinatah, ni nujno normaliziran).

    Namenoma NI "look-at" (kamera ne sledi delu, kamorkoli se ta premakne)
    - uporablja rotation_aligning_vectors, da dobi minimalno rotacijo, ki
    OpenCV-jevo privzeto smer naprej (+Z) preslika v forward_direction, kar
    ustreza fizično pravilnemu modelu: prava kamera je pritrjena in vedno
    gleda v isto smer (navzdol vzdolž up_axis), ne glede na to, kje na mizi
    del dejansko pristane.
    """
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    R_cam_to_world = rotation_aligning_vectors(np.array([0.0, 0.0, 1.0]), forward_direction)
    R_world_to_cam = R_cam_to_world.T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_world_to_cam
    extrinsic[:3, 3] = -R_world_to_cam @ camera_location
    return extrinsic


def simulate_camera_scan_raycast(mesh: o3d.geometry.TriangleMesh,
                                 transform: Optional[np.ndarray] = None,
                                 camera_location: Optional[np.ndarray] = None,
                                 up_axis: int = 2,
                                 flip_up_direction: bool = False,
                                 width_px: int = 1280,
                                 height_px: int = 960,
                                 fov_deg: float = 25.0,
                                 max_incidence_deg: float = 75.0,
                                 depth_noise_at_1m: float = 0.15,
                                 noise_distance_power: float = 2.0,
                                 working_distance: Optional[float] = None) -> o3d.geometry.PointCloud:
    """Fizikalno utemeljena zamenjava za hidden_point_removal-based okluzijo:
    za vsak piksel resničnega pinhole kamera modela požene žarek skozi
    dejansko CAD mrežo (ne skozi vzorčeno/točkovno aproksimacijo), kar da:

      - geometrijsko točno okluzijo (najbližji presek žarka s trikotnikom) -
        brez radius parametra za uglaševanje, brez artefakta izgube točk z
        razdaljo, ki ga je imela hidden_point_removal (empirično potrjeno:
        pri 1m razdalji je HPR z radij=premer_dela*100 ohranil samo ~18%
        resnično vidnih točk, medtem ko pravilno umerjen radij pokaže ~58%,
        raycasting pa je po konstrukciji točen - ni več "kako umeriti radij").
      - FOV kot pravi, na datasheet kamere vezan parameter (width_px,
        height_px, fov_deg) namesto nemodeliranega - del ali področje izven
        vidnega stožca preprosto ni zadeto, tako kot pri pravem senzorju.
      - ločljivost, vezano na dejansko število slikovnih pik senzorja, ne na
        poljubno Poisson-disk gostoto, ki nima zveze z nobeno realno kamero.
      - kot vpadanja žarka (med žarkom in lokalno normalo površine) za vsako
        zadeto točko - omogoča realističen izpad pri robnih/strmih kotih
        (glej max_incidence_deg) in realistično, od razdalje odvisno globinsko
        šum (glej depth_noise_at_1m), kar je bila pri hidden_point_removal
        popolnoma nemodelirana dimenzija.

    Kamera je NAMENOMA fiksno usmerjena navzdol vzdolž up_axis (ne "look-at"
    proti delu) - to ustreza pravi pritrjeni kameri, ki se ne obrača glede na
    to, kam del pristane na mizi (enaka predpostavka, ki jo že uporablja
    top_camera_location/crop_top_region povsod drugje v tej kodi).

    max_incidence_deg: realni triangulacijski/strukturirano-svetlobni 3D
    senzorji zanesljivo izgubijo globino, ko se površina preveč nagne stran
    od kamere (bleščanje, senčenje strukturiranega vzorca, prevelik kot med
    izvorom svetlobe in senzorjem) - privzetih 75 deg je smiseln začetni
    približek, prilagodi glede na datasheet kandidatne kamere, če ta podaja
    "maximum surface angle" ali podobno specifikacijo.

    depth_noise_at_1m, noise_distance_power: globinski šum se doda VZDOLŽ
    žarka (ne izotropno v XYZ, kot je delala stara funkcija), ker je tako
    dejansko strukturiran šum triangulacijskih senzorjev - resnično se meri
    razdalja vzdolž optične osi/žarka. Amplituda šuma pri triangulacijskih
    senzorjih (stereo, strukturirana svetloba) narašča približno s KVADRATOM
    razdalje (šum ~ razdalja^2 / (baseline * goriščna_razdalja)), ne
    konstantno kot v stari kodi - noise_distance_power=2.0 je ta privzeta
    fizikalna oblika; depth_noise_at_1m naj pride iz kandidatne kamere
    datasheeta ("depth accuracy/repeatability @ 1m"), če je na voljo -
    privzeta vrednost 0.15mm je le okvirna ocena za industrijsko kamero
    srednjega razreda in NI nadomestilo za pravi podatek.
    """
    if transform is None:
        transform = SIMULATED_TRANSFORM

    mesh_t = copy.deepcopy(mesh)
    mesh_t.transform(transform)
    mesh_t.compute_triangle_normals()

    if camera_location is None:
        # Enaka logika kot pri stari funkciji: kamera se postavi nad
        # dejanski (transformirani) položaj dela, če eksplicitna lokacija
        # (pritrjena kamera) ni podana.
        camera_location = top_camera_location(
            o3d.geometry.PointCloud(mesh_t.vertices), up_axis=up_axis, flip_up_direction=flip_up_direction,
            working_distance=working_distance)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_t))

    forward = np.zeros(3)
    forward[up_axis] = -1.0 if not flip_up_direction else 1.0

    extrinsic = build_camera_extrinsic(camera_location, forward)
    intrinsic = build_camera_intrinsic(width_px, height_px, fov_deg)

    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(intrinsic),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=width_px, height_px=height_px)
    ans = scene.cast_rays(rays)

    t_hit = ans['t_hit'].numpy()
    hit_mask = np.isfinite(t_hit)
    if not np.any(hit_mask):
        print("  OPOZORILO: noben žarek ni zadel mreže - preveri camera_location/fov_deg/working_distance "
              "(kamera verjetno sploh ne gleda proti delu)")
        return o3d.geometry.PointCloud()

    rays_np = rays.numpy()
    origins, dirs = rays_np[..., :3], rays_np[..., 3:]
    normals = ans['primitive_normals'].numpy()

    hit_origins = origins[hit_mask]
    hit_dirs = dirs[hit_mask]
    hit_t = t_hit[hit_mask]
    hit_normals = normals[hit_mask]

    # Kot vpadanja: kot med vpadnim žarkom in lokalno normalo površine.
    cos_incidence = np.abs(np.einsum('ij,ij->i', hit_dirs, hit_normals))
    incidence_deg = np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))
    keep = incidence_deg <= max_incidence_deg
    n_dropped_incidence = (~keep).sum()
    hit_origins, hit_dirs, hit_t = hit_origins[keep], hit_dirs[keep], hit_t[keep]

    points = hit_origins + hit_t[:, None] * hit_dirs

    # Globinski šum vzdolž žarka, kvadratno naraščajoč z razdaljo (glej
    # docstring zgoraj). Predpostavlja mm enote, kot preostanek te kode.
    distance_m = hit_t / 1000.0
    noise_std = depth_noise_at_1m * (distance_m ** noise_distance_power)
    depth_noise = np.random.normal(scale=noise_std)
    points = points + hit_dirs * depth_noise[:, None]

    print(f"  Raycast scan: {width_px}x{height_px} px, fov={fov_deg} deg -> "
          f"{hit_mask.sum()} zadetkov mreže, {n_dropped_incidence} izločenih "
          f"(kot vpadanja > {max_incidence_deg} deg), {len(points)} končnih točk")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# za nalaganje realnega skeniranja iz datoteke, za pol k bomo realno skeniral
def load_real_scan(scan_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(scan_path))
    if pcd.is_empty():
        raise FileNotFoundError(f"Real scan not found: {scan_path}")
    return pcd

# za filtriranje oblaka točk, da odstranimo statistične odstopanja (outlierje),
# odstrani napake pri skeniranje, ko je kakšna točka ki močno odstopa jo odstani
def filter_scan(pcd: o3d.geometry.PointCloud,
                nb_neighbors: int = 30,
                std_ratio: float = 1.5) -> o3d.geometry.PointCloud:
    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered

# pravilno orientera normalne vektorje v oblaku točk, da so skladni z oblakomточk
def ensure_oriented_normals(pcd: o3d.geometry.PointCloud,  # oblak točk, ki ga obdelujemo
                            normal_radius: float,   # radij za izračun normalnih vektorjev, določa velikost lokalnega območja okoli vsake točke, ki se uporablja za izračun normale na površino
                            is_partial_view: bool,  # če je True, pomeni, da oblak točk predstavlja delno vidno površino objekta (npr. skeniranje iz ene kamere), če je False, pomeni, da oblak točk predstavlja celotno zaprto površino objekta (npr. CAD model)
                            camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),    # lokacija kamere, ki se uporablja za orientacijo normalnih vektorjev, če je is_partial_view=True
                            max_nn: int = 30,   # maksimalno število sosednjih točk, ki se upoštevajo pri izračunu normalne za vsako točko, večje število pomeni bolj gladke normale, vendar počasnejše izračune
                            use_knn: bool = False) -> None:     # Če je True, se uporabi KNN (k-nearest neighbors) metoda za izračun normalnih vektorjev, sicer se uporabi hibridna metoda z radijem
                                                                # False --> enakomerni oblaki točk, True --> neenakomerni oblaki točk
    if use_knn:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))     # normalo oceni glede na najbližje točke, ne glede na njihovo oddaljenost
    else:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))   # normalo oceni glede na točke v določeni bližini

    if is_partial_view:
        # A single camera view (real or simulated) only ever sees one side
        # of the object, so every normal should point back toward the
        # sensor.
        pcd.orient_normals_towards_camera_location(camera_location)
    else:
        # The CAD model is a full, closed surface - there is no single
        # "camera side" to orient towards, so propagate a globally
        # consistent orientation across the whole surface instead. This
        # also sidesteps any inconsistent winding baked into the STL.
        pcd.orient_normals_consistent_tangent_plane(max_nn)


def crop_top_region(pcd: o3d.geometry.PointCloud,  # vhodni oblak točk, ki ga želimo odrezat
                    top_fraction: float = 0.35,     # obdrži 35% zgornjega dela oblaka točk
                    up_axis: int = 2,               # os, ki predstavlja 'gor' (0=x, 1=y, 2=z), privzeto je z os
                    flip_up_direction: bool = False) -> o3d.geometry.PointCloud:
    min_bound = pcd.get_min_bound()         # poišče skrajne točke oblaka točk
    max_bound = pcd.get_max_bound()
    if not flip_up_direction:               # sam izračuna kok je treba odrezat
        cutoff = max_bound[up_axis] - top_fraction * (max_bound[up_axis] - min_bound[up_axis])
        crop_min = min_bound.copy()
        crop_min[up_axis] = cutoff
        bbox = o3d.geometry.AxisAlignedBoundingBox(crop_min, max_bound)
    else:                               # isto sam obratno, če je flip_up_direction=True
        cutoff = min_bound[up_axis] + top_fraction * (max_bound[up_axis] - min_bound[up_axis])
        crop_max = max_bound.copy()
        crop_max[up_axis] = cutoff
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, crop_max)
    return pcd.crop(bbox)

# TO JE SINTETIČNA KAMERA, KI JO BOMO POTREBOVAL TUDI PRI REALNEM SKENIRANJE
# NAMEN JE, DA SE NORMALE OBLAKA TOČK CAD MODELA OBRNEJO V ISTO SMER KOT BODO OD SKENIRANGA POINT CLOUDA
# ČE TE FUNKCIJE NI SO NORMALE OBLAKA TOČK SOURCE MODELA OBRNJENE V NAPAČNO SMER ( NE MORE PRIMERJAT S TARGET MODELOM)
def top_camera_location(pcd: o3d.geometry.PointCloud,  # vhodni oblak točk, ki ga želimo uporabiti za določitev lokacije kamere
                        up_axis: int = 2,   # os, ki predstavlja "gor" (0=x, 1=y, 2=z), privzeto je z os
                        offset_factor: float = 5.0,     # faktor, ki določa, kako daleč nad oblakom točk bo kamera postavljena (v enotah dolžine oblaka točk) - uporabljen samo, če working_distance ni podan
                        flip_up_direction: bool = False,
                        working_distance: Optional[float] = None) -> np.ndarray:
    # working_distance: eksplicitna, resnična razdalja kamera-do-dela (npr.
    # iz datasheeta kandidatne kamere), v istih enotah kot oblak točk (mm).
    # Kadar je podana, ima prednost pred offset_factor*extent hevristiko -
    # ta hevristika je bila zasnovana samo za "dovolj daleč, da je normalna
    # orientacija nedvoumna" (kjer natančna razdalja ni pomembna), NE za
    # fizikalno natančno simulacijo kamere (kjer razdalja neposredno vpliva
    # na FOV/pokritost pri raycastingu spodaj) - glej simulate_camera_scan_raycast.
    center = pcd.get_center()           # prostorska sredina oblaka točk
    extent = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())     # razlika med največjo in najmanjšo mejo oblaka točk, da dobimo velikost oblaka točk
    camera_location = np.array(center)
    offset = working_distance if working_distance is not None else offset_factor * max(extent[up_axis], 1.0)
    if not flip_up_direction:
        camera_location[up_axis] = pcd.get_max_bound()[up_axis] + offset
    else:
        camera_location[up_axis] = pcd.get_min_bound()[up_axis] - offset
    return camera_location


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float,
                           is_partial_view: bool = False,
                           camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),
                           cache_key: Optional[str] = None,
                           use_cache: bool = True) -> o3d.geometry.PointCloud:
    """Voxel-downsamples pcd and computes correctly-oriented normals -
    everything yaw_sweep_registration and the multi-scale ICP refinement
    actually need. (An earlier version of this pipeline also built ISS
    keypoints and FPFH descriptors here for FPFH+RANSAC global registration;
    that path was dropped once yaw_sweep_registration proved both simpler
    and far more accurate for this part, since the part's real placement
    constraint - it always rests on the table the same way - makes blind
    FPFH-based 6-DOF matching solve a harder problem than the real one.)

    cache_key identifies the *input* (e.g. "source_partname_20000pts_top0.35_axis2") -
    combined with every parameter below that affects the output, it forms
    a cache path.
    """
    cache_path = None
    if cache_key is not None and use_cache:
        params = (PREPROCESS_CACHE_VERSION, voxel_size, is_partial_view,
                  tuple(np.round(np.asarray(camera_location), 3)))
        digest = hashlib.md5(repr(params).encode()).hexdigest()[:10]
        PREPROCESS_CACHE_DIR.mkdir(exist_ok=True)
        # cache_key is human-readable (e.g. "source_partname_20000pts_top0.35_axis2")
        # but can get long - the CAD filename alone plus every test-tilt/
        # test-translation tag appended in run_registration's callers can
        # push the full path past Windows' 260-char MAX_PATH, which makes
        # Open3D's write_point_cloud fail with "invalid wchar_t filename
        # argument" (not an informative error about *why*). Hashing
        # cache_key itself keeps the filename short and length-independent
        # of how descriptive cache_key gets; a truncated human-readable
        # prefix is kept alongside purely for browsing .preprocess_cache/
        # by eye, not for uniqueness (the hash guarantees that).
        cache_key_digest = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        readable_prefix = "".join(c if c.isalnum() else "_" for c in cache_key)[:40]
        cache_path = PREPROCESS_CACHE_DIR / f"{readable_prefix}_{cache_key_digest}_{digest}"
        cloud_path = cache_path.parent / (cache_path.name + ".ply")
        if cloud_path.exists():
            pcd_down = o3d.io.read_point_cloud(str(cloud_path))
            if not pcd_down.is_empty():
                print(f"    Loaded cached preprocessing for '{cache_key}': {len(pcd_down.points)} points")
                return pcd_down

    extent = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diagonal = np.linalg.norm(extent)
    print(f"    Input cloud extent (xyz): {extent}, diagonal={diagonal:.3f}, "
          f"voxel_size={voxel_size} -> ~{diagonal / voxel_size:.0f} voxels across the diagonal")

    pcd_down = pcd.voxel_down_sample(voxel_size)    # naredi 3d mrežo kock, v vsaki kocki vzame eno točko (povprečje), da zmanjša število točk in pospeši izračune

    ensure_oriented_normals(pcd_down, normal_radius=voxel_size * 2.0, is_partial_view=is_partial_view,
                            camera_location=camera_location) # pokliče funkcijo od prej

    if cache_path is not None:
        o3d.io.write_point_cloud(str(cache_path.parent / (cache_path.name + ".ply")), pcd_down)

    return pcd_down


def estimate_surface_normal(points: np.ndarray) -> np.ndarray:
    """PCA-based normal estimate: the direction the points vary *least*
    along. Robust to noise/occlusion since it's a least-squares fit over
    every point, not a single extremal one - unlike picking the single
    topmost point, a handful of noisy/occluded points barely move the
    result."""
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # eigh sorts ascending - smallest eigenvalue first
    return normal / np.linalg.norm(normal)


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation formula: a rotation matrix mapping unit vector a
    onto unit vector b (the minimal-angle rotation that does so)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # a and b are anti-parallel - any axis perpendicular to a works
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotation by angle_deg about an
    arbitrary unit axis (generalizes a pure-Z rotation to any axis, once
    tilt estimation means the "up" axis isn't exactly Z anymore)."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def yaw_sweep_registration(source_down: o3d.geometry.PointCloud,
                           target_down: o3d.geometry.PointCloud,
                           voxel_size: float,
                           up_axis: int = 2,
                           yaw_step_deg: float = 3.0,
                           distance_threshold_factor: float = 3.0,
                           refine_iterations: int = 5) -> o3d.pipelines.registration.RegistrationResult:
    """Coarse global registration exploiting the *real* placement constraint
    instead of solving blind 6-DOF registration: the part always rests on
    the table in the same face-down orientation, so Z/roll/pitch are fixed
    by table contact, X/Y are only loosely known (tens of mm), and yaw
    (rotation about up_axis) is the only genuinely free parameter.

    Rotation about up_axis through a point cloud's own centroid doesn't
    move that centroid's coordinates, so translation is estimated directly
    from source/target centroids - independent of the unknown yaw, no
    search needed for it. Yaw itself is swept exhaustively (not randomly
    sampled) since it's a single bounded, fully-enumerable parameter.

    Real placement is never *perfectly* flat though (a stray chip of
    debris, a slight warp), so this also estimates and corrects actual
    roll/pitch tilt instead of assuming exactly zero: target's surface
    normal is estimated via PCA (the direction its points vary least
    along - robust, since it's a least-squares fit over every point, not
    one extremal point), and the rotation aligning source's known
    reference "up" direction to that estimated normal becomes the base
    orientation. What's left over - rotation *about* that now-aligned
    normal axis - is exactly the yaw ambiguity, so the sweep still runs,
    just composed on top of the tilt correction instead of a flat
    assumption. Tested empirically to hold up to ~25-27 deg of injected
    tilt before breaking down around 30 deg.

    Each coarse yaw candidate gets a *quick* local ICP polish
    (refine_iterations, kept low - this is not the final precision pass,
    the full multi-scale ICP after this function does that) before scoring,
    rather than scoring the raw unrefined snapshot. This matters a lot on a
    part with repeated features (mounting bosses): a wrong yaw can align
    the repeated bulk geometry well enough to win on raw inlier *count*
    alone, even though it completely misaligns the few asymmetric features
    (locator pins, connector) that actually distinguish the correct
    orientation. A short local refinement snaps each candidate toward its
    true nearby optimum first - the correct yaw converges to a tight fit,
    a merely bulk-aligned wrong yaw generally doesn't, because the
    asymmetric detail won't line up no matter how it's nudged locally.

    Only up_axis=2 (Z) is supported for the *reference* "up" direction
    (matching every other up-axis assumption in this codebase) - the tilt
    correction itself works for an arbitrary resulting orientation, only
    the starting reference direction is fixed to the Z axis.
    """
    if up_axis != 2:
        raise NotImplementedError("yaw_sweep_registration only supports up_axis=2 (Z) as the "
                                  "reference 'up' direction - the part's nominal resting "
                                  "orientation before tilt correction")

    source_pts = np.asarray(source_down.points)
    target_pts = np.asarray(target_down.points)
    target_centroid = target_pts.mean(axis=0)

    up_vector = np.zeros(3)
    up_vector[up_axis] = 1.0
    target_normal = estimate_surface_normal(target_pts)
    if np.dot(target_normal, up_vector) < 0:
        target_normal = -target_normal  # PCA normal has a sign ambiguity - pick the "up-ish" one
    tilt_deg = np.degrees(np.arccos(np.clip(np.dot(up_vector, target_normal), -1.0, 1.0)))
    tilt_correction = rotation_aligning_vectors(up_vector, target_normal)
    print(f"  Estimated tilt from target's PCA surface normal: {tilt_deg:.2f} deg off {up_axis}-axis")

    distance_threshold = voxel_size * distance_threshold_factor

    def build_transform(yaw_deg: float) -> np.ndarray:
        # Tilt correction first (aligns source's reference "up" to target's
        # actual estimated normal), then yaw as a twist about that now-
        # correct normal axis - this is the remaining 1-DOF ambiguity the
        # sweep searches, same role as a pure-Z rotation played before tilt
        # estimation was added, just no longer assuming the normal axis is
        # exactly up_axis.
        R = rotation_about_axis(target_normal, yaw_deg) @ tilt_correction
        T = np.eye(4)
        T[:3, :3] = R
        source_pts_rotated = (R @ source_pts.T).T
        # Centroid-matching for translation is still valid for any rotation
        # (rotation about a point cloud's own centroid never moves that
        # centroid), but still occlusion-biased along up_axis specifically
        # (hidden_point_removal preferentially drops self-occluded points,
        # skewing the *visible* subset's average up_axis position) - so
        # up_axis is still overridden with the more occlusion-robust
        # max(up_axis) difference, computed on the *rotated* source
        # (source_pts_rotated) so the comparison is apples-to-apples once
        # tilt is actually accounted for, instead of comparing an untilted
        # source's max against a tilted target's max directly.
        translation = target_centroid - source_pts_rotated.mean(axis=0)
        translation[up_axis] = target_pts[:, up_axis].max() - source_pts_rotated[:, up_axis].max()
        T[:3, 3] = translation
        return T

    yaw_candidates = np.arange(0.0, 360.0, yaw_step_deg)
    print(f"  Yaw sweep: {len(yaw_candidates)} candidates every {yaw_step_deg} deg "
          f"(source={len(source_pts)} pts, target={len(target_pts)} pts), "
          f"distance_threshold={distance_threshold:.3f}, refine_iterations={refine_iterations}")

    # Point-to-point (not point-to-plane) for this quick per-candidate scoring
    # pass - doesn't depend on normal correctness, keeping this scoring step
    # simple and robust; the full multi-scale ICP afterward still does the
    # precise point-to-plane refinement on the winning candidate.
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=refine_iterations)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    best_yaw, best_fitness, best_rmse, best_transform = 0.0, -1.0, float("inf"), np.eye(4)
    for yaw in yaw_candidates:
        T_coarse = build_transform(yaw)
        icp_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold, T_coarse, estimation, criteria)
        is_better = (icp_result.fitness > best_fitness or
                    (icp_result.fitness == best_fitness and icp_result.inlier_rmse < best_rmse))
        if is_better:
            best_fitness = icp_result.fitness
            best_rmse = icp_result.inlier_rmse
            best_yaw = yaw
            best_transform = icp_result.transformation

    print(f"  Best yaw={best_yaw:.1f} deg (after {refine_iterations}-iteration local polish): "
          f"fitness={best_fitness:.4f}, inlier_rmse={best_rmse:.4f}")

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = best_transform
    result.fitness = best_fitness
    result.inlier_rmse = best_rmse if np.isfinite(best_rmse) else 0.0
    return result


def refine_registration(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        init_transformation: np.ndarray,
                        voxel_size: float,
                        icp_distance_factor: float = 2.0,
                        icp_voxel_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
                        source_camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),
                        target_camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),
                        common_eval_threshold_factor: float = 1.5,
                        icp_max_iterations: int = 100) -> o3d.pipelines.registration.RegistrationResult:
    """Coarse-to-fine (multi-scale) point-to-plane ICP.

    A single full-resolution pass struggles when source (dense - the full,
    un-occluded CAD crop) and target (sparse - limited by what the camera
    actually sees) have very different real point densities, even at the
    same nominal voxel_size: target's KNN normals end up averaged over a
    much wider physical neighborhood than source's, biasing the
    normal-projected residual point-to-plane minimizes, which can make a
    single fine pass converge to a worse local optimum than the coarse
    seed.

    Running several passes from coarse to fine helps with this - at coarse
    scales, downsampling brings source's density down close to target's
    natural density, so both clouds' normals are comparably precise - but
    it's not guaranteed to be monotonic: a later, finer stage can still end
    up worse than an earlier one if that scale's normals happen to be
    biased. Each stage's own reported fitness isn't even a fair way to
    check, since it's computed under a different threshold per stage, so
    every candidate (the coarse seed plus every stage's result) is
    evaluated under one fixed common threshold instead.

    Selection among those candidates prioritizes inlier_rmse (how tightly
    the matched points actually fit), not fitness (how many points matched)
    - a finer stage often converges to a *more* geometrically accurate pose
    while matching slightly *fewer* points under a fixed threshold (a
    tightly-converged pose can push a few marginal points just outside a
    fixed distance cutoff even as the true matched points fit far more
    precisely). Selecting by fitness alone was empirically picking a
    coarser, looser-fitting stage over a finer one that was both more
    precise (lower rmse) *and* closer to ground truth in held-out tests.
    Fitness still guards against a genuinely diverged stage: only
    candidates within fitness_tolerance of the best fitness seen are
    eligible, so a low-coverage stage can't win purely on a tiny, tight
    but unrepresentative subset.
    """
    common_threshold = voxel_size * common_eval_threshold_factor
    baseline_eval = evaluate_registration(source, target, init_transformation, common_threshold)
    print(f"  ICP baseline (coarse seed) under common threshold {common_threshold:.3f}: "
          f"fitness={baseline_eval.fitness:.4f}, inlier_rmse={baseline_eval.inlier_rmse:.4f}")
    candidates = [(init_transformation, baseline_eval.fitness, baseline_eval.inlier_rmse)]

    current_transformation = init_transformation
    for scale in sorted(icp_voxel_scales, reverse=True):
        stage_voxel_size = voxel_size * scale
        distance_threshold = stage_voxel_size * icp_distance_factor
        source_stage = source.voxel_down_sample(stage_voxel_size)
        target_stage = target.voxel_down_sample(stage_voxel_size)
        # use_knn=True for the same reason as the full-resolution recompute
        # elsewhere - target's density isn't uniform even after
        # downsampling (occlusion, not voxel size, is the limiting factor).
        ensure_oriented_normals(source_stage, normal_radius=stage_voxel_size * 2.0, is_partial_view=True,
                                camera_location=source_camera_location, use_knn=True)
        ensure_oriented_normals(target_stage, normal_radius=stage_voxel_size * 2.0, is_partial_view=True,
                                camera_location=target_camera_location, use_knn=True)
        print(f"  ICP stage voxel_size={stage_voxel_size:.3f}, distance_threshold={distance_threshold:.3f}, "
              f"source={len(source_stage.points)} pts, target={len(target_stage.points)} pts")
        result = o3d.pipelines.registration.registration_icp(
            source_stage, target_stage, distance_threshold, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations))
        print(f"    fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}")
        current_transformation = result.transformation

        stage_eval = evaluate_registration(source, target, current_transformation, common_threshold)
        print(f"    under common threshold {common_threshold:.3f}: fitness={stage_eval.fitness:.4f}, "
              f"inlier_rmse={stage_eval.inlier_rmse:.4f}")
        candidates.append((current_transformation, stage_eval.fitness, stage_eval.inlier_rmse))

    max_fitness = max(fitness for _, fitness, _ in candidates)
    fitness_tolerance = 0.9
    eligible = [c for c in candidates if c[1] >= max_fitness * fitness_tolerance]
    best_transformation, best_fitness, best_rmse = min(eligible, key=lambda c: c[2])
    print(f"  Selected transformation with fitness={best_fitness:.4f}, inlier_rmse={best_rmse:.4f} "
          f"(lowest inlier_rmse among candidates within {fitness_tolerance:.0%} of the best "
          f"fitness seen, {max_fitness:.4f})")

    best_eval = o3d.pipelines.registration.RegistrationResult()
    best_eval.transformation = best_transformation
    best_eval.fitness = best_fitness
    best_eval.inlier_rmse = best_rmse
    return best_eval


def transformation_error(estimated: np.ndarray, reference: np.ndarray) -> dict:
    """Decompose the discrepancy between an estimated and a reference 4x4
    rigid transform into a rotation angle error (degrees) and a translation
    error (same units as the point cloud), instead of a raw matrix diff that
    mixes both into numbers that are hard to interpret."""
    r_est, t_est = estimated[:3, :3], estimated[:3, 3]
    r_ref, t_ref = reference[:3, :3], reference[:3, 3]

    r_delta = r_est @ r_ref.T
    # Rotation angle from a rotation matrix's trace: trace(R) = 1 + 2*cos(theta).
    cos_theta = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error_deg = np.degrees(np.arccos(cos_theta))
    translation_error = np.linalg.norm(t_est - t_ref)

    return {
        "rotation_error_deg": rotation_error_deg,
        "translation_error": translation_error,
        "frobenius_norm": np.linalg.norm(estimated - reference),
    }


def evaluate_registration(source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          transformation: np.ndarray,
                          threshold: float) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, transformation)


def run_registration(cad_path: Path,
                     scan_path: Optional[Path] = None,
                     use_real_scan: bool = False,
                     voxel_size: float = 1.0,
                     sample_point_count: int = 200000,
                     filter_real_scan: bool = True,
                     noise_std: float = 0.1,
                     disable_occlusion: bool = False,
                     top_fraction: float = 0.35,
                     up_axis: int = 2,
                     flip_up_direction: bool = False,
                     coarse_distance_factor: float = 3.0,
                     yaw_step_deg: float = 3.0,
                     yaw_sweep_refine_iterations: int = 5,
                     icp_distance_factor: float = 2.0,
                     icp_voxel_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
                     icp_max_iterations: int = 100,
                     min_coarse_fitness_for_icp: float = 0.1,
                     use_cad_cache: bool = True,
                     use_preprocess_cache: bool = True,
                     fixed_camera_location: Optional[np.ndarray] = None,
                     camera_width_px: int = 1280,
                     camera_height_px: int = 960,
                     camera_fov_deg: float = 25.0,
                     camera_max_incidence_deg: float = 75.0,
                     camera_depth_noise_at_1m: float = 0.15,
                     camera_noise_distance_power: float = 2.0,
                     camera_working_distance: Optional[float] = None,
                     cache_key_tag: str = "") -> tuple[
                         o3d.geometry.PointCloud,
                         o3d.geometry.PointCloud,
                         o3d.pipelines.registration.RegistrationResult,
                         o3d.pipelines.registration.RegistrationResult]:
    print("Phase 1: Loading CAD model...")
    source = load_cad_model(cad_path, sample_point_count, use_cache=use_cad_cache)
    source = crop_top_region(source, top_fraction=top_fraction, up_axis=up_axis,
                             flip_up_direction=flip_up_direction)
    print(f"  Cropped CAD model to top {top_fraction:.0%} along axis {up_axis} "
          f"(flip_up_direction={flip_up_direction}) "
          f"(the seal region the camera actually scans): {len(source.points)} points remain")
    # Cropping turns source from a closed surface into a one-sided top
    # patch, same as target - orient_normals_consistent_tangent_plane's
    # closed-manifold assumption no longer holds, so source needs a virtual
    # "looking down from above" camera location instead, same as target.
    source_camera_location = top_camera_location(source, up_axis=up_axis,
                                                  flip_up_direction=flip_up_direction,
                                                  working_distance=camera_working_distance)

    print("Phase 2: Loading target scan...")
    if use_real_scan and scan_path is not None and scan_path.exists():
        target = load_real_scan(scan_path)
        if filter_real_scan:
            target = filter_scan(target)
    else:
        if disable_occlusion:
            print("  OPOZORILO: --disable_occlusion je bil specifičen za staro "
                  "hidden_point_removal simulacijo in nima učinka z raycasting simulacijo "
                  "(okluzija je pri raycastingu geometrijsko točna, ne izbirna). Če želiš "
                  "primerjati z/brez okluzije, uporabi simulate_camera_scan_legacy_hpr.")
        if noise_std != 0.1:
            print(f"  OPOZORILO: --noise_std={noise_std} je ignoriran pri raycasting simulaciji - "
                  f"uporabi --camera_depth_noise_at_1m (in --camera_noise_distance_power) namesto tega.")
        # Raycasting proti dejanski CAD mreži (ne proti source-ovem že
        # vzorčenem/odrezanem point cloudu) - potrebuje polno mrežo, saj
        # crop_top_region na source-u ni relevanten za to, kaj kamera
        # dejansko vidi (FOV + kot vpadanja to naravno omejita sama).
        cad_mesh = load_cad_mesh(cad_path)
        target = simulate_camera_scan_raycast(
            cad_mesh, camera_location=fixed_camera_location,
            up_axis=up_axis, flip_up_direction=flip_up_direction,
            width_px=camera_width_px, height_px=camera_height_px, fov_deg=camera_fov_deg,
            max_incidence_deg=camera_max_incidence_deg,
            depth_noise_at_1m=camera_depth_noise_at_1m,
            noise_distance_power=camera_noise_distance_power,
            working_distance=camera_working_distance)
        if len(target.points) == 0:
            raise RuntimeError("Raycasting scan returned 0 points - camera does not see the part "
                              "at all with the current camera_location/fov_deg/working_distance. "
                              "Check --camera_fov_deg and the working distance used to derive the "
                              "camera location.")
        print(f"  Simulated (raycast) target has {len(target.points)} points "
              f"(source reference has {len(source.points)})")

    if fixed_camera_location is not None:
        # The real robot's camera is bolted in place - it does not move just
        # because the part landed somewhere else on the table. Re-deriving
        # the camera location from target's own (possibly offset) position,
        # as the dynamic fallback below does, silently cancels out whatever
        # X/Y placement error we're trying to measure tolerance for.
        target_camera_location = fixed_camera_location
    else:
        # target's own physical position (real scan) or wherever
        # SIMULATED_TRANSFORM placed it (simulated scan) - NOT a fixed world
        # origin. A hardcoded (0,0,0) camera location can end up on the wrong
        # side of the object once it's translated away from the origin, which
        # silently flips all of target's normals backwards relative to source's.
        target_camera_location = top_camera_location(target, up_axis=up_axis,
                                                      flip_up_direction=flip_up_direction,
                                                      working_distance=camera_working_distance)

    print("Phase 3: Preprocessing point clouds...")
    # Cache keys identify the *input* (everything upstream that determines
    # source/target's points before preprocessing even runs); preprocess_point_cloud
    # itself folds in every parameter that affects its own output on top of this.
    source_cache_key = (f"source_{cad_path.stem}_{sample_point_count}pts_top{top_fraction}_"
                       f"axis{up_axis}_flip{flip_up_direction}")
    if use_real_scan and scan_path is not None and scan_path.exists():
        target_cache_key = f"target_real_{scan_path.stem}_filter{filter_real_scan}"
    else:
        # cache_key_tag folds in anything that perturbs SIMULATED_TRANSFORM
        # itself (test tilt/translation injected in main()) but isn't
        # otherwise reflected in these parameters - without it, two
        # different perturbations that happen to produce the same
        # fixed_camera_location would collide on the same cache entry and
        # silently serve one test's cached target to another.
        # camera_width_px/height_px/fov_deg/max_incidence_deg/depth_noise_at_1m/
        # noise_distance_power/camera_working_distance vsi neposredno vplivajo
        # na target-ove surove točke (raycasting), zato morajo biti del cache
        # ključa - drugače bi sprememba nastavitev kamere tiho postregla star,
        # neveljaven cache zapis namesto ponovnega izračuna.
        target_cache_key = (f"target_sim_raycast_{cad_path.stem}_top{top_fraction}_"
                            f"axis{up_axis}_flip{flip_up_direction}_"
                            f"cam{camera_width_px}x{camera_height_px}_fov{camera_fov_deg}_"
                            f"inc{camera_max_incidence_deg}_noise{camera_depth_noise_at_1m}_"
                            f"pow{camera_noise_distance_power}_wd{camera_working_distance}"
                            f"{cache_key_tag}")

    # source = top-cropped CAD model (one-sided patch, same as target - not
    # a closed surface anymore, see source_camera_location above)
    # target = a one-sided camera view (real or simulated)
    source_down = preprocess_point_cloud(
        source, voxel_size, is_partial_view=True, camera_location=source_camera_location,
        cache_key=source_cache_key, use_cache=use_preprocess_cache)
    target_down = preprocess_point_cloud(
        target, voxel_size, is_partial_view=True, camera_location=target_camera_location,
        cache_key=target_cache_key, use_cache=use_preprocess_cache)

    print("Phase 4: Global registration (yaw sweep)...")
    # Exploits the real placement constraint (table contact fixes Z/roll/
    # pitch, only yaw is free, and actual tilt away from that gets estimated
    # and corrected too) instead of solving blind 6-DOF registration - see
    # yaw_sweep_registration()'s docstring.
    coarse_result = yaw_sweep_registration(
        source_down, target_down, voxel_size,
        up_axis=up_axis,
        yaw_step_deg=yaw_step_deg,
        distance_threshold_factor=coarse_distance_factor,
        refine_iterations=yaw_sweep_refine_iterations)
    print(f"  Yaw sweep fitness: {coarse_result.fitness:.4f}, "
          f"inlier_rmse: {coarse_result.inlier_rmse:.4f}")
    if not use_real_scan:
        err = transformation_error(coarse_result.transformation, SIMULATED_TRANSFORM)
        print(f"  Yaw sweep vs reference transform: "
              f"rotation_error={err['rotation_error_deg']:.2f} deg, "
              f"translation_error={err['translation_error']:.3f}, "
              f"frobenius_norm={err['frobenius_norm']:.3f}")
    draw_registration_result(source, target, coarse_result.transformation, "Yaw Sweep Result")

    # Normals for ICP are computed per-stage inside refine_registration's
    # multi-scale loop (each stage re-downsamples and needs normals matching
    # *that* stage's density), so no separate full-resolution recompute is
    # needed here.

    if coarse_result.fitness < min_coarse_fitness_for_icp:
        # Point-to-plane ICP linearizes around the initial guess and assumes
        # it's already close - handing it a coarse seed this bad (usually
        # tens+ of degrees of rotation error) doesn't just fail to converge,
        # it can diverge to nonsense (transforms with meter/kilometer-scale
        # translations). Skip it rather than produce a misleading "final"
        # matrix; the real problem to fix is the coarse stage's fitness, not ICP.
        print(f"  SKIPPING ICP: coarse fitness {coarse_result.fitness:.4f} is below "
              f"--min_coarse_fitness_for_icp={min_coarse_fitness_for_icp}. ICP can only "
              f"refine a seed that's already roughly correct - feeding it this one is "
              f"likely to diverge, not improve it.")
        icp_result = coarse_result
    else:
        # refine_registration tracks the best-scoring candidate across the
        # coarse seed and every ICP stage under one common threshold (see
        # its docstring), so icp_result is guaranteed at least as good as
        # coarse_result here - no separate post-hoc reject check needed.
        icp_result = refine_registration(source, target, coarse_result.transformation, voxel_size,
                                         icp_distance_factor=icp_distance_factor,
                                         icp_voxel_scales=icp_voxel_scales,
                                         source_camera_location=source_camera_location,
                                         target_camera_location=target_camera_location,
                                         icp_max_iterations=icp_max_iterations)
        print(f"  ICP fitness: {icp_result.fitness:.4f}, "
              f"inlier_rmse: {icp_result.inlier_rmse:.4f}")

    draw_registration_result(source, target, icp_result.transformation, "ICP Result")
    print("Final transformation matrix:")
    print(icp_result.transformation)

    if not use_real_scan:
        print(f"Reference transformation matrix (used to generate the simulated scan):")
        print(SIMULATED_TRANSFORM)
        err = transformation_error(icp_result.transformation, SIMULATED_TRANSFORM)
        print(f"ICP vs reference transform: "
              f"rotation_error={err['rotation_error_deg']:.2f} deg, "
              f"translation_error={err['translation_error']:.3f}, "
              f"frobenius_norm={err['frobenius_norm']:.3f}")

    evaluation_before = evaluate_registration(source, target, coarse_result.transformation, voxel_size * 1.5)
    evaluation_after = evaluate_registration(source, target, icp_result.transformation, voxel_size * 1.5)
    print("Evaluation before ICP:", evaluation_before)
    print("Evaluation after ICP:", evaluation_after)

    return source, target, coarse_result, icp_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAD-to-scan point cloud registration")
    parser.add_argument("--cad", default="eHDS S Housing + CC s pottingom, fine.STL",
                        help="Path to the CAD mesh file")
    parser.add_argument("--scan", default="scan.ply",
                        help="Path to the real scan point cloud file")
    parser.add_argument("--use_real_scan", action="store_true",
                        help="Use a real scan instead of a simulated scan")
    parser.add_argument("--voxel_size", type=float, default=1.0,
                        help="Voxel size for downsampling")
    parser.add_argument("--sample_points", type=int, default=200000,
                        help="Number of Poisson-disk sample points to generate from the "
                             "*whole* CAD mesh (before crop_top_region keeps ~35%% and "
                             "simulated occlusion keeps a further fraction). Default 200000 "
                             "puts the realistically-visible region at ~13k points, i.e. "
                             "sub-mm point spacing over a ~150mm part - the density a real "
                             "industrial structured-light/laser scanner delivers. Denser "
                             "raw input also averages out more sensor noise per voxel for "
                             "free. First run Poisson-samples for a few minutes, then the CAD "
                             "cache makes it instant; raise this further if your real scanner "
                             "is denser (memory scales linearly, ~a few MB per 100k in cache)")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Std dev of Gaussian noise added to the simulated scan")
    parser.add_argument("--disable_occlusion", action="store_true",
                        help="Skip hidden_point_removal so the simulated scan keeps full "
                             "coverage, to check whether partial-view occlusion is what's "
                             "causing the recovered transform to diverge from the reference")
    parser.add_argument("--test_tilt_deg", type=float, default=0.0,
                        help="Add this much extra rotation (degrees) about a horizontal axis "
                             "to the ground-truth SIMULATED_TRANSFORM, to empirically test how "
                             "much roll/pitch tilt (part not lying perfectly flat on the table) "
                             "the pipeline can tolerate. yaw_sweep_registration estimates and "
                             "corrects tilt from target's PCA surface normal, tested to hold up "
                             "to ~25-27 deg before breaking down around 30. 0 = no tilt "
                             "(matches the real placement assumption exactly)")
    parser.add_argument("--test_tilt_axis", type=int, default=0, choices=[0, 1],
                        help="Which horizontal axis --test_tilt_deg rotates about: 0=X (roll), "
                             "1=Y (pitch). Only meaningful if --test_tilt_deg is nonzero")
    parser.add_argument("--test_translation_x", type=float, default=0.0,
                        help="Add this much extra X translation (mm) to the ground-truth "
                             "SIMULATED_TRANSFORM's nominal placement, to empirically test how "
                             "far off-center (within the camera's fixed field of view) the part "
                             "can sit on the table and still be registered correctly. When this "
                             "or --test_translation_y is nonzero, the simulated camera location "
                             "is fixed at the *nominal* (pre-offset) placement instead of being "
                             "re-derived from the part's actual position, matching a real camera "
                             "bolted in place above the table rather than one that tracks the part")
    parser.add_argument("--test_translation_y", type=float, default=100.0,
                        help="Same as --test_translation_x but for the Y axis")
    parser.add_argument("--top_fraction", type=float, default=0.35,
                        help="Fraction of the CAD model's height (from the top, along "
                             "--up_axis) to keep for matching, since the robot's camera "
                             "only scans the top sealing region, not the whole housing")
    parser.add_argument("--up_axis", type=int, default=2, choices=[0, 1, 2],
                        help="Axis index (0=X, 1=Y, 2=Z) along which 'top' is measured")
    parser.add_argument("--flip_up_direction", action="store_true", default=False,
                        help="Crop/camera from the low end of --up_axis instead of the high "
                             "end. The STL's own +up_axis direction doesn't necessarily match "
                             "which physical face actually faces the camera once the part is "
                             "placed on the table - if the cropped/scanned face turns out to "
                             "be the one that ends up against the table instead of facing the "
                             "camera, this is the flag to flip, not --top_fraction")
    parser.add_argument("--coarse_distance_factor", type=float, default=3.0,
                        help="Yaw sweep's max_correspondence_distance as a multiple of "
                             "voxel_size. Raise this (e.g. to 4-5) for noisier scans so "
                             "correspondences aren't rejected just for not landing exactly on "
                             "top of each other")
    parser.add_argument("--yaw_step_deg", type=float, default=3.0,
                        help="Degrees between candidate yaw angles in the sweep. Smaller = "
                             "finer coarse search but slower; the multi-scale ICP stage "
                             "refines the winning angle to full precision afterward regardless")
    parser.add_argument("--yaw_sweep_refine_iterations", type=int, default=5,
                        help="ICP iterations used to locally polish each yaw candidate before "
                             "scoring it. Without this, scoring by raw inlier count lets a "
                             "wrong yaw that happens to align repeated bulk geometry (e.g. "
                             "mounting bosses) outscore the true yaw, since it ignores whether "
                             "the few asymmetric distinguishing features (locator pins, "
                             "connector) actually line up. A short local refinement fixes "
                             "that: the true yaw converges to a tight fit, a merely "
                             "bulk-aligned wrong one generally doesn't. Keep this low - it's "
                             "not the final precision pass, just enough to discriminate "
                             "between candidates")
    parser.add_argument("--icp_distance_factor", type=float, default=2.0,
                        help="ICP max_correspondence_distance as a multiple of voxel_size. "
                             "Must be large enough to exceed the coarse seed's printed "
                             "inlier_rmse, or ICP finds 0 correspondences and silently leaves "
                             "the coarse estimate unrefined (visible as ICP fitness=0.0000)")
    parser.add_argument("--icp_voxel_scales", type=str, default="4.0,2.0,1.0,0.5",
                        help="Comma-separated multiples of voxel_size for coarse-to-fine ICP "
                             "passes (run largest to smallest regardless of list order). Each "
                             "stage re-downsamples source/target and recomputes normals at "
                             "that scale, so source's density gets closer to target's naturally "
                             "sparser (occlusion-limited) density at coarse scales, and each "
                             "stage hands the next a better-converged starting pose. Not going "
                             "below 0.5 on purpose: 0.25mm would drop under a real scanner's "
                             "point spacing, leaving voxels with 0-1 points and degenerate "
                             "per-stage normals")
    parser.add_argument("--icp_max_iterations", type=int, default=100,
                        help="Max iterations per ICP stage (Open3D's own default is 30). "
                             "Raise this for a more fully-converged final transformation, "
                             "especially at the finer voxel scales - each stage runs "
                             "independently up to this budget or until Open3D's relative "
                             "fitness/rmse convergence tolerance is hit, whichever comes first")
    parser.add_argument("--min_coarse_fitness_for_icp", type=float, default=0.1,
                        help="Skip ICP entirely if the coarse (yaw sweep) fitness is below "
                             "this. ICP's linearized solver can diverge to nonsense (meter/"
                             "km-scale translations) when seeded with a badly wrong pose, so "
                             "it's safer to skip it than trust its output on a bad seed")
    parser.add_argument("--no_cad_cache", dest="use_cad_cache", action="store_false", default=True,
                        help="Re-run Poisson-disk sampling on the STL every time instead of "
                             "caching the sampled cloud to a .ply next to it. The cache is "
                             "keyed by --sample_points and auto-invalidated if the STL file "
                             "is newer, so this is only needed to force a fresh (differently "
                             "randomized) sample without changing --sample_points")
    parser.add_argument("--no_preprocess_cache", dest="use_preprocess_cache", action="store_false",
                        default=True,
                        help="Re-run voxel downsampling/normals every time instead of caching "
                             "the result under .preprocess_cache/. The cache key covers every "
                             "parameter that affects the output (voxel_size, crop, noise, "
                             "etc.), so this is rarely needed - mainly useful if you suspect a "
                             "stale cache entry")
    parser.add_argument("--camera_working_distance", type=float, default=500.0,
                        help="Resnična razdalja (mm) med kamero in vrhom dela, vzdolž --up_axis. "
                             "Vzemi to iz datasheeta kandidatne kamere ('working distance'). To "
                             "je zdaj glavni parameter, ki nadomešča staro posredno "
                             "offset_factor*extent hevristiko - z raycasting simulacijo razdalja "
                             "neposredno določa footprint/pokritost/ločljivost skena, zato mora "
                             "biti prava fizična vrednost, ne poljubna")
    parser.add_argument("--camera_width_px", type=int, default=1280,
                        help="Horizontalna ločljivost senzorja (piksli) - iz datasheeta kamere")
    parser.add_argument("--camera_height_px", type=int, default=960,
                        help="Vertikalna ločljivost senzorja (piksli) - iz datasheeta kamere")
    parser.add_argument("--camera_fov_deg", type=float, default=25.0,
                        help="Horizontalno vidno polje (FOV) v stopinjah. Pogosto neposredno v "
                             "datasheetu, sicer izračunaj kot "
                             "2*degrees(atan((footprint_mm/2)/working_distance_mm)) iz podane "
                             "pokritosti pri specificirani delovni razdalji. Mora biti dovolj "
                             "širok, da pri --camera_working_distance pokrije del PLUS mejo za "
                             "placement toleranco (kolikor se del lahko premakne na mizi), sicer "
                             "bo del delno ali popolnoma izven kadra")
    parser.add_argument("--camera_max_incidence_deg", type=float, default=75.0,
                        help="Največji kot vpadanja žarka glede na lokalno normalo površine, pri "
                             "katerem senzor še zanesljivo izmeri globino. Nad tem kotom (blizu "
                             "robov, strmih sten) realni triangulacijski/strukturirano-svetlobni "
                             "senzorji izgubijo signal - prilagodi glede na datasheet, če ta "
                             "podaja 'maximum surface angle' ali podobno")
    parser.add_argument("--camera_depth_noise_at_1m", type=float, default=0.15,
                        help="Std. odklon globinskega šuma (mm) PRI 1m razdalji, dodan vzdolž "
                             "žarka (ne izotropno XYZ). Vzemi iz datasheeta kandidatne kamere "
                             "('depth accuracy'/'repeatability @ 1m') - privzeta vrednost je le "
                             "groba ocena za industrijsko kamero srednjega razreda in NI "
                             "nadomestilo za pravi podatek, če odločitev o nakupu temelji na tej "
                             "simulaciji")
    parser.add_argument("--camera_noise_distance_power", type=float, default=2.0,
                        help="Eksponent, s katerim globinski šum narašča z razdaljo "
                             "(noise_std = depth_noise_at_1m * distance_m^power) - 2.0 ustreza "
                             "tipičnemu kvadratnemu naraščanju šuma pri triangulacijskih "
                             "senzorjih (stereo, strukturirana svetloba); za ToF kamere je "
                             "realnejša vrednost bližje 1.0 - preveri karakteristiko kandidatne "
                             "kamere")
    return parser.parse_args()


def main() -> None:
    # Declared once, up front - a second `global SIMULATED_TRANSFORM` later
    # in this function (e.g. inside the translation-test block below) would
    # be a SyntaxError, since Python requires a name's global declaration to
    # precede every assignment to it in the same function, even assignments
    # that happen inside an earlier, already-global-declared block.
    global SIMULATED_TRANSFORM
    np.random.seed(42)
    base_path = Path(__file__).resolve().parent
    args = parse_args()

    if args.test_tilt_deg != 0.0:
        # Injects roll/pitch into the ground truth that yaw_sweep_registration
        # estimates and corrects from target's PCA surface normal - lets us
        # empirically confirm how much tilt that estimate (plus the
        # multi-scale ICP refinement) can actually recover from, rather than
        # guessing at a tolerance analytically.
        theta = np.radians(args.test_tilt_deg)
        c, s = np.cos(theta), np.sin(theta)
        if args.test_tilt_axis == 0:
            tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        else:
            tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        tilt_4x4 = np.eye(4)
        tilt_4x4[:3, :3] = tilt
        SIMULATED_TRANSFORM = tilt_4x4 @ SIMULATED_TRANSFORM
        axis_name = "X/roll" if args.test_tilt_axis == 0 else "Y/pitch"
        print(f"Test tilt injected: {args.test_tilt_deg} deg about {axis_name}")
        print(f"Modified ground-truth transform:\n{SIMULATED_TRANSFORM}\n")

    cad_path = base_path / args.cad
    real_scan_path = base_path / args.scan

    fixed_camera_location = None
    if args.test_translation_x != 0.0 or args.test_translation_y != 0.0:
        # The real camera is bolted in place above the table and does not
        # move just because the part happens to land somewhere else within
        # its known placement area - so derive its location once from the
        # *nominal* (pre-offset) placement, before injecting the X/Y test
        # offset below, and hold it fixed for the rest of this run. Without
        # this, simulate_camera_scan's dynamic fallback (camera_location=None)
        # re-centers the camera over wherever the part actually ends up,
        # which would silently cancel out the exact placement error this
        # flag exists to test.
        nominal_source = crop_top_region(
            load_cad_model(cad_path, args.sample_points, use_cache=args.use_cad_cache),
            top_fraction=args.top_fraction, up_axis=args.up_axis, flip_up_direction=args.flip_up_direction)
        nominal_target = copy.deepcopy(nominal_source)
        nominal_target.transform(SIMULATED_TRANSFORM)
        fixed_camera_location = top_camera_location(
            nominal_target, up_axis=args.up_axis, flip_up_direction=args.flip_up_direction,
            working_distance=args.camera_working_distance)
        SIMULATED_TRANSFORM = SIMULATED_TRANSFORM.copy()
        SIMULATED_TRANSFORM[0, 3] += args.test_translation_x
        SIMULATED_TRANSFORM[1, 3] += args.test_translation_y
        print(f"Test translation injected: x+={args.test_translation_x}, y+={args.test_translation_y} "
              f"(camera fixed at nominal location {fixed_camera_location})")
        print(f"Modified ground-truth transform:\n{SIMULATED_TRANSFORM}\n")

    run_registration(
        cad_path=cad_path,
        scan_path=real_scan_path,
        use_real_scan=args.use_real_scan,
        voxel_size=args.voxel_size,
        sample_point_count=args.sample_points,
        noise_std=args.noise_std,
        disable_occlusion=args.disable_occlusion,
        top_fraction=args.top_fraction,
        up_axis=args.up_axis,
        flip_up_direction=args.flip_up_direction,
        coarse_distance_factor=args.coarse_distance_factor,
        yaw_step_deg=args.yaw_step_deg,
        yaw_sweep_refine_iterations=args.yaw_sweep_refine_iterations,
        icp_distance_factor=args.icp_distance_factor,
        icp_voxel_scales=tuple(float(s) for s in args.icp_voxel_scales.split(",")),
        icp_max_iterations=args.icp_max_iterations,
        min_coarse_fitness_for_icp=args.min_coarse_fitness_for_icp,
        use_cad_cache=args.use_cad_cache,
        use_preprocess_cache=args.use_preprocess_cache,
        fixed_camera_location=fixed_camera_location,
        camera_width_px=args.camera_width_px,
        camera_height_px=args.camera_height_px,
        camera_fov_deg=args.camera_fov_deg,
        camera_max_incidence_deg=args.camera_max_incidence_deg,
        camera_depth_noise_at_1m=args.camera_depth_noise_at_1m,
        camera_noise_distance_power=args.camera_noise_distance_power,
        camera_working_distance=args.camera_working_distance,
        cache_key_tag=(f"_tilt{args.test_tilt_deg}ax{args.test_tilt_axis}"
                      f"_tx{args.test_translation_x}_ty{args.test_translation_y}"))


if __name__ == "__main__":
    main()