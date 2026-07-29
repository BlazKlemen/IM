import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

try:
    from joblib import Parallel, delayed   # neobvezna odvisnost - glej yaw_sweep_registration za vzporedno ocenjevanje yaw kandidatov
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

PREPROCESS_CACHE_DIR = Path(__file__).resolve().parent / ".preprocess_cache"   # mapa za predpomnjene predobdelane oblake
# To številko povečaj vsakič, ko preprocess_point_cloud() ali karkoli, kar
# kliče, spremeni obnašanje pri enakih vhodih (npr. ocena normal) - cache
# ključ pokriva samo *parametre*, ne verzije kode, zato bi popravek
# brez spremembe parametra sicer še naprej neomejeno servisiral star,
# napačen predpomnjen rezultat namesto ponovnega izračuna.
PREPROCESS_CACHE_VERSION = 6   # trenutna verzija formata predpomnilnika

# funkcija za vizualizacijo rezultatov registracije
def draw_registration_result(source: o3d.geometry.PointCloud,   # izvorni oblak točk
                             target: o3d.geometry.PointCloud,   # ciljni oblak točk
                             transformation: np.ndarray,      # transformacijska matrika
                             window_name: str = "Alignment") -> None:   # ime okna za vizualizacijo
    source_temp = copy.deepcopy(source)     # kopiramo izvorni oblak točk, da ne spremenimo originala
    target_temp = copy.deepcopy(target)     # kopiramo ciljni oblak točk, da ne spremenimo originala
    source_temp.paint_uniform_color([1.0, 0.706, 0.0])  # source pobarvamo oranžno
    target_temp.paint_uniform_color([0.0, 0.651, 0.929])  # target pobarvamo modro
    source_temp.transform(transformation)   # uporabimo transformacijsko matriko na izvorni oblak točk
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)  # odpre okno in nariše oba oblaka skupaj


# nalaganje cad modela
def load_cad_model(mesh_path: Path, n_points: int = 200000, use_cache: bool = True) -> o3d.geometry.PointCloud:  # pot do modela + število točk
    # Poisson-disk vzorčenje na natančnem STL-ju je najpočasnejši del vsakega
    # zagona, poleg tega ni ponovljivo samo po sebi - Open3D-jev vzorčevalnik
    # ima svoj notranji generator naključnih števil, ki ga np.random.seed()
    # ne nadzoruje, zato se število točk med zagoni rahlo spreminja.
    # Predpomnjenje vzorčenega oblaka na disk (ključ = n_points, neveljavno,
    # če se STL spremeni) reši oboje: takojšen ponovni zagon po prvem
    # poganjanju in enake točke vsakič po tem.
    # OPOMBA glede privzetega n_points=200000: to je vzorčeno čez CELOTNO
    # ohišje, preden crop_top_region obdrži ~35% in simulirana okluzija
    # obdrži še del tega, zato realno-vidno/relevantno območje konča pri
    # približno 1/15 tega (~13k točk pri 200k). Ta gostota (pod-mm razmik
    # točk čez ~150mm del) je tisto, kar dejansko dostavi pravi industrijski
    # strukturirano-svetlobni/laserski skener - stari privzeti 20000 je
    # pustil target pri samo ~1300 točkah, nerealno redko in šumno občutljivo.
    # Poisson-disk pri 200k stane nekaj minut PRVI zagon, nato predpomnilnik
    # naredi vsak naslednji zagon takojšen.
    cache_path = mesh_path.with_name(f"{mesh_path.stem}.sampled_{n_points}pts.ply")   # pot do cache datotetke, da ni treba vsakič vzorčit CAD fila
    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= mesh_path.stat().st_mtime:  # cache obstaja in je novejši od STL-ja
        pcd = o3d.io.read_point_cloud(str(cache_path))  # preverimo, če cache datoteka obstaja in je novejša od STL datoteke, če je tako, jo preberemo in vrnemo oblak točk
        if not pcd.is_empty():   # predpomnjen oblak je uporaben
            pcd.paint_uniform_color([1.0, 0.706, 0.0])   # pobarvamo oranžno
            return pcd   # vrnemo predpomnjen oblak, brez ponovnega vzorčenja

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))    # preberemo STL datoteko in ustvarimo mrežo trikotnikov
    if mesh.is_empty():     # preverimo, če je mreža prazna (če STL datoteka ne obstaja ali je poškodovana)
        raise FileNotFoundError(f"CAD mesh not found: {mesh_path}")
    mesh.compute_vertex_normals()   # izračunamo normale za vsako točko v mreži, kar je potrebno za registracijo
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)  # vzorčenje točk iz mreže s Poisson disk metodo, da dobimo oblak točk

    if use_cache:   # shranimo na disk za naslednji zagon
        o3d.io.write_point_cloud(str(cache_path), pcd)

    pcd.paint_uniform_color([1.0, 0.706, 0.0])   # pobarvamo oranžno
    return pcd   # vrnemo sveže vzorčen oblak točk


def load_cad_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    # simulate_camera_scan() pošilja žarke proti pravim trikotnikom, ne
    # proti Poisson-disk vzorčenemu oblaku točk, ki ga vrne load_cad_model()
    # - branje samega STL-ja je hitro (za razliko od Poisson-disk
    # vzorčenja), zato to ne potrebuje lastnega predpomnilnika.
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))   # preberemo STL datoteko v mrežo trikotnikov
    if mesh.is_empty():   # datoteka ne obstaja ali je poškodovana
        raise FileNotFoundError(f"CAD mesh not found: {mesh_path}")
    mesh.compute_vertex_normals()   # izračunamo normale mreže
    return mesh   # vrnemo celotno, neobrezano mrežo


def build_reference_transform(translation_x: float, translation_y: float, translation_z: float,
                              rotation_x_deg: float, rotation_y_deg: float, rotation_z_deg: float) -> np.ndarray:
    """Sestavi referenčno (ground-truth) 4x4 transformacijsko matriko dela
    iz šestih neposrednih parametrov - to je poza, ki jo mora registracijski
    cevovod sam oceniti/izračunati (glej main() in kontrolna_plosca.py).

    Kamera je VEDNO fiksirana v izhodišču svetovnega koordinatnega sistema
    (0,0,0), za realne IN simulirane skene enako (glej run_registration -
    target_camera_location je zdaj vedno np.zeros(3), ne dinamično
    izračunana glede na to, kam pristane transformiran del) - zato
    translation_x/y/z neposredno predstavlja pozicijo dela GLEDE NA
    kamero, ne poljubno svetovno koordinato.

    Privzeto (glej parse_args) so rotation_x/y/z_deg=0 (del ne naklonjen
    ne zasukan) in translation_x/y=0 (del natanko pod kamero v X/Y) -
    edini smiselno neničeln privzetek je translation_z, ki mora biti
    negativen (kamera gleda navzdol, del je pod njo) in enak (negativni)
    idealni sweet spot/fokusni delovni razdalji SIMULIRANE kamere (glej
    kamere.py in kontrolna_plosca.py-jev resolve_camera_params, ki to
    samodejno nastavi glede na izbrano kamero).

    Rotacija: R = Rz(rotation_z_deg) @ Ry(rotation_y_deg) @ Rx(rotation_x_deg)
    (zunanja/svetovna XYZ Eulerjeva konvencija - vsak zasuk je okoli
    FIKSNE svetovne osi, ne okoli že zasukane lastne osi telesa)."""
    theta_x, theta_y, theta_z = np.radians([rotation_x_deg, rotation_y_deg, rotation_z_deg])
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])   # rotacija okoli X (roll)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])   # rotacija okoli Y (pitch)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])   # rotacija okoli Z (yaw)
    transform = np.eye(4)
    transform[:3, :3] = rot_z @ rot_y @ rot_x
    transform[:3, 3] = [translation_x, translation_y, translation_z]
    return transform


# Privzeta referenčna transformacija, uporabljena za generiranje simuliranega
# skeniranja iz CAD modela - main() jo pri vsakem zagonu prepiše z
# build_reference_transform() iz dejansko podanih --translation_x/y/z/
# --rotation_x/y/z_deg (privzeto ravno te iste vrednosti, glej parse_args).
# Ta modulska vrednost je torej samo placeholder za uvoz/uporabo
# main_trial.py kot modula brez klica main() (npr. v self-testih).
SIMULATED_TRANSFORM = build_reference_transform(
    translation_x=0.0, translation_y=0.0, translation_z=-500.0,
    rotation_x_deg=0.0, rotation_y_deg=0.0, rotation_z_deg=0.0)

def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    """Separabilno 2D Gaussovo glajenje, samo z numpy.
    Uporabljeno spodaj za prostorsko koreliranje šuma globine po pikslih:
    matching/korelacijsko okno pravega globinskega senzorja povpreči več
    sosednjih pikslov hkrati, zato napaka meritve sosednjih pikslov ni
    neodvisna (iid), kot bi to modeliral navaden np.random.normal na
    vsakem pikslu posebej. Validirano proti brute-force referenčni
    konvoluciji (max abs razlika ~1e-16)."""
    if sigma <= 0:   # brez glajenja - vrnemo nespremenjeno kopijo
        return array.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))   # polmer jedra v pikslih, glede na sigma
    x = np.arange(-radius, radius + 1)   # indeksi jedra okoli sredine
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))   # 1D Gaussovo jedro
    kernel /= kernel.sum()   # normaliziramo, da se uteži seštejejo v 1

    def convolve_along_axis(a: np.ndarray, axis: int) -> np.ndarray:   # 1D konvolucija vzdolž ene osi
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (radius, radius)   # dopolnimo rob samo na izbrani osi
        padded = np.pad(a, pad_width, mode="reflect")   # zrcalno dopolnimo robove, da glajenje ne potemni robov
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)   # drseča okna dolžine jedra
        return np.tensordot(windows, kernel, axes=([-1], [0]))   # uteženo povprečje vsakega okna z jedrom

    return convolve_along_axis(convolve_along_axis(array, axis=0), axis=1)   # najprej po vrsticah, nato po stolpcih = 2D glajenje


def heteroscedastic_noise_std_mm(distance_m: np.ndarray,
                                 incidence_cos: np.ndarray,
                                 depth_noise_at_1m: float,
                                 noise_distance_power: float,
                                 max_incidence_deg: float,
                                 noise_reference_distance_m: float = 1.0) -> np.ndarray:
    """Std (mm) globinskega šuma za triangulacijski senzor: raste z
    razdaljo (razmerje distance_m/noise_reference_distance_m na potenco
    noise_distance_power, umerjeno tako, da je rezultat PRI
    noise_reference_distance_m natanko depth_noise_at_1m) IN s kotom
    vpadanja - projicirana baseline med kamero in projektorjem/drugo kamero
    se skrči, ko je površina nagnjena stran od naravnost, kar poveča
    triangulacijsko negotovost približno kot 1/cos(kot vpadanja).
    incidence_cos je najprej omejen na cos(max_incidence_deg), da faktor
    1/cos ne eksplodira tik pred max_incidence_deg, kjer gre cos proti 0.

    noise_reference_distance_m (privzeto 1.0, torej nazaj-kompatibilno z
    imenom parametra depth_noise_at_1m): OPOZORILO iz kamere.py - realni
    specsi triangulacijskih senzorjev (Photoneo, Zivid, Mech-Eye) podajo
    šum pri svoji dejanski delovni razdalji (npr. 384-520mm, 350-900mm,
    500-600mm), NIKOLI pri 1m. Ekstrapolacija privzetega
    noise_distance_power=2.0 iz teh vrednosti do 1m je nezanesljiva, ker
    tako ozek razpon pogosto sploh ne razkrije pravega eksponenta rasti
    (npr. Photoneo-jev temporalni šum je skoraj raven čez celo svoje
    delovno območje). Nastavite noise_reference_distance_m na dejansko
    razdaljo, pri kateri je bil depth_noise_at_1m izmerjen (npr. 0.442 za
    Photoneo-jev sweet spot), namesto da privzeta vrednost tiho ekstrapolira
    do 1m."""
    max_incidence_cos = np.cos(np.radians(max_incidence_deg))   # cos praga - spodnja meja za clip
    incidence_factor = 1.0 / np.clip(incidence_cos, max_incidence_cos, 1.0)   # 1/cos faktor, omejen navzgor
    relative_distance = np.maximum(distance_m, 1e-6) / max(noise_reference_distance_m, 1e-6)   # razdalja relativno na referenčno razdaljo
    return depth_noise_at_1m * np.power(relative_distance, noise_distance_power) * incidence_factor   # končen std: osnova × (razdalja/referenca)^power × kot vpadanja


def apply_distance_bias_mm(distance_mm: np.ndarray, bias_permille: float) -> np.ndarray:
    """Uporabi SISTEMATIČNO (ne naključno na piksel) napako kalibracije
    razdalje: fiksen multiplikativen faktor (1 + bias_permille/1000),
    konstanten za CEL zajem in za VSAK piksel v njem - za razliko od
    heteroscedastic_noise_std_mm-ovega šuma zgoraj, ki je neodvisen na
    vsak piksel posebej.

    Ustreza kamere.py-jevemu "Relative distance accuracy (‰)"/"Dimension
    Trueness Error" specu - skupna, pretežno sistematična natančnost
    meritve razdalje kot delež same razdalje, ne naključni šum, zato je
    prej nemodelirana v main_trial.py-jevem sicer čisto naključnem
    šumovnem modelu.

    bias_permille je DETERMINISTIČEN parameter tega zagona (enaka vloga
    kot --translation_x/y/z/--rotation_x/y/z_deg), ne naključno vzorčen ob
    vsakem klicu - namen je nadzorovano testiranje tolerance na znano
    sistematično napako (npr. --distance_bias_permille 1.25 za Photoneo-jev
    specificiran worst-case), ne simulacija naključne variacije med
    posameznimi senzorskimi enotami."""
    return distance_mm * (1.0 + bias_permille / 1000.0)   # + bias_permille‰ sistematičen zamik razdalje


def generate_global_planarity_bias_mm(shape: tuple[int, int],
                                      amplitude_mm: float,
                                      low_freq_sigma_px: float) -> np.ndarray:
    """Generira gladko, NIZKOFREKVENČNO 2D polje sistematičnega odstopanja
    (bias) čez celotno vidno polje - ustreza kamere.py-jevemu "Global
    planarity" specu (rahel upogib/popačenje pri skeniranju sicer ravne
    referenčne ploskve). main_trial.py-jev obstoječi šumovni model
    (heteroscedastic gaussian, osamelci, leteči piksli, kvantizacija)
    modelira SAMO naključen, ničelno-povprečen šum na posamezen piksel -
    nič od tega ne predstavlja gladke sistematične napake čez celotno
    sliko, zato je to ločena funkcija, ne razširitev obstoječih.

    Implementirano kot Gaussovo (gaussian_blur_2d) zglajen iid šum z zelo
    velikim sigma (v pikslih - velik del slike, "nizka frekvenca" v
    prostorskem smislu), rescaliran tako, da je razpon (max-min) natanko
    amplitude_mm - kar ustreza vendor-jevi definiciji "Global planarity"
    kot odstopanja od ravnine čez celotno FOV. Za en klic simulate_camera_scan
    (en "zajem") je to polje KONSTANTNO čez vse piksle tega zajema - isti
    fizični senzor na isti poziciji bi pri ponovljenem zajemu pokazal
    približno isto obliko tega popačenja, za razliko od preostalega šuma,
    ki je neodvisen med zajemi."""
    if amplitude_mm <= 0:   # brez sistematičnega popačenja - vrnemo ničelno polje
        return np.zeros(shape)
    raw = np.random.normal(size=shape)   # iid osnova, glajena spodaj v gladko nizkofrekvenčno polje
    smooth = gaussian_blur_2d(raw, low_freq_sigma_px)
    smooth -= smooth.mean()   # centriramo okoli 0, da je popačenje simetrično navzgor/navzdol
    current_range = smooth.max() - smooth.min()   # dejanski razpon pred rescaliranjem
    if current_range < 1e-12:   # degeneriran rob primer (praktično nemogoč, a izognemo se deljenju z 0)
        return np.zeros(shape)
    return smooth * (amplitude_mm / current_range)   # rescaliramo, da je razpon natanko amplitude_mm


def build_table_mesh(up_axis: int, table_height: float, center: np.ndarray,
                     size_x: float, size_y: float) -> o3d.geometry.TriangleMesh:
    """Zgradi ploščato pravokotno mrežo (samo 2 trikotnika), ki predstavlja
    mizo pod delom - uporabi jo simulate_camera_scan(), kadar je
    add_table_background=True, da ray casting zajame tudi ozadje mize
    okoli dela, ne le dela samega. table_height je pozicija vzdolž
    up_axis, center/size_x/size_y pa določajo pravokotnik v preostalih
    dveh oseh (enaka konvencija kot table_size_x/y/table_center pri
    fizičnem priorju v yaw_sweep_registration - gre za isto fizično mizo)."""
    plane_axes = [a for a in range(3) if a != up_axis]   # dve osi, v katerih leži miza
    half_extent = np.array([size_x, size_y]) / 2.0   # polovična širina/globina mize
    corners_2d = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half_extent + center   # 4 vogali pravokotnika
    vertices = np.zeros((4, 3))
    vertices[:, plane_axes] = corners_2d   # vstavimo vogale v pravo ravnino
    vertices[:, up_axis] = table_height   # vsi vogali na isti višini (miza je ravna)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])   # dva trikotnika sestavita pravokotnik
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()   # izračunamo normalo (ravna ploskev, ena sama smer)
    return mesh   # vrnemo mrežo mize


# Simulacija skeniranja kamere iz CAD modela z ray castingom (o3d.t.geometry.RaycastingScene)
def simulate_camera_scan(mesh: o3d.geometry.TriangleMesh,           # NEtransformirana, neobrezana CAD mreža (iz load_cad_mesh)
                         transform: np.ndarray | None = None,       # transformacijska matrika
                         camera_location: Optional[np.ndarray] = None,   # lokacija kamere - če None, privzeto izhodišče (0,0,0), glej spodaj
                         up_axis: int = 2,                          # os, ki predstavlja "gor" - kamera je vedno nad delom vzdolž te osi, nikoli spodaj
                         flip_up_direction: bool = False,           # če True, je kamera na nasprotni (-up_axis) strani - glej crop_top_region()
                         top_fraction: float = 0.35,                # enako kot crop_top_region - omeji mrežo pred ray castingom na isto zgornje "sealing" območje, ki ga predstavlja source, sicer target zajame tudi stranske izbokline pod tem območjem, ki bi bile sicer vidne od zgoraj, a jih source (obrezan CAD) ne vsebuje
                         fov_deg: float = 40.0,                     # horizontalno vidno polje kamere v stopinjah
                         width_px: int = 640,                       # širina simulirane slike v pikslih (= število žarkov po širini)
                         height_px: int = 480,                      # višina simulirane slike v pikslih (= število žarkov po višini)
                         disable_occlusion: bool = False,           # če je True, se preskoči ray casting in namesto tega gosto vzorči obrezano mrežo, za testiranje vpliva delne vidljivosti na registracijo
                         depth_noise_at_1m: float = 0.1,            # std (mm) globinskega šuma pri razdalji noise_reference_distance_m in kotu vpadanja 0 (naravnost)
                         noise_distance_power: float = 2.0,         # eksponent rasti šuma z razdaljo (distance_m/noise_reference_distance_m ** to)
                         noise_reference_distance_m: float = 1.0,   # razdalja (m), pri kateri je depth_noise_at_1m dejansko izmerjen/kalibriran - glej heteroscedastic_noise_std_mm docstring, kamere.py opozorilo o ekstrapolaciji do 1m
                         max_incidence_deg: float = 75.0,           # zgornja meja kota vpadanja za 1/cos faktor, da ne eksplodira tik pred robom
                         distance_bias_permille: float = 0.0,       # sistematičen (ne naključen na piksel) multiplikativen zamik razdalje, konstanten za CEL zajem - glej apply_distance_bias_mm, kamere.py "Relative distance accuracy"/"Dimension Trueness Error"
                         global_planarity_mm: float = 0.0,          # amplituda (razpon max-min, mm) gladkega nizkofrekvenčnega sistematičnega popačenja čez celo FOV - glej generate_global_planarity_bias_mm, kamere.py "Global planarity"/"Global Planarity Trueness Error"; 0 = izklopljeno (privzeto, nazaj-kompatibilno)
                         global_planarity_spatial_scale_px: Optional[float] = None,   # sigma (v pikslih) glajenja za global_planarity_mm polje - privzeto (None) min(width_px,height_px)/4, dovolj velik za "nizko frekvenco" (gladko čez velik del slike, ne lokalen vzorec)
                         outlier_probability: float = 0.01,         # delež pikslov, ki dobijo grobo napako (multipath/šibek SNR/napačno ujemanje) namesto navadnega šuma
                         outlier_std_multiplier: float = 15.0,      # kolikokrat širša porazdelitev za osamelce v primerjavi z navadnim šumom na tem pikslu
                         flying_pixel_depth_jump_mm: float = 3.0,   # globinski skok med sosednjima pikseloma, ki šteje za diskontinuiteto (rob dela/pina/šiva)
                         flying_pixel_probability: float = 0.3,     # verjetnost, da piksel na diskontinuiteti postane "leteč" (vmesna, neresnična globina)
                         quantization_step_at_1m: float = 0.05,     # korak kvantizacije (mm) pri 1m, skalira se z razdaljo enako kot šum
                         noise_spatial_correlation_px: float = 1.5,  # sigma (v pikslih) prostorske korelacije šuma - velikost tipičnega matching okna senzorja
                         add_table_background: bool = True,        # če True, dodamo ravno mizo pod del v sceno za ray casting, da target zajame tudi ozadje
                         table_size_x: float = 500.0,                # širina mize (mm) - ista miza kot pri fizičnem priorju v yaw_sweep_registration
                         table_size_y: float = 500.0,                # globina mize (mm)
                         table_center: Optional[np.ndarray] = None   # center mize (X/Y) - privzeto (0,0)
                         ) -> o3d.geometry.PointCloud:
    """Simulira pravi globinski skener z ray castingom namesto prejšnjega
    hidden_point_removal pristopa.

    hidden_point_removal projicira točke na kroglo okoli točke pogleda in
    oceni vidljivost iz te projekcije - za to potrebuje ročno nastavljen
    parameter `radius`, ki je posredno vezan na razdaljo kamere. To je
    delovalo, dokler je bila simulirana kamera le nekaj sto mm nad delom;
    ko se je kamera premaknila na realističnih ~1m (glej top_camera_location's
    offset_distance), je isti radij začel zavračati veliko preveč točk,
    daleč več, kot bi resnična okluzija upravičila.

    Ray casting nima takega od razdalje odvisnega parametra: iz dejanske
    pozicije kamere se skozi dejansko vidno polje sproži en žarek na
    simuliran piksel, prvi zadeti trikotnik pa JE vidna površina - natanko
    to, kar izmeri pravi strukturirano-svetlobni/laserski skener, ne glede
    na razdaljo kamere.

    mesh je NEtransformirana, NEobrezana CAD mreža (iz load_cad_mesh) - ta
    funkcija jo obreže z isto crop_top_region logiko, ki jo run_registration
    uporabi za source, PRED ray castingom. To ni nujno "kar bi prava kamera
    videla" (izbokline pod top_fraction, ki gledajo neposredno navzgor in
    niso ničesar zakrite, bi bile realno vidne tudi njej) - a source (CAD
    referenca za primerjavo) je namenoma omejen na isto zgornje "sealing"
    območje, zato mora target predstavljati isto fizično območje, sicer
    primerjava source/target zajema različno geometrijo in se ujemanje
    podre (preverjeno empirično: brez tega obrezovanja target zajame tudi
    stranske konektorje/izbokline pod top_fraction, PCA oceno naklona pa to
    popolnoma zavede).

    Šumovni model (na t_hit mreži, PRED flatten v seznam točk - popolnoma
    nadomešča prejšnji plosk izotropen Gaussov šum na koncu):
      1. heteroscedastičen std po razdalji IN kotu vpadanja
         (heteroscedastic_noise_std_mm)
      2. prostorsko koreliran Gaussov šum s tem std (gaussian_blur_2d na
         iid N(0,1), renormaliziran na std=1, šele nato pomnožen z (1))
      3. osamelci (gross errors): mešanica dveh porazdelitev - del pikslov
         namesto (2) dobi neodvisen vzorec iz veliko širše N(0, std*outlier_std_multiplier)
      4. leteči piksli na globinskih diskontinuitetah: piksel na robu
         (skok globine med sosedi > flying_pixel_depth_jump_mm) z neko
         verjetnostjo dobi globino, interpolirano med svojo in sosednjo
         (bolj oddaljeno) globino, namesto prave zadete globine
      5. kvantizacija: zaokrožitev na korak, ki prav tako raste z razdaljo

    Zgornjih 5 členov je VSE naključnih, ničelno-povprečnih na posamezen
    piksel - noben ne predstavlja SISTEMATIČNE napake senzorja (glej
    kamere.py opozorila o "Global planarity" in "Relative distance
    accuracy"/"Dimension Trueness Error", ki ju resnični specsi podajo, a
    prejšnja verzija te datoteke ni modelirala). Dva dodatna, privzeto
    izklopljena (0) člena to pokrijeta:
      6. distance_bias_permille (apply_distance_bias_mm): DETERMINISTIČEN
         multiplikativen zamik cele izmerjene razdalje, konstanten za cel
         zajem - sistematična napaka umerjanja razdalje kot delež same
         razdalje.
      7. global_planarity_mm (generate_global_planarity_bias_mm): gladko,
         nizkofrekvenčno 2D polje, prav tako konstantno za cel zajem -
         sistematičen upogib/popačenje čez celotno vidno polje.
    Oba sta namenoma ločena od zgornjih 5 (naključnih) členov, ker gre za
    KVALITATIVNO drugačen mehanizem napake - konstanten znotraj enega
    zajema, ne neodvisen na vsak piksel/klic.

    Kamera ima FIKSNO orientacijo - vedno gleda naravnost navzdol (vzdolž
    -up_axis), NE glede na to, kje dejansko pristane del (translation_x/y).
    To ni golo poenostavitveno poenostavljanje, ampak namerna fizikalna
    lastnost: prava kamera je nepremično montirana in ne "sledi" delu, zato
    del, ki se dovolj odmakne vstran od kamerine osi, dejansko (delno ali
    povsem) izpade iz njenega vidnega polja (FOV) - glej `look_at_point`
    spodaj. Če translation_x/y potisne CEL del izven FOV, ray casting ne
    zadene ničesar in funkcija dvigne RuntimeError - to je pričakovano in
    pravilno obnašanje, ne napaka nastavitve.

    add_table_background=True doda v sceno za ray casting še ravno mizo
    (build_table_mesh) na višini, kjer se NEobrezan del dejansko dotika
    mize - žarki, ki zgrešijo del, tako namesto v prazno (brez zadetka)
    zadenejo mizo, target pa dobi tudi ozadje okoli dela, ne le del sam.
    Poenostavitev: miza je dodana v sceno SKUPAJ z že obrezano (top_fraction)
    mrežo dela, ne s celotno neobrezano mrežo, zato lahko na mestih, kjer bi
    v resnici spodnji (obrezani) del ohišja zakril pogled na mizo, ray
    casting mizo napačno pokaže kot vidno - sprejemljivo za diagnostično
    testiranje, ne pa fizikalno popolnoma natančno.
    """
    if transform is None:       # če transformacija ni podana, uporabimo simulirano transformacijo
        transform = SIMULATED_TRANSFORM
    mesh = copy.deepcopy(mesh)          # ne spreminjamo originala
    mesh.transform(transform)           # uporabimo transformacijsko matriko na CAD mrežo, da dobimo ciljno mrežo v svetovnih koordinatah

    table_mesh = None
    if add_table_background:
        # Višino mize moramo izračunati iz NEobrezane mreže (preden jo
        # crop_top_region skrči na zgornje območje), saj je prav spodnji
        # del (ki ga obrezovanje odstrani) tisti, ki se dotika mize.
        table_center_arr = table_center if table_center is not None else np.zeros(2)   # privzet center mize (0,0)
        table_height = (mesh.get_min_bound()[up_axis] if not flip_up_direction
                       else mesh.get_max_bound()[up_axis])   # miza je na nasprotni strani od kamere
        table_mesh = build_table_mesh(up_axis, table_height, table_center_arr, table_size_x, table_size_y)   # zgradimo mrežo mize

    mesh = crop_top_region(mesh, top_fraction=top_fraction, up_axis=up_axis, flip_up_direction=flip_up_direction)   # obrežemo na isto zgornje območje kot source

    if camera_location is None:
        # Kamera je VEDNO fiksirana v izhodišču svetovnega koordinatnega
        # sistema (0,0,0) - za realne IN simulirane skene enako, glej
        # run_registration in build_reference_transform docstring. Prejšnja
        # različica te funkcije je tu dinamično izračunala pozicijo "vedno
        # nad transformiranim delom" (top_camera_location(mesh, ...)) - to
        # je tiho pomenilo, da se kamera premika skupaj s testno
        # transformacijo (SIMULATED_TRANSFORM), kar realna, fiksno montiranaqqq
        # kamera nikoli ne počne. Zdaj je transform (in s tem translation_x/
        # y/z) tisto, kar se premika GLEDE NA fiksno kamero, ne obratno.
        camera_location = np.zeros(3)

    if disable_occlusion:
        # Preskoči ray casting in gosto vzorči CELO (transformirano) mrežo -
        # namerno VEČ pokritosti, kot jo pravi skener kadarkoli dobi, da lahko
        # testiramo, ali je delna vidljivost (okluzija + vidno polje) - ne
        # gostota ali kadriranje - tisto, kar povzroča odstopanje dobljene
        # transformacije od referenčne. Brez piksel mreže tu ni prostorske
        # korelacije/letečih pikslov/kvantizacije - samo heteroscedastičen
        # (razdalja+kot vpadanja) šum in osamelci, izotropno na vsako točko,
        # namesto plosko izotropnega Gaussovega šuma kot prej.
        pcd = mesh.sample_points_poisson_disk(number_of_points=width_px * height_px)   # gosto vzorčenje cele obrezane mreže
        points = np.asarray(pcd.points)   # točke kot numpy tabela
        distance_m = np.linalg.norm(points - camera_location, axis=-1) / 1000.0   # razdalja vsake točke od kamere v metrih
        if pcd.has_normals() and len(pcd.normals) == len(points):   # imamo uporabne normale za oceno kota vpadanja
            view_dir = camera_location - points   # smer od točke proti kameri
            view_dir /= np.clip(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-12, None)   # normaliziramo na enotsko dolžino
            incidence_cos = np.clip(np.abs(np.sum(view_dir * np.asarray(pcd.normals), axis=-1)), 1e-3, 1.0)   # kosinus kota med pogledom in normalo
        else:   # ni normal - privzamemo naravnost gledanje (brez faktorja kota)
            incidence_cos = np.ones(len(points))
        noise_std_mm = heteroscedastic_noise_std_mm(
            distance_m, incidence_cos, depth_noise_at_1m, noise_distance_power, max_incidence_deg,
            noise_reference_distance_m=noise_reference_distance_m)   # std šuma za vsako točko
        # distance_bias_permille/global_planarity_mm namerno NISO uporabljena
        # v tej veji: to je diagnostičen način brez piksel mreže, dodani šum
        # je izotropen na vsako točko (ne vzdolž smeri pogleda kamere), zato
        # sistematičen RADIALNI zamik razdalje (distance_bias_permille) in
        # 2D slikovno-ravninsko polje (global_planarity_mm) tu nimata
        # smiselne, jasno definirane vloge - oba veljata samo za glavno,
        # fizikalno podrobno ray-casting vejo spodaj.
        is_outlier = np.random.random(len(points)) < outlier_probability   # katere točke postanejo osamelci
        effective_std = np.where(is_outlier, noise_std_mm * outlier_std_multiplier, noise_std_mm)   # večji std za osamelce
        points += np.random.normal(size=points.shape) * effective_std[:, None]   # dodamo šum vsaki točki
        pcd.points = o3d.utility.Vector3dVector(points)   # shranimo zašumljene točke nazaj v oblak
        return pcd   # vrnemo brez ray castinga (diagnostičen način)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)   # pretvorimo mrežo v obliko za ray casting
    scene = o3d.t.geometry.RaycastingScene()   # ustvarimo prazno sceno za ray casting
    scene.add_triangles(mesh_t)   # dodamo mrežo dela v sceno
    if table_mesh is not None:
        table_t = o3d.t.geometry.TriangleMesh.from_legacy(table_mesh)   # pretvorimo mizo v obliko za ray casting
        scene.add_triangles(table_t)   # dodamo mizo v isto sceno - žarki, ki zgrešijo del, zdaj zadenejo njo

    # Katerakoli os razen up_axis deluje kot referenčni "up" vektor virtualne
    # kamere (mora biti le ne-vzporedna s smerjo pogleda eye->center) -
    # (up_axis + 1) % 3 izbere eno izmed preostalih dveh, dosledno.
    up_vector = np.zeros(3)
    up_vector[(up_axis + 1) % 3] = 1.0   # izberemo os, ki ni up_axis, kot referenčni "up" kamere

    # Kamera ima FIKSNO orientacijo - vedno gleda naravnost navzdol (vzdolž
    # -up_axis, ali navzgor vzdolž +up_axis, če je flip_up_direction), NE
    # GLEDE NA TO, kje dejansko pristane del (translation_x/y v
    # build_reference_transform lahko del postavi poljubno vstran od
    # kamere). Prejšnja koda je tu namesto tega uporabila
    # center=mesh.get_center() - ker je `mesh` tu že TRANSFORMIRAN (torej
    # premaknjen za translation_x/y), bi to kamero vedno znova "naravnalo"
    # naravnost na del, ne glede na to, kako daleč vstran je - kar bi tiho
    # izničilo ravno to, kar naj bi testiranje X/Y odmika
    # (--translation_x/--translation_y) pokazalo: pravi, fiksno montiran
    # senzor NE sledi delu, zato del, ki se dovolj odmakne vstran, dejansko
    # (delno ali povsem) izpade iz njegovega vidnega polja (FOV) - točno
    # tako, kot mora.
    look_direction = np.zeros(3)
    look_direction[up_axis] = -1.0 if not flip_up_direction else 1.0   # vedno navzdol (ali navzgor, če flip) vzdolž up_axis
    look_at_point = camera_location + look_direction   # poljubna točka v tej fiksni smeri - oddaljenost od kamere ne vpliva na smer žarkov

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov_deg, center=look_at_point, eye=camera_location, up=up_vector,
        width_px=width_px, height_px=height_px)   # ustvarimo mrežo žarkov za virtualno kamero s FIKSNO smerjo pogleda
    result = scene.cast_rays(rays)   # sprožimo žarke proti sceni in dobimo zadetke

    t_hit = result["t_hit"].numpy()             # razdalja do prvega zadetka na žarek, inf če žarek ni zadel ničesar
    hit_mask = np.isfinite(t_hit)   # kateri piksli so dejansko zadeli mrežo
    if not hit_mask.any():   # noben žarek ni zadel ničesar - napačna nastavitev kamere
        raise RuntimeError("Ray casting ni zadel ničesar - camera_location/fov_deg/width_px/"
                           "height_px verjetno ne kažejo na del")

    rays_np = rays.numpy()   # žarki kot numpy tabela (izvor + smer)
    ray_dirs = rays_np[..., 3:6]   # smerni del vsakega žarka
    dir_length = np.linalg.norm(ray_dirs, axis=-1)   # dolžina smernega vektorja vsakega žarka

    # Non-hit piksli nosijo t_hit/razdaljo = inf skozi vsak šumovni člen
    # spodaj (zavržejo se šele čisto na koncu prek hit_mask, glej
    # points = hit_points[hit_mask]) - inf v kombinaciji z množenjem z 0 ali
    # drugim inf (npr. depth_noise_at_1m * inf, inf/inf pri kvantizaciji) je
    # matematično neškodljiv nan v celicah, ki se tako ali tako zavržejo,
    # a numpy o tem privzeto vseeno opozori; to opozorilo je pričakovano in
    # zatrto za ves ta blok, namesto da bi kodo prestrukturirali, saj bi
    # zgodnje maskiranje non-hit pikslov po nepotrebnem zapletlo vsako
    # vektorizirano operacijo spodaj, brez razlike v obnašanju.
    with np.errstate(invalid="ignore", divide="ignore"):   # zatremo pričakovana opozorila inf/nan
        # Enota t_hit je dolžina (nenormaliziranega, perspektivno
        # spremenljivega) smernega vektorja tega piksla, ne mm neposredno -
        # pretvorba v pravo evklidsko razdaljo tukaj omogoči, da je vsak
        # šumovni člen spodaj podan v pravih mm, ne v tej po pikslih
        # spremenljivi enoti.
        distance_mm = np.where(hit_mask, t_hit * dir_length, np.inf)   # prava razdalja v mm, inf za non-hit
        distance_mm = apply_distance_bias_mm(distance_mm, distance_bias_permille)   # sistematičen (ne naključen) zamik kalibracije razdalje, glej apply_distance_bias_mm - neškodljivo tudi za inf (non-hit) piksle
        distance_m = distance_mm / 1000.0   # ista razdalja v metrih (za formulo šuma), zdaj vključno z distance_bias_permille

        primitive_normals = result["primitive_normals"].numpy()   # normala zadetega trikotnika na vsak piksel
        dir_unit = ray_dirs / np.clip(dir_length[..., None], 1e-12, None)   # normalizirana smer žarka
        # abs() se izogne dvoumnosti predznaka normale (odvisno od
        # orientacije trikotnika) - za šumovni model je pomembno le, kako
        # "naravnost" je površina obrnjena proti žarku, ne v katero smer
        # normala dejansko kaže.
        incidence_cos = np.clip(np.abs(np.sum(dir_unit * primitive_normals, axis=-1)), 1e-3, 1.0)   # kosinus kota vpadanja

        noise_std_grid_mm = np.where(
            hit_mask,
            heteroscedastic_noise_std_mm(distance_m, incidence_cos, depth_noise_at_1m,
                                         noise_distance_power, max_incidence_deg,
                                         noise_reference_distance_m=noise_reference_distance_m),
            0.0)   # std šuma za vsak piksel (0 za non-hit)

        # Globalna planarnost: gladko, nizkofrekvenčno sistematično polje
        # čez CELO sliko (KONSTANTNO za ta zajem, ne naključno na piksel kot
        # preostali šum) - glej generate_global_planarity_bias_mm docstring
        # in kamere.py "Global planarity" opozorilo. 0 amplitude (privzeto)
        # vrne ničelno polje, torej brez sprememb obnašanja, če ni nastavljeno.
        global_planarity_bias_mm = generate_global_planarity_bias_mm(
            t_hit.shape,
            amplitude_mm=global_planarity_mm,
            low_freq_sigma_px=(global_planarity_spatial_scale_px
                              if global_planarity_spatial_scale_px is not None
                              else min(width_px, height_px) / 4.0))

        # Prostorsko korelirani heteroscedastičen šum globine: NAJPREJ
        # zgladimo iid enotski šum (matching okno pravega senzorja meša
        # sosednje piksle, zato njihove napake niso neodvisne), nato
        # renormaliziramo izgubo variance zaradi glajenja nazaj na std=1,
        # ŠELE NATO pomnožimo s std vsakega piksla - skaliranje pred
        # glajenjem bi glajenje spralo prav tisto heteroscedastičnost
        # (kot vpadanja/oddaljenost), ki naj bi jo ohranilo.
        raw_noise = np.random.normal(size=t_hit.shape)   # neodvisen enotski šum, en na piksel
        correlated_noise = gaussian_blur_2d(raw_noise, noise_spatial_correlation_px)   # prostorsko zgladimo
        correlated_std = correlated_noise.std()   # dejanski std po glajenju (glajenje ga zmanjša)
        if correlated_std > 1e-12:
            correlated_noise = correlated_noise / correlated_std   # renormaliziramo nazaj na std=1
        depth_noise_mm = correlated_noise * noise_std_grid_mm   # šele zdaj pomnožimo s pravim std po pikslih

        # Osamelci (multipath, šibek SNR, napačna stereo/strukturirano-
        # svetlobna ujemanja): resnično drugačen mehanizem napake kot
        # navaden šum senzorja, ne "malo več istega šuma" - del pikslov
        # dobi neodvisen vzorec iz veliko širše Gaussove porazdelitve
        # NAMESTO (ne poleg) koreliranega šuma zgoraj.
        is_outlier = np.random.random(t_hit.shape) < outlier_probability   # kateri piksli postanejo osamelci
        outlier_noise_mm = np.random.normal(size=t_hit.shape) * (noise_std_grid_mm * outlier_std_multiplier)   # veliko širši šum za osamelce
        depth_noise_mm = np.where(is_outlier, outlier_noise_mm, depth_noise_mm)   # zamenjamo šum osamelcev, ne dodamo poleg

        distance_mm_noisy = distance_mm + depth_noise_mm + global_planarity_bias_mm   # razdalja z dodanim naključnim ŠUMOM + sistematičnim popačenjem (global planarity)

        # Leteči piksli: na globinski diskontinuiteti (rob dela, rob pina,
        # šiv) matching okno pravega senzorja hkrati zajame oba globinska
        # nivoja in lahko poroča interpolirano, fizično neobstoječo globino
        # med njima - "lebdeče" točke, vidne na robovih pravih skenov, ki
        # jih navaden Gaussov šum na točko nikoli ne ustvari. Primerja le
        # pare sosedov, ki sta OBA zadetek (hit_mask & shifted_hit), zato
        # meja med objektom in ozadjem - kjer je "ozadje" preprosto
        # odsotnost zadetka, ne druga prava površina - nikoli napačno ne
        # šteje kot skok.
        finite_distance = np.where(hit_mask, distance_mm, 0.0)   # razdalja, 0 za non-hit (namesto inf, za lažjo primerjavo)
        depth_jump = np.zeros_like(hit_mask)   # maska diskontinuitet, začetno prazna
        max_neighbor_distance = finite_distance.copy()   # tekoč maksimum razdalje med sabo in sosedi
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:   # 4 sosedske smeri (gor/dol/levo/desno)
            shifted_distance = np.roll(finite_distance, (dy, dx), axis=(0, 1))   # razdalja soseda v tej smeri
            shifted_hit = np.roll(hit_mask, (dy, dx), axis=(0, 1))   # ali je ta sosed zadetek
            both_hit = hit_mask & shifted_hit   # oba piksla morata biti zadetka
            depth_jump |= both_hit & (np.abs(finite_distance - shifted_distance) > flying_pixel_depth_jump_mm)   # velik skok globine = diskontinuiteta
            max_neighbor_distance = np.where(shifted_hit, np.maximum(max_neighbor_distance, shifted_distance),
                                             max_neighbor_distance)   # obdržimo najbolj oddaljenega soseda
        is_flying = depth_jump & hit_mask & (np.random.random(t_hit.shape) < flying_pixel_probability)   # kateri piksli dejansko postanejo leteči
        if is_flying.any():
            blend = np.random.uniform(0.2, 0.8, size=t_hit.shape)   # naključen delež mešanja med bližnjo in daljno globino
            flying_distance = distance_mm_noisy * (1.0 - blend) + max_neighbor_distance * blend   # vmesna, neresnična globina
            distance_mm_noisy = np.where(is_flying, flying_distance, distance_mm_noisy)   # uveljavimo samo na letečih pikslih

        # Kvantizacija: ADC/disparity ločljivost senzorja poroča globino v
        # diskretnih korakih, ne zvezno - zaokrožitev PO vsem šumu zgoraj
        # (ne prej, sicer bi šum na vrhu spet zabrisal korake). Velikost
        # koraka se z razdaljo skalira enako kot std šuma.
        step_mm = np.maximum(
            quantization_step_at_1m * np.power(np.maximum(distance_m, 1e-6), noise_distance_power), 1e-9)   # korak kvantizacije za vsak piksel
        distance_mm_noisy = np.round(distance_mm_noisy / step_mm) * step_mm   # zaokrožimo na najbližji korak

        t_hit_noisy = np.where(hit_mask, distance_mm_noisy / np.maximum(dir_length, 1e-12), t_hit)   # nazaj v t_hit enote

    hit_points = rays_np[..., :3] + rays_np[..., 3:6] * t_hit_noisy[..., None]  # izvor žarka + t_hit_noisy * smer žarka = 3D točka zadetka
    points = hit_points[hit_mask]   # obdržimo samo dejanske zadetke

    pcd = o3d.geometry.PointCloud()   # ustvarimo nov, prazen oblak točk
    pcd.points = o3d.utility.Vector3dVector(points)   # vstavimo zašumljene točke
    return pcd   # vrnemo simuliran skeniran oblak točk

# za nalaganje realnega skeniranja iz datoteke, za pol k bomo realno skeniral
def load_real_scan(scan_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(scan_path))   # preberemo datoteko z realnim skenom
    if pcd.is_empty():   # datoteka ne obstaja ali je prazna
        raise FileNotFoundError(f"Real scan not found: {scan_path}")
    return pcd   # vrnemo prebran oblak točk


def load_and_merge_real_scans(scan_paths: list[Path]) -> o3d.geometry.PointCloud:
    """Združi več zaporednih skenov istega STATIČNEGA dela v en oblak točk,
    brez kakršnekoli med-sken registracije - ker gre za isti fiksno
    montiran senzor, ki skenira isto nepremaknjeno postavitev (del na mizi
    se med zajemi ne premakne), so zaporedni zajemi že v istem
    koordinatnem sistemu, zgolj konkatenacija zadostuje.

    Namerno NE downsampliramo tukaj: kasnejši voxel_down_sample v Phase 3
    predobdelavi (preprocess_point_cloud) POVPREČI vse točke, ki padejo v
    isto celico, kar samo od sebe povpreči točke iz različnih zajemov, ki
    merijo isto fizično lokacijo - temporalno povprečenje nekoreliranega
    šuma senzorja med N zaporednimi zajemi zniža njegov std za približno
    sqrt(N), praktično zastonj za del, ki med zajemi miruje na mizi.
    Ločen downsample tukaj bi to preprosto podvojil brez dodatne koristi."""
    combined = o3d.geometry.PointCloud()
    total_points = 0
    for scan_path in scan_paths:
        pcd = load_real_scan(scan_path)   # vsak sken naložimo in preverimo posebej (load_real_scan sam preveri prazno/manjkajočo datoteko)
        total_points += len(pcd.points)
        combined += pcd   # združimo v en oblak - že v istem koordinatnem sistemu, brez potrebe po registraciji
    print(f"  Merged {len(scan_paths)} scans ({[p.name for p in scan_paths]}): "
          f"{total_points} points total (temporal noise averaging happens naturally "
          f"in Phase 3's voxel downsampling)")
    return combined   # vrnemo združen, še ne zmanjšan oblak - Phase 3 ga bo voxel-zmanjšala in s tem povprečila šum


# za filtriranje oblaka točk, da odstranimo statistične odstopanja (outlierje),
# odstrani napake pri skeniranje, ko je kakšna točka ki močno odstopa jo odstani
def filter_scan(pcd: o3d.geometry.PointCloud,
                nb_neighbors: int = 30,
                std_ratio: float = 1.5) -> o3d.geometry.PointCloud:
    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)   # odstranimo statistične osamelce
    return filtered   # vrnemo očiščen oblak točk


def filter_grazing_incidence_points(pcd: o3d.geometry.PointCloud,
                                    camera_location: np.ndarray,
                                    max_incidence_deg: float = 80.0,
                                    normal_radius: float = 1.0,
                                    max_nn: int = 30) -> o3d.geometry.PointCloud:
    """Odstrani domnevne leteče piksle na podlagi kota vpadanja: točka,
    katere lokalna normala je skoraj pravokotna na smer pogleda kamere
    (kot vpadanja nad max_incidence_deg, tipično ~80 stopinj), je z veliko
    verjetnostjo leteč piksel na robu (rob dela/pina/šiva - natanko to
    fizikalno ozadje simulira flying_pixel model v simulate_camera_scan)
    namesto prave meritve površine, saj pri tako strmem kotu senzorjevo
    matching okno hkrati zajame obe sosednji globinski ravni.

    filter_scan (statistični outlier removal, razdalja do k najbližjih
    sosedov) tega zanesljivo NE pokrije: leteči piksli na robu tvorijo
    LOKALNO GOSTO skupino vzdolž celotnega roba (interpolirane vrednosti
    med dvema sosednjima površinama), ne izoliranih osamelcev, zato jih
    statistika sosednje razdalje pogosto ne zazna kot "oddaljene" - to je
    standarden dodaten filter pri strukturirani svetlobi/laserskih
    skenerjih, ki cilja na fizikalni vzrok (kot vpadanja), ne na
    geometrijski simptom (izolirana oddaljenost)."""
    pcd_normals = copy.deepcopy(pcd)   # ne spreminjamo izvirnika
    pcd_normals.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))   # lokalne normale za oceno kota vpadanja
    pcd_normals.orient_normals_towards_camera_location(camera_location)   # usmerimo jih proti kameri, da je kot vpadanja pravilno definiran
    points = np.asarray(pcd_normals.points)
    normals = np.asarray(pcd_normals.normals)
    view_dir = camera_location - points   # smer od vsake točke proti kameri
    view_dir /= np.clip(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-12, None)   # normaliziramo na enotsko dolžino
    incidence_cos = np.clip(np.abs(np.sum(view_dir * normals, axis=-1)), -1.0, 1.0)   # kosinus kota med pogledom in normalo
    incidence_deg = np.degrees(np.arccos(incidence_cos))   # kot vpadanja v stopinjah
    keep_mask = incidence_deg <= max_incidence_deg   # obdržimo samo dovolj "naravnost" gledane točke
    kept = pcd.select_by_index(np.where(keep_mask)[0])   # izberemo iz IZVIRNEGA oblaka (brez tu izračunanih normal)
    print(f"  filter_grazing_incidence_points: removed {len(points) - len(kept.points)}/{len(points)} "
          f"points with incidence angle > {max_incidence_deg:.0f} deg "
          f"({100 * len(kept.points) / max(len(points), 1):.1f}% kept)")
    return kept   # vrnemo oblak brez domnevnih letečih pikslov na robovih


def remove_table_background(target: o3d.geometry.PointCloud,
                            up_axis: int,
                            flip_up_direction: bool,
                            part_height_mm: float,
                            cropped_region_height_mm: Optional[float] = None,
                            margin_mm: float = 10.0,
                            table_level_percentile: float = 1.0) -> o3d.geometry.PointCloud:
    """Odstrani domnevne mizne/ozadje točke iz target, na podlagi ZNANE
    višine dela (part_height_mm, iz source/CAD modela - vedno na voljo in
    natančno znana, ne glede na velikost ali obliko dela v XY), namesto
    zaznavanja "največje ravnine" (npr. Open3D-jev segment_plane, RANSAC).

    Zakaj ne "največja ravnina": če del zaseda velik del vidnega polja
    kamere (velik kos, ali del z veliko ravno zgornjo "sealing" površino -
    natanko taka, kot jo ta cevovod že skenira), bi RANSAC lahko namesto
    mize po nesreči zaznal in odstranil TA del kot "največjo ravnino". Ta
    filter se sploh ne opira na relativno velikost ravnin v sceni, zato je
    od velikosti/oblike dela neodvisen.

    Namesto tega izkoristimo fizikalno dejstvo, da del ne more segati 
    mizo, na kateri leži - miza (če je v skenu sploh prisotna) je zato
    vedno pri najnižjih (ali najvišjih, glede na flip_up_direction)
    opaženih vrednostih vzdolž up_axis. table_level_percentile (namesto
    golega min()) ublaži vpliv posameznih osamelcev/letečih pikslov, ki bi
    sicer navidezno postavili "mizo" veliko nižje, kot dejansko je.
    Zgornja meja obdržanega pasu (miza + part_height_mm + margin_mm)
    dodatno odreže ekstremne osamelce, ki segajo višje, kot bi del sploh
    lahko fizično segal.

    cropped_region_height_mm (neobvezno) ZOŽI pas ŠE OD SPODAJ: kamera
    dejansko skenira SAMO zgornjo top_fraction regijo dela (glej
    crop_top_region) - source je obrezan na natanko to regijo, zato vse
    MED mizo in (vrh dela - ta regija) v target-u ni ne miza ne kaj, s čimer
    bi se source sploh lahko ujemal: gre za poševno vidne stranske stene
    dela ("zavesa") ali prazen prostor - v obeh primerih samo šum/osamelci
    za ICP, ne uporabna geometrija. Če je podano, se spodnja meja namesto
    "tik nad mizo" postavi na
    table_level + part_height_mm - cropped_region_height_mm - margin_mm
    (z varovalko navzdol na table_level + margin_mm, če bi bil
    cropped_region_height_mm blizu part_height_mm - tedaj ni prave
    "zavese" za odrezati, obnašanje se prelije nazaj v prejšnjo mejo tik
    nad mizo). None (privzeto) ohrani prejšnje obnašanje nespremenjeno."""
    points = np.asarray(target.points)
    coord = points[:, up_axis] if not flip_up_direction else -points[:, up_axis]   # normaliziramo tako, da je miza vedno pri min(coord)
    table_level = np.percentile(coord, table_level_percentile)   # ocena višine mize, robustna na osamelce
    just_above_table = table_level + margin_mm   # tik nad mizo - prejšnja (in varovalna) spodnja meja
    if cropped_region_height_mm is not None:
        # Zoženo od spodaj: obdržimo samo pas okoli DEJANSKO skenirane
        # (top_fraction) regije, ne cele višine dela od mize navzgor -
        # max(...) zagotovi, da nikoli ne pademo pod "tik nad mizo", tudi
        # če je cropped_region_height_mm blizu part_height_mm.
        lower_cutoff = max(just_above_table,
                          table_level + part_height_mm - cropped_region_height_mm - margin_mm)
    else:
        lower_cutoff = just_above_table
    upper_cutoff = table_level + part_height_mm + margin_mm   # nad tem del fizično ne more segati
    keep_mask = (coord > lower_cutoff) & (coord < upper_cutoff)   # obdržimo samo pas, kjer lahko biva del
    kept = target.select_by_index(np.where(keep_mask)[0])
    print(f"  remove_table_background: table_level={table_level:.2f}, "
          f"keeping band ({lower_cutoff:.2f}, {upper_cutoff:.2f}) - "
          f"kept {len(kept.points)}/{len(points)} points "
          f"({100 * len(kept.points) / max(len(points), 1):.1f}%)")
    return kept   # vrnemo target brez domnevnih mizno/ozadje točk


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
        # En sam pogled kamere (pravi ali simuliran) vedno vidi le eno
        # stran objekta, zato mora vsaka normala kazati nazaj proti
        # senzorju.
        pcd.orient_normals_towards_camera_location(camera_location)   # obrnemo vse normale proti kameri
    else:
        # CAD model je cela, zaprta površina - ni ene same "strani proti
        # kameri", zato namesto tega razširimo globalno usklajeno
        # orientacijo čez celo površino. To se obenem izogne
        # nekonsistentnemu vrstnemu redu oglišč (winding), ki bi lahko bil
        # zapisan v STL datoteki.
        pcd.orient_normals_consistent_tangent_plane(max_nn)   # globalno usklajena orientacija normal


def crop_top_region(pcd: o3d.geometry.PointCloud,  # vhodni oblak točk, ki ga želimo odrezat
                    top_fraction: float = 0.35,     # obdrži 35% zgornjega dela oblaka točk
                    up_axis: int = 2,               # os, ki predstavlja 'gor' (0=x, 1=y, 2=z), privzeto je z os
                    flip_up_direction: bool = False) -> o3d.geometry.PointCloud:
    min_bound = pcd.get_min_bound()         # poišče skrajne točke oblaka točk
    max_bound = pcd.get_max_bound()   # zgornja meja oblaka točk po vseh oseh
    if not flip_up_direction:               # sam izračuna kok je treba odrezat
        cutoff = max_bound[up_axis] - top_fraction * (max_bound[up_axis] - min_bound[up_axis])   # meja, pod katero obrežemo
        crop_min = min_bound.copy()   # spodnja meja izreza - dvignjena na cutoff
        crop_min[up_axis] = cutoff
        bbox = o3d.geometry.AxisAlignedBoundingBox(crop_min, max_bound)   # škatla, ki zajame samo zgornji del
    else:                               # isto sam obratno, če je flip_up_direction=True
        cutoff = min_bound[up_axis] + top_fraction * (max_bound[up_axis] - min_bound[up_axis])   # meja, nad katero obrežemo
        crop_max = max_bound.copy()   # zgornja meja izreza - znižana na cutoff
        crop_max[up_axis] = cutoff
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, crop_max)   # škatla, ki zajame samo spodnji del
    return pcd.crop(bbox)   # obrežemo oblak točk na to škatlo


def compute_native_reference_point(mesh: o3d.geometry.TriangleMesh,
                                   top_fraction: float,
                                   up_axis: int,
                                   flip_up_direction: bool) -> np.ndarray:
    """Izračuna referenčno točko (X,Y,Z) v NATIVE (NEtransformiranem) CAD
    okviru dela - geometrijsko središče zgornjega ("sealing") prereza, ki
    ga simulate_camera_scan dejansko zajame (ista crop_top_region logika).

    ZAKAJ JE TO POTREBNO: main()-ovi --translation_x/y/z naj bi opisovali
    dejansko pozicijo/razdaljo TE skenirane površine glede na fiksno
    kamero (translation_x=y=0 => del na sredini kamerinega pogleda,
    translation_z=-D => skenirana površina D mm pod kamero) - a CAD-ov
    lastni izvor koordinat NI nujno poravnan niti s centrom tega prereza
    (X/Y) niti z njegovo globinsko sredino (Z). Brez tega popravka bi
    surova uporaba CAD-ovega izvora pomenila, da translation_x=0,y=0 NE
    postavi dela na sredino pogleda in da translation_z NE ustreza pravi
    razdalji do skenirane površine - kar neposredno pokvari, kako natančno
    simulacija ustreza kamerini specificirani "scanning area" (glej
    kamere.py) - simulirano vidno polje bi bilo lahko ožje/širše in na
    napačni razdalji centrirano, kot ga specifikacija dejansko obljublja.

    main() ta referenčni odmik izračuna ENKRAT (na netransformirani mreži,
    z rotation=0) in ga odšteje od R-rotiranega uporabniškega translation_x/
    y/z, PREDEN se sestavi končni SIMULATED_TRANSFORM - tako je popravek
    zajet v ENI sami, dosledni transformacijski matriki, ki jo
    simulate_camera_scan, transformation_error itd. vsi enako uporabijo,
    namesto ločenega, lahko neusklajenega popravka na več mestih."""
    cropped = crop_top_region(mesh, top_fraction=top_fraction, up_axis=up_axis,
                              flip_up_direction=flip_up_direction)   # ista "sealing" regija, ki jo simulate_camera_scan dejansko zajame
    return np.asarray(cropped.get_center())   # geometrijsko središče te regije v CAD-ovem lastnem (netransformiranem) okviru

# TO JE SINTETIČNA KAMERA, KI JO BOMO POTREBOVAL TUDI PRI REALNEM SKENIRANJU
# NAMEN JE, DA SE NORMALE OBLAKA TOČK CAD MODELA OBRNEJO V ISTO SMER KOT BODO OD SKENIRANGA POINT CLOUDA
# ČE TE FUNKCIJE NI SO NORMALE OBLAKA TOČK SOURCE MODELA OBRNJENE V NAPAČNO SMER ( NE MORE PRIMERJAT S TARGET MODELOM)
def top_camera_location(pcd: o3d.geometry.PointCloud,  # vhodni oblak točk, ki ga želimo uporabiti za določitev lokacije kamere
                        up_axis: int = 2,   # os, ki predstavlja "gor" (0=x, 1=y, 2=z), privzeto je z os
                        offset_factor: float = 5.0,     # faktor, ki določa, kako daleč nad oblakom točk bo kamera postavljena (v enotah dolžine oblaka točk) - uporabljen samo, če offset_distance ni podan
                        flip_up_direction: bool = False,
                        offset_distance: Optional[float] = None) -> np.ndarray:  # če je podan (v mm), je kamera postavljena na to fiksno razdaljo nad/pod oblakom točk, NE glede na velikost oblaka - hidden_point_removal je le kvalitativno občutljiv na razdaljo kamere (ne potrebuje prave perspektivne geometrije), zato poljuben "dovolj velik" offset_factor deluje enako dobro kot prava fizična razdalja za vse dosedanje klice; za statično kamero (glej main()) pa mora razdalja odražati realno fizično postavitev (~1m), ne pa se raztezati/krčiti z velikostjo modela
    center = pcd.get_center()           # prostorska sredina oblaka točk
    extent = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())     # razlika med največjo in najmanjšo mejo oblaka točk, da dobimo velikost oblaka točk
    camera_location = np.array(center)   # začnemo s pozicijo v sredini oblaka
    distance = offset_distance if offset_distance is not None else offset_factor * max(extent[up_axis], 1.0)   # izberemo fiksno ali od velikosti odvisno razdaljo
    if not flip_up_direction:
        camera_location[up_axis] = pcd.get_max_bound()[up_axis] + distance   # kamera nad zgornjo mejo
    else:
        camera_location[up_axis] = pcd.get_min_bound()[up_axis] - distance   # kamera pod spodnjo mejo
    return camera_location   # vrnemo izračunano pozicijo kamere


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float,
                           is_partial_view: bool = False,
                           camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),
                           cache_key: Optional[str] = None,
                           use_cache: bool = True) -> o3d.geometry.PointCloud:
    """Zmanjša gostoto pcd (voxel downsampling) in izračuna pravilno
    orientirane normale - vse, kar yaw_sweep_registration in
    večnivojski ICP dejansko potrebujeta.

    cache_key identificira *vhod* (npr. "source_partname_20000pts_top0.35_axis2") -
    skupaj z vsemi parametri spodaj, ki vplivajo na izhod, tvori pot do
    predpomnilnika.
    """
    cache_path = None
    if cache_key is not None and use_cache:   # predpomnilnik je omogočen in imamo ključ
        params = (PREPROCESS_CACHE_VERSION, voxel_size, is_partial_view,
                  tuple(np.round(np.asarray(camera_location), 3)))   # vsi parametri, ki vplivajo na izhod
        digest = hashlib.md5(repr(params).encode()).hexdigest()[:10]   # kratek hash teh parametrov
        PREPROCESS_CACHE_DIR.mkdir(exist_ok=True)   # ustvarimo mapo za predpomnilnik, če še ne obstaja
        # cache_key je berljiv za človeka (npr. "source_partname_20000pts_top0.35_axis2"),
        # a lahko postane dolg - samo ime CAD datoteke plus vsaka test-tilt/
        # test-translation oznaka, ki jo doda klicatelj run_registration, lahko
        # potisne celotno pot čez Windows-ovo mejo 260 znakov (MAX_PATH), zaradi
        # česar Open3D-jev write_point_cloud spodleti z "invalid wchar_t
        # filename argument" (nepovedna napaka o TEM, zakaj). Hash samega
        # cache_key ohrani ime datoteke kratko in neodvisno od tega, kako
        # opisen cache_key postane; skrajšan berljiv predpona je zraven
        # ohranjena zgolj za brskanje po .preprocess_cache/ na oko, ne za
        # enoličnost (to zagotavlja hash).
        cache_key_digest = hashlib.md5(cache_key.encode()).hexdigest()[:12]   # hash celotnega cache_key
        readable_prefix = "".join(c if c.isalnum() else "_" for c in cache_key)[:40]   # skrajšan berljiv del imena
        cache_path = PREPROCESS_CACHE_DIR / f"{readable_prefix}_{cache_key_digest}_{digest}"   # pot brez pripone
        cloud_path = cache_path.parent / (cache_path.name + ".ply")   # dodamo pripono .ply
        if cloud_path.exists():   # predpomnjena datoteka že obstaja
            pcd_down = o3d.io.read_point_cloud(str(cloud_path))   # preberemo jo
            if not pcd_down.is_empty():
                print(f"    Loaded cached preprocessing for '{cache_key}': {len(pcd_down.points)} points")
                return pcd_down   # vrnemo predpomnjen rezultat, brez ponovnega računanja

    extent = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())   # velikost oblaka po vsaki osi
    diagonal = np.linalg.norm(extent)   # dolžina diagonale oblaka
    print(f"    Input cloud extent (xyz): {extent}, diagonal={diagonal:.3f}, "
          f"voxel_size={voxel_size} -> ~{diagonal / voxel_size:.0f} voxels across the diagonal")

    pcd_down = pcd.voxel_down_sample(voxel_size)    # naredi 3d mrežo kock, v vsaki kocki vzame eno točko (povprečje), da zmanjša število točk in pospeši izračune

    ensure_oriented_normals(pcd_down, normal_radius=voxel_size * 2.0, is_partial_view=is_partial_view,
                            camera_location=camera_location) # pokliče funkcijo od prej

    if cache_path is not None:   # shranimo rezultat za naslednji zagon
        o3d.io.write_point_cloud(str(cache_path.parent / (cache_path.name + ".ply")), pcd_down)

    return pcd_down   # vrnemo zmanjšan oblak z izračunanimi normalami


def estimate_surface_normal_ransac(points: np.ndarray,
                                   up_axis: int,
                                   top_band_fraction: float = 0.5,
                                   distance_threshold: Optional[float] = None,
                                   num_iterations: int = 1000) -> np.ndarray:
    """RANSAC-prileganje ravnine na zgornji pas točk, namesto golega PCA
    čez CEL oblak.

    Gol PCA (prejšnja implementacija, glej estimate_surface_normal spodaj -
    ohranjena samo za primerjavo/debug) nima robustne uteži: vsaka točka
    prispeva enako v kovariančno matriko, zato en sam gost šop točk zunaj
    prave zgornje ploskve dela (najpogosteje MIZA, če remove_table_background
    ni bila poklicana ali ni bila dovolj agresivna) lahko oceno normale
    zavede proti mizi namesto proti dejanski nagnjeni zgornji ploskvi dela -
    naklon dela se s tem v celoti izgubi. RANSAC namesto tega poišče
    ravnino, ki jo podpira NAJVEČ soglasnih (inlier) točk, in vse ostalo
    (miza, robovi, leteči piksli, drugi osamelci) preprosto zavrže -
    bistveno bolj robustno na to, da je ozadje/miza v oblaku sploh
    prisotno.

    top_band_fraction dodatno omeji vhod RANSAC-u na zgornji pas točk (po
    percentilu vzdolž up_axis) - tudi če RANSAC sam prenese nekaj mizinih
    točk v vhodu, jih ta pred-filter v veliki večini primerov že izloči,
    preden RANSAC sploh steče, kar zniža število potrebnih iteracij za
    zanesljivo konvergenco."""
    coord = points[:, up_axis]
    band_cutoff = np.percentile(coord, 100.0 * (1.0 - top_band_fraction))   # meja zgornjega pasu po percentilu
    band_points = points[coord >= band_cutoff]   # obdržimo samo zgornji pas točk
    if len(band_points) < 3:   # premalo točk za prileganje ravnine (RANSAC potrebuje vsaj 3)
        band_points = points   # nazaj na cel oblak kot skrajna rešitev
    if distance_threshold is None:
        extent = np.linalg.norm(band_points.max(axis=0) - band_points.min(axis=0))   # velikost pasu, za smiseln privzet prag
        distance_threshold = max(extent * 0.005, 0.1)   # 0.5% velikosti pasu, vsaj 0.1 (enote oblaka točk)
    band_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(band_points))   # zgornji pas kot samostojen oblak za segment_plane
    plane_model, _inliers = band_pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)   # RANSAC prileganje ravnine
    normal = np.array(plane_model[:3])   # normala ravnine (prva 3 koeficienta ax+by+cz+d=0)
    return normal / np.linalg.norm(normal)   # vrnemo normaliziran (enotski) vektor normale


def estimate_surface_normal(points: np.ndarray) -> np.ndarray:
    """PCA ocena normale: smer, v kateri točke variirajo NAJMANJ. Nima
    robustne uteži do osamelcev/mize v oblaku (za razliko od
    estimate_surface_normal_ransac zgoraj, ki jo dejansko uporablja
    yaw_sweep_registration) - ohranjena zgolj za primerjavo/debug, ne
    kliče je noben produkcijski del kode."""
    centered = points - points.mean(axis=0)   # točke premaknemo tako, da je njihovo povprečje v izhodišču
    cov = centered.T @ centered   # kovariančna matrika (kako točke variirajo v vsaki smeri)
    eigvals, eigvecs = np.linalg.eigh(cov)   # lastne vrednosti in lastni vektorji kovariančne matrike
    normal = eigvecs[:, 0]  # eigh sorts ascending - smallest eigenvalue first
    return normal / np.linalg.norm(normal)   # vrnemo normaliziran (enotski) vektor normale


def robust_top_value(values: np.ndarray, percentile: float) -> float:
    """Visok percentil namesto golega max() za oceno "vrha" množice
    vrednosti (npr. višina zgornje ploskve dela vzdolž up_axis).

    max() je po definiciji določen z EN(o) TOČKO - v lastnem šumovnem
    modelu tega cevovoda (glej simulate_camera_scan) lahko osamelec/leteč
    piksel doseže std tudi do outlier_std_multiplier (privzeto 15x) širši
    od navadnega šuma tega piksla, zato lahko ena sama taka točka premakne
    max() (in s tem celotno ocenjeno Z-translacijo v build_transform
    spodaj) za precej mm. Visok percentil (privzeto 99.5) namesto tega
    obravnava vrh kot pas zgornjih nekaj promilov točk, en sam osamelec
    ga komaj premakne."""
    return float(np.percentile(values, percentile))


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodriguesova rotacijska formula: rotacijska matrika, ki preslika
    enotski vektor a na enotski vektor b (rotacija z najmanjšim kotom, ki
    to doseže)."""
    a = a / np.linalg.norm(a)   # normaliziramo vektor a na enotsko dolžino
    b = b / np.linalg.norm(b)   # normaliziramo vektor b na enotsko dolžino
    v = np.cross(a, b)   # os rotacije (pravokotna na a in b)
    s = np.linalg.norm(v)   # sinus kota med a in b
    c = np.dot(a, b)   # kosinus kota med a in b
    if s < 1e-8:   # a in b sta že (skoraj) vzporedna ali nasprotna
        if c > 0:
            return np.eye(3)   # a in b že kažeta v isto smer - ni potrebna rotacija
        # a in b sta antiparalelna - katerakoli os pravokotna na a deluje
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])   # pomožen vektor, ki ni vzporeden z a
        axis = np.cross(a, perp)   # poljubna os pravokotna na a
        axis /= np.linalg.norm(axis)   # normaliziramo os
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])   # poševno-simetrična matrika osi
        return np.eye(3) + 2 * (K @ K)   # rotacija za natanko 180 stopinj okoli te osi
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])   # poševno-simetrična matrika osi v
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))   # Rodriguesova formula - sestavljena rotacijska matrika


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodriguesova rotacijska formula: rotacija za angle_deg okoli
    poljubne enotske osi (posplošitev čiste Z rotacije na poljubno os, saj
    ocena naklona pomeni, da "gor" ni več nujno natanko os Z)."""
    axis = axis / np.linalg.norm(axis)   # normaliziramo os na enotsko dolžino
    theta = np.radians(angle_deg)   # pretvorimo kot iz stopinj v radiane
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])   # poševno-simetrična matrika osi
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)   # Rodriguesova formula - rotacijska matrika za kot theta


def build_height_map_grid(points_2d: np.ndarray, heights: np.ndarray, origin: np.ndarray,
                          resolution: float, grid_shape: tuple[int, int]) -> np.ndarray:
    """Rasterizira 2D točke v mrežo, vsaka celica hrani max višino točk, ki
    padejo vanjo (mapa "kako visok je del tukaj") - uporablja jo
    estimate_xy_translation_phase_correlation spodaj. Celice, kamor ne
    pade nobena točka, ostanejo 0, kar je ločeno od pravih (vedno
    pozitivnih, glej klicatelja) višin."""
    idx = np.floor((points_2d - origin) / resolution).astype(int)   # indeks celice mreže za vsako točko
    valid = (idx[:, 0] >= 0) & (idx[:, 0] < grid_shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < grid_shape[1])   # točke, ki padejo znotraj mreže
    grid = np.zeros(grid_shape, dtype=np.float64)   # prazna mreža, začetno vse 0
    if valid.any():
        np.maximum.at(grid, (idx[valid, 0], idx[valid, 1]), heights[valid])   # v vsako celico vpišemo max višino
    return grid   # vrnemo height-map mrežo


def estimate_xy_translation_phase_correlation(source_pts: np.ndarray,
                                              target_pts: np.ndarray,
                                              up_axis: int,
                                              resolution: float,
                                              margin_factor: float = 1.5) -> np.ndarray:
    """Oceni translacijo v ravnini pravokotni na up_axis (X/Y za up_axis=2)
    prek 2D FFT fazne korelacije med projekcijami height-map, namesto
    surove razlike centroidov.

    Ujemanje centroidov povpreči pozicijo vsake vidne točke - okluzija, ki
    obreže eno stran obrisa bolj kot drugo (samo-okluzija zaradi zasukanega
    dela, ali kamera, ki ni točno nad delom, ko je ta pomaknjen na stran -
    glej test tolerance X/Y pozicije), to povprečje pristransko premakne ne
    le vzdolž up_axis (to je že obravnavano ločeno prek primerjave max
    višine v yaw_sweep_registration), ampak tudi v pravokotni ravnini.
    Fazna korelacija namesto tega ujema celotno OBLIKO obrisa - veliko
    večji, bolj redundanten signal, ki ga peščica manjkajočih robnih točk
    komaj moti, podobno kot registracija slik ni zmedena zaradi delno
    obrezane fotografije. Validirano na sintetičnih podatkih z 40% točk,
    odstranjenih z ene strani asimetričnega obrisa: fazna korelacija je
    povrnila pravi zamik na 0.03 enote natančno, v primerjavi z 13+
    enotami napake pri navadni razliki centroidov.

    Vsak oblak se rasterizira v svojo height-map mrežo
    (build_height_map_grid), sidrano na svoj bounding box plus margina -
    ne na skupno svetovno izhodišče, saj je ta zamik natanko tisto
    neznano, ki ga funkcija rešuje. Deljenje cross-power spektra z lastno
    magnitudo (od tod "faza" v fazni korelaciji) spremeni čisto translacijo
    med mrežama v en oster korelacijski vrh, robusten na to, da imata
    mreži različno gostoto/pokritost točk (navadna cross-korelacija bi bila
    na to občutljiva). Lokacija vrha da OSTANEK zamika znotraj lokalnega
    okvirja vsake mreže; dodajanje razlike med izhodiščema bounding boxov
    obeh mrež nazaj povrne dejansko metrično translacijo.
    """
    plane_axes = [a for a in range(3) if a != up_axis]   # dve osi pravokotni na up_axis (npr. X,Y za up_axis=Z)
    src_2d = source_pts[:, plane_axes]   # projekcija source točk na to ravnino
    tgt_2d = target_pts[:, plane_axes]   # projekcija target točk na to ravnino
    # Zamaknjeno tako, da je striktno pozitivno znotraj vsakega oblaka
    # (lasten min -> resolution, ne 0), da prave-a-nizke višine nikoli ne
    # zamenjamo s prazno celico.
    src_h = source_pts[:, up_axis] - source_pts[:, up_axis].min() + resolution   # višina source točk, vedno > 0
    tgt_h = target_pts[:, up_axis] - target_pts[:, up_axis].min() + resolution   # višina target točk, vedno > 0

    # Vsaka mreža je velika glede na svoj obris plus margina, da ima
    # (neznan) ostanek zamika, ki ga rešuje fazna korelacija, dovolj
    # prostora, ne da bi se ovil čez krožno mejo FFT.
    src_extent = src_2d.max(axis=0) - src_2d.min(axis=0)   # velikost obrisa source
    tgt_extent = tgt_2d.max(axis=0) - tgt_2d.min(axis=0)   # velikost obrisa target
    span = np.maximum(src_extent, tgt_extent) * margin_factor   # velikost mreže z margino
    grid_shape = tuple(np.maximum(np.ceil(span / resolution).astype(int), 8))   # dimenzije mreže v pikslih

    src_origin = src_2d.min(axis=0) - (np.array(grid_shape) * resolution - src_extent) / 2.0   # izhodišče mreže source, obris centriran
    tgt_origin = tgt_2d.min(axis=0) - (np.array(grid_shape) * resolution - tgt_extent) / 2.0   # izhodišče mreže target, obris centriran

    src_grid = build_height_map_grid(src_2d, src_h, src_origin, resolution, grid_shape)   # height-map mreža source
    tgt_grid = build_height_map_grid(tgt_2d, tgt_h, tgt_origin, resolution, grid_shape)   # height-map mreža target

    f_src = np.fft.fft2(src_grid)   # 2D Fourierova transformacija source mreže
    f_tgt = np.fft.fft2(tgt_grid)   # 2D Fourierova transformacija target mreže
    cross_power = f_src * np.conj(f_tgt)   # cross-power spekter
    magnitude = np.abs(cross_power)   # magnituda spektra
    magnitude[magnitude < 1e-10] = 1e-10  # preprečimo deljenje s skoraj 0 v območjih spektra brez prekrivanja
    correlation = np.fft.ifft2(cross_power / magnitude).real   # normalizirana (fazna) korelacija nazaj v prostorsko domeno

    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape), dtype=float)   # pozicija najvišjega vrha korelacije
    for i in range(2):
        # ifft2 vrne zamike v [0, grid_shape) - vse čez polovico previjemo
        # nazaj na ustrezen negativen zamik.
        if peak[i] > grid_shape[i] / 2:
            peak[i] -= grid_shape[i]
    residual_shift = peak * resolution   # ostanek zamika v pravih enotah (mm)

    return residual_shift + (tgt_origin - src_origin)   # ostanek + razlika izhodišč = dejanska translacija


def _evaluate_yaw_candidate(yaw_deg: float,
                            source_pts: np.ndarray,
                            target_pts: np.ndarray,
                            source_center: np.ndarray,
                            target_normal: np.ndarray,
                            tilt_correction: np.ndarray,
                            plane_axes: list[int],
                            up_axis: int,
                            translation_grid_resolution: float,
                            height_percentile: float,
                            table_center: np.ndarray,
                            table_half_extent: np.ndarray,
                            distance_threshold: float,
                            refine_iterations: int) -> Optional[tuple[float, float, float, np.ndarray]]:
    """Oceni EN yaw kandidat - izločeno iz yaw_sweep_registration kot
    samostojna, MODULSKA (ne znotraj nje zaprta) funkcija, da jo je mogoče
    poslati v ločen proces (glej yaw_sweep_registration - joblib.Parallel).

    Zaprtja (closures, kot je bil prejšnji `build_transform`) in Open3D
    objekti (PointCloud, ICPConvergenceCriteria, TransformationEstimation*)
    se NE dajo picklati (preverjeno empirično - poskus pickla vrže
    "cannot pickle ... object"), zato ta funkcija prejme SAMO navadne
    numpy/python vrednosti in si Open3D objekte, ki jih rabi, zgradi SAMA,
    znotraj delovnega procesa - source_down/target_down se zato tu
    zgradita na novo iz surovih točk (poceni, O(n), zanemarljivo v
    primerjavi z ICP klicem samim).

    Vrne None, če fizični prior (implicirana pozicija izven mize) zavrne
    tega kandidata - klicatelj tak rezultat preprosto prezre. Sicer vrne
    (yaw_deg, fitness, inlier_rmse, transformation)."""
    R = rotation_about_axis(target_normal, yaw_deg) @ tilt_correction   # kombinirana rotacija: naklon + yaw
    T = np.eye(4)
    T[:3, :3] = R
    source_pts_rotated = (R @ source_pts.T).T   # source točke, zasukane s kandidatno rotacijo

    translation = np.zeros(3)
    translation[plane_axes] = estimate_xy_translation_phase_correlation(
        source_pts_rotated, target_pts, up_axis, translation_grid_resolution)   # X/Y del translacije
    translation[up_axis] = (robust_top_value(target_pts[:, up_axis], height_percentile) -
                            robust_top_value(source_pts_rotated[:, up_axis], height_percentile))   # up_axis del translacije
    T[:3, 3] = translation

    part_center = (T[:3, :3] @ source_center) + T[:3, 3]   # kam bi ta transformacija postavila center dela
    if np.any(np.abs(part_center[plane_axes] - table_center) > table_half_extent):   # izven mize - fizično nemogoče
        return None

    # np.array(..., copy=True) - joblib (loky backend) prenese velike numpy
    # tabele delovnim procesom prek BRALNO-ZAŠČITENEGA (read-only) memory-mapa
    # (za učinkovitost, brez ponovnega kopiranja) - o3d.utility.Vector3dVector
    # pa zahteva zapisljiv buffer in vrže "array is not writeable" na takem
    # vhodu. Eksplicitna kopija tu to popravi (poceni - source/target sta že
    # zmanjšana/downsampled na to točko).
    source_down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(source_pts, copy=True)))
    target_down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(target_pts, copy=True)))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=refine_iterations)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, T, estimation, criteria)   # hiter lokalni ICP polish tega kandidata
    return (float(yaw_deg), float(icp_result.fitness), float(icp_result.inlier_rmse),
           np.asarray(icp_result.transformation))


def yaw_sweep_registration(source_down: o3d.geometry.PointCloud,
                           target_down: o3d.geometry.PointCloud,
                           voxel_size: float,
                           up_axis: int = 2,
                           yaw_step_deg: float = 6.0,
                           distance_threshold_factor: float = 3.0,
                           refine_iterations: int = 5,
                           translation_grid_resolution: Optional[float] = None,
                           table_size_x: float = 500.0,
                           table_size_y: float = 500.0,
                           table_center: Optional[np.ndarray] = None,
                           tilt_normal_top_band_fraction: float = 0.5,
                           height_percentile: float = 99.5,
                           n_jobs: int = -1) -> o3d.pipelines.registration.RegistrationResult:
    """Groba globalna registracija, ki izkorišča PRAVO omejitev postavitve
    namesto reševanja slepe 6-DOF registracije: del vedno leži na mizi v
    isti orientaciji z licem navzdol, zato so Z/roll/pitch fiksirani zaradi
    stika z mizo, X/Y sta znana le ohlapno (nekaj deset mm), yaw (rotacija
    okoli up_axis) pa je edini zares prosti parameter.

    Translacija je ocenjena neposredno, ne iskana: ravnina pravokotna na
    up_axis (X/Y) prek 2D fazne korelacije na obliki obrisa
    (estimate_xy_translation_phase_correlation - robustna na asimetrijo
    okluzije, za razliko od surove razlike centroidov), up_axis sam pa
    prek primerjave visokega percentila višine (height_percentile, glej
    build_transform in robust_top_value spodaj - namerno NE gol max(), ki
    bi ga en sam osamelec/leteč piksel premaknil za precej mm) - oboje
    neodvisno od neznanega yaw, zato translacije sploh ni treba iskati.
    Yaw sam je preiskan izčrpno (ne naključno vzorčen), saj gre za en
    omejen, popolnoma naštevljiv parameter.

    Prava postavitev nikoli ni POPOLNOMA ravna (drobec umazanije, rahel
    zvitek), zato ta funkcija oceni in popravi tudi dejanski roll/pitch
    naklon, namesto da bi predpostavila natanko nič: normala površine
    target je ocenjena z RANSAC prileganjem ravnine na njen zgornji pas
    točk (estimate_surface_normal_ransac, tilt_normal_top_band_fraction
    izbere ta pas) - namerno NE gol PCA čez cel target, ki bi ga (če
    remove_table_background ni bila poklicana, ali del zaseda le del
    vidnega polja) zlahka zavedla soprisotnost mize/ozadja v oblaku, saj
    PCA nima robustne uteži do takih osamelcev, RANSAC pa jih po
    konstrukciji izloči kot ne-soglasne (outlier) točke. Rotacija, ki
    poravna znano referenčno smer "gor" source-a s to ocenjeno normalo, pa
    postane osnovna orientacija. Kar ostane - rotacija OKOLI te zdaj
    poravnane osi normale - je natanko yaw negotovost, zato sweep še vedno
    teče, le sestavljen na vrhu popravka naklona namesto ravne
    predpostavke. Empirično testirano do ~25-27 stopinj vbrizganega
    naklona, pred porušitvijo okoli 30 stopinj.

    Vsak grob yaw kandidat dobi HITER lokalni ICP polish
    (refine_iterations, nizek - to ni končni natančnostni korak, to
    naredi celoten večnivojski ICP po tej funkciji) pred ocenjevanjem,
    namesto ocenjevanja surovega nerafiniranega posnetka. To je pomembno
    pri delu s ponavljajočimi se značilnostmi (montažne izbokline): napačen
    yaw lahko poravna ponavljajočo se osnovno geometrijo dovolj dobro, da
    zmaga po surovem ŠTEVILU inlierjev, čeprav popolnoma napačno poravna
    tiste redke asimetrične značilnosti (locator pini, konektor), ki
    dejansko ločijo pravilno orientacijo. Kratka lokalna refinacija
    vsakega kandidata najprej potisne proti njegovemu resničnemu bližnjemu
    optimumu - pravi yaw konvergira v tesno prileganje, zgolj na grobo
    poravnan napačen yaw pa navadno ne, ker se asimetrična podrobnost ne
    poravna, ne glede na to, kako lokalno jo premikamo.

    Podprt je le up_axis=2 (Z) kot REFERENČNA smer "gor" (skladno z vsemi
    drugimi predpostavkami o up-axis v tej kodni bazi) - sam popravek
    naklona deluje za poljubno končno orientacijo, fiksirana je le
    začetna referenčna smer na os Z.

    table_size_x/table_size_y/table_center dodajo fizični prior:
    kombinacija (yaw, dx, dy), ki bi postavila center dela izven znanega
    obsega mize, je fizično nemogoča - del leži na mizi znane velikosti in
    ne more biti izven nje - zato so taki kandidati kar zavrnjeni, namesto
    da bi lahko zmagali zgolj na podlagi ICP fitness. Ker je translacija
    tu IZPELJANA iz yaw (ne iskana neodvisno, glej zgoraj), gre pravzaprav
    za smiselnostno preverjanje implicirane postavitve za vsak yaw posebej.
    table_size_x/y privzeto 500x500mm, table_center privzeto (0,0) -
    prilagodi oboje dejanskemu delovnemu območju pravega robota.

    n_jobs: yaw kandidati so med seboj popolnoma neodvisni (vsak samo bere
    source_pts/target_pts, ne piše vanju), zato jih (prek _evaluate_yaw_candidate
    in joblib.Parallel) ovrednotimo v ločenih procesih - z n_jobs=-1
    (privzeto, joblib-ova konvencija za "vsa jedra") to na več-jedrnem
    stroju pohitri sweep skoraj linearno s številom jeder. Rezultat je
    numerično ENAK serijskemu izvajanju (isti kandidati, ista ICP klica na
    vsakega, le drugačen vrstni red ocenjevanja) - n_jobs=1 vrne na
    serijsko izvajanje (npr. za odpravljanje napak ali če joblib ni
    nameščen)."""
    if up_axis != 2:   # ta funkcija podpira samo Z kot referenčno "gor" os
        raise NotImplementedError("yaw_sweep_registration only supports up_axis=2 (Z) as the "
                                  "reference 'up' direction - the part's nominal resting "
                                  "orientation before tilt correction")

    source_pts = np.asarray(source_down.points)   # source oblak kot numpy tabela
    target_pts = np.asarray(target_down.points)   # target oblak kot numpy tabela
    source_center = source_pts.mean(axis=0)   # centroid source oblaka (za fizični prior spodaj)
    plane_axes = [a for a in range(3) if a != up_axis]   # dve osi pravokotni na up_axis
    if translation_grid_resolution is None:   # če ni podana, jo izpeljemo iz voxel_size
        translation_grid_resolution = voxel_size * 2.0
    if table_center is None:   # privzeto izhodišče, če center mize ni podan
        table_center = np.zeros(2)
    table_half_extent = np.array([table_size_x, table_size_y]) / 2.0   # polovična širina/globina mize v vsaki smeri

    up_vector = np.zeros(3)
    up_vector[up_axis] = 1.0   # enotski vektor "gor" (privzeto Z os)
    target_normal = estimate_surface_normal_ransac(
        target_pts, up_axis, top_band_fraction=tilt_normal_top_band_fraction)   # ocenjena normala površine target (RANSAC na zgornjem pasu)
    if np.dot(target_normal, up_vector) < 0:
        target_normal = -target_normal  # normala ima dvoumen predznak - izberemo tistega, ki kaže "navzgor"
    tilt_deg = np.degrees(np.arccos(np.clip(np.dot(up_vector, target_normal), -1.0, 1.0)))   # ocenjen kot naklona v stopinjah
    tilt_correction = rotation_aligning_vectors(up_vector, target_normal)   # rotacija, ki popravi naklon
    print(f"  Estimated tilt from target's RANSAC surface normal: {tilt_deg:.2f} deg off {up_axis}-axis")

    distance_threshold = voxel_size * distance_threshold_factor   # prag razdalje za ujemanje točk pri hitrem ICP

    yaw_candidates = np.arange(0.0, 360.0, yaw_step_deg)   # vsi testirani yaw koti
    use_parallel = _JOBLIB_AVAILABLE and n_jobs != 1
    print(f"  Yaw sweep: {len(yaw_candidates)} candidates every {yaw_step_deg} deg "
          f"(source={len(source_pts)} pts, target={len(target_pts)} pts), "
          f"distance_threshold={distance_threshold:.3f}, refine_iterations={refine_iterations}, "
          f"n_jobs={'parallel/' + str(n_jobs) if use_parallel else 'serial (joblib not used)'}")

    # Vsak yaw kandidat je popolnoma neodvisen od vseh ostalih (samo BERE
    # source_pts/target_pts/target_normal/tilt_correction, nič ne piše
    # vanje) - _evaluate_yaw_candidate zato lahko teče v ločenih procesih
    # (joblib.Parallel) brez sprememb rezultata, samo hitreje na več-jedrnem
    # stroju. Padec nazaj na serijsko izvajanje (navaden seznam), če joblib
    # ni nameščen ali je n_jobs=1 eksplicitno zahtevan.
    #
    # inner_max_num_threads=1: brez tega bi vsak od n_jobs delovnih procesov
    # ŠE SAM ZASE poskusil uporabiti Open3D-jevo lastno notranjo
    # (OpenMP/TBB) vzporednost za KDTree/ICP - N procesov x lastna notranja
    # nitnost vsak = prekomerna naročenost istih jeder (oversubscription),
    # kar lahko upočasni namesto pospeši. Empirično preverjeno (10 ponovljenih
    # zagonov, 5 z in 5 brez te omejitve, pri fiksnem target_normal/
    # tilt_correction) - rezultat je bil v obeh primerih enak zmagovalni
    # kandidat, torej to NI popravek nedeterminizma (tega nismo zaznali), le
    # preventiva pred oversubscription.
    #
    # OPOMBA o dejanski pohitritvi: prvi (hladen) klic Parallel() v tem
    # procesu plača enkratni strošek zagona bazena delovnih procesov
    # (loky) - pri EDINEM klicu yaw_sweep_registration v enem zagonu
    # main_trial.py (typičen primer) se ta strošek ne povrne vedno v celoti
    # (izmerjeno: hladen vzporeden klic je bil PRIBLIŽNO ENAKO HITER ali
    # rahlo počasnejši od serijskega za 120 kandidatov). Pri PONOVLJENIH
    # klicih znotraj istega procesa (npr. camera_benchmark.py, ki pokliče
    # run_registration stotine-krat) loky bazen ostane "topel" med klici in
    # dejansko izmerjeno pohitritev je bila ~1.7x na sam yaw sweep.
    if use_parallel:
        raw_results = Parallel(n_jobs=n_jobs, inner_max_num_threads=1)(
            delayed(_evaluate_yaw_candidate)(
                yaw, source_pts, target_pts, source_center, target_normal, tilt_correction,
                plane_axes, up_axis, translation_grid_resolution, height_percentile,
                table_center, table_half_extent, distance_threshold, refine_iterations)
            for yaw in yaw_candidates)
    else:
        raw_results = [
            _evaluate_yaw_candidate(
                yaw, source_pts, target_pts, source_center, target_normal, tilt_correction,
                plane_axes, up_axis, translation_grid_resolution, height_percentile,
                table_center, table_half_extent, distance_threshold, refine_iterations)
            for yaw in yaw_candidates]

    rejected_count = sum(1 for r in raw_results if r is None)   # koliko kandidatov je bilo zavrnjenih zaradi fizičnega priorja
    valid_results = [r for r in raw_results if r is not None]
    if valid_results:
        # Isto pravilo kot prejšnja serijska zanka: najprej maksimiziraj
        # fitness, ob izenačenju minimiziraj inlier_rmse (glej prvotni
        # is_better - max() s tem ključem je temu numerično enakovreden).
        best_yaw, best_fitness, best_rmse, best_transform = max(
            valid_results, key=lambda r: (r[1], -r[2]))
    else:
        best_yaw, best_fitness, best_rmse, best_transform = 0.0, -1.0, float("inf"), np.eye(4)

    if rejected_count > 0:
        print(f"  Physical prior (table {table_size_x:.0f}x{table_size_y:.0f}mm centered at "
              f"{(float(table_center[0]), float(table_center[1]))}): rejected {rejected_count}/{len(yaw_candidates)} yaw "
              f"candidates whose implied placement fell off the table")
    if best_fitness < 0.0:
        raise RuntimeError(
            f"Every yaw candidate's implied placement fell outside the table "
            f"({table_size_x:.0f}x{table_size_y:.0f}mm centered at {(float(table_center[0]), float(table_center[1]))}) - "
            f"table_size_x/table_size_y/table_center are too tight for where the part "
            f"actually ended up, or the phase-correlation translation estimate is off")

    print(f"  Best yaw={best_yaw:.1f} deg (after {refine_iterations}-iteration local polish): "
          f"fitness={best_fitness:.4f}, inlier_rmse={best_rmse:.4f}")

    result = o3d.pipelines.registration.RegistrationResult()   # ustvarimo objekt za rezultat
    result.transformation = best_transform   # najboljša najdena transformacija
    result.fitness = best_fitness   # njena fitness vrednost
    result.inlier_rmse = best_rmse if np.isfinite(best_rmse) else 0.0   # njena natančnost (0, če ni bilo nobenega veljavnega kandidata)
    return result   # vrnemo rezultat groba registracije


def refine_registration(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        init_transformation: np.ndarray,
                        voxel_size: float,
                        icp_distance_factor: float = 2.0,
                        icp_voxel_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
                        source_camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),
                        target_camera_location: np.ndarray = np.array([0.0, 0.0, 0.0]),
                        common_eval_threshold_factor: float = 1.5,
                        icp_max_iterations: int = 100,
                        robust_kernel_k_factor: float = 1.0) -> o3d.pipelines.registration.RegistrationResult:
    """Postopen (večnivojski, coarse-to-fine) point-to-plane ICP.

    En sam korak pri polni ločljivosti se muči, kadar imata source (gost -
    celoten, neokludiran izrez CAD) in target (redek - omejen s tem, kar
    kamera dejansko vidi) zelo različno dejansko gostoto točk, tudi pri
    enakem nominalnem voxel_size: KNN normale target-a se povprečijo čez
    veliko širšo fizično okolico kot pri source-u, kar pristransko vpliva
    na normal-projicirani residual, ki ga point-to-plane minimizira, to pa
    lahko povzroči, da en sam fin korak konvergira v slabši lokalni
    optimum kot grobi začetni približek.

    Izvajanje več korakov od grobega proti finemu pri tem pomaga - pri
    grobih merilih downsampling zniža gostoto source-a blizu naravne
    gostote target-a, zato so normale obeh oblakov primerljivo natančne -
    ni pa zagotovljeno, da je to monotono: kasnejši, finejši korak lahko
    kljub temu konča slabše od prejšnjega, če so normale pri tem merilu
    pristranske. Lastna prijavljena fitness vsakega koraka niti ni pošten
    način preverjanja, ker je izračunana pod drugačnim pragom za vsak
    korak, zato se vsak kandidat (grobi začetni približek plus rezultat
    vsakega koraka) namesto tega ovrednoti pod enim fiksnim skupnim pragom.

    Izbira med temi kandidati daje prednost inlier_rmse (kako tesno se
    ujemajoče točke dejansko prilegajo), ne fitness (koliko točk se je
    ujemalo) - finejši korak pogosto konvergira v BOLJ geometrijsko
    natančno pozo, medtem ko pod fiksnim pragom ujame nekoliko MANJ točk
    (tesno konvergirana poza lahko potisne nekaj mejnih točk tik izven
    fiksnega razdaljnega praga, čeprav se prave ujemajoče točke prilegajo
    veliko natančneje). Izbira zgolj po fitness je empirično izbirala
    grobejši, ohlapneje prilegajoč se korak namesto finejšega, ki je bil
    hkrati natančnejši (nižji rmse) IN bližje resnični vrednosti v ločenih
    testih. Fitness še vedno varuje pred zares divergiranim korakom: le
    kandidati znotraj fitness_tolerance najboljše videne fitness so
    upravičeni, zato grobo pokrit korak ne more zmagati zgolj na majhni,
    tesni, a nereprezentativni podmnožici.

    robust_kernel_k_factor skalira Tukeyjevo robustno jedro (TukeyLoss),
    uporabljeno v point-to-plane oceni spodaj namesto navadnih najmanjših
    kvadratov: brez njega osamelci/leteči piksli ZNOTRAJ distance_threshold
    (torej sprejeti kot korespondence) vlečejo rešitev s polno, neomejeno
    težo, enako kot vsaka čista točka. Tukey uteži vsak residual navzdol,
    ko preseže k (tukajšnji privzeti k = stage_voxel_size, isto merilo kot
    distance_threshold za ta korak), in ga popolnoma izniči nad 2k -
    učinkovito izklopi osamelce namesto da bi jih obravnaval enakovredno.
    Poceni sprememba (ena vrstica), ki tipično opazno zniža rmse na šumnih
    (pravih) skenih, ne da bi spremenila obnašanje na čistih podatkih, kjer
    itak ni kaj izklopiti.
    """
    common_threshold = voxel_size * common_eval_threshold_factor   # skupen prag za pošteno primerjavo vseh korakov
    baseline_eval = evaluate_registration(source, target, init_transformation, common_threshold)   # ocena začetnega (grobega) približka
    print(f"  ICP baseline (coarse seed) under common threshold {common_threshold:.3f}: "
          f"fitness={baseline_eval.fitness:.4f}, inlier_rmse={baseline_eval.inlier_rmse:.4f}")
    candidates = [(init_transformation, baseline_eval.fitness, baseline_eval.inlier_rmse)]   # seznam kandidatov, začne z začetnim približkom

    current_transformation = init_transformation   # trenutno najboljša transformacija, izboljšuje se skozi korake
    for scale in sorted(icp_voxel_scales, reverse=True):   # gremo od najbolj grobega merila proti najfinejšemu
        stage_voxel_size = voxel_size * scale   # velikost voxla za ta korak
        distance_threshold = stage_voxel_size * icp_distance_factor   # prag ujemanja točk za ta korak
        source_stage = source.voxel_down_sample(stage_voxel_size)   # source, zmanjšan na to gostoto
        target_stage = target.voxel_down_sample(stage_voxel_size)   # target, zmanjšan na to gostoto
        # use_knn=True iz istega razloga kot pri preračunu s polno
        # ločljivostjo drugje - gostota target-a ni enakomerna niti po
        # zmanjšanju gostote (omejitveni dejavnik je okluzija, ne velikost
        # voxla).
        ensure_oriented_normals(source_stage, normal_radius=stage_voxel_size * 2.0, is_partial_view=True,
                                camera_location=source_camera_location, use_knn=True)   # normale source-a za ta korak
        ensure_oriented_normals(target_stage, normal_radius=stage_voxel_size * 2.0, is_partial_view=True,
                                camera_location=target_camera_location, use_knn=True)   # normale target-a za ta korak
        robust_kernel_k = stage_voxel_size * robust_kernel_k_factor   # skala Tukeyjevega jedra za ta korak, glej docstring
        print(f"  ICP stage voxel_size={stage_voxel_size:.3f}, distance_threshold={distance_threshold:.3f}, "
              f"robust_kernel_k={robust_kernel_k:.3f}, "
              f"source={len(source_stage.points)} pts, target={len(target_stage.points)} pts")
        loss = o3d.pipelines.registration.TukeyLoss(k=robust_kernel_k)   # robustno jedro - navadni osamelci/leteči piksli znotraj praga dobijo zniževano/izničeno težo namesto polne
        result = o3d.pipelines.registration.registration_icp(
            source_stage, target_stage, distance_threshold, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations))   # poženemo point-to-plane ICP za ta korak
        print(f"    fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}")
        current_transformation = result.transformation   # posodobimo trenutno transformacijo za naslednji, finejši korak

        stage_eval = evaluate_registration(source, target, current_transformation, common_threshold)   # ocenimo ta korak pod skupnim pragom
        print(f"    under common threshold {common_threshold:.3f}: fitness={stage_eval.fitness:.4f}, "
              f"inlier_rmse={stage_eval.inlier_rmse:.4f}")
        candidates.append((current_transformation, stage_eval.fitness, stage_eval.inlier_rmse))   # dodamo ta korak med kandidate

    max_fitness = max(fitness for _, fitness, _ in candidates)   # najboljša videna fitness med vsemi kandidati
    fitness_tolerance = 0.9   # kandidat mora doseči vsaj 90% najboljše fitness, da je upravičen
    eligible = [c for c in candidates if c[1] >= max_fitness * fitness_tolerance]   # kandidati, ki niso zares divergirali
    best_transformation, best_fitness, best_rmse = min(eligible, key=lambda c: c[2])   # med njimi izberemo z najnižjim inlier_rmse
    print(f"  Selected transformation with fitness={best_fitness:.4f}, inlier_rmse={best_rmse:.4f} "
          f"(lowest inlier_rmse among candidates within {fitness_tolerance:.0%} of the best "
          f"fitness seen, {max_fitness:.4f})")

    best_eval = o3d.pipelines.registration.RegistrationResult()   # ustvarimo objekt za končni rezultat
    best_eval.transformation = best_transformation   # izbrana najboljša transformacija
    best_eval.fitness = best_fitness   # njena fitness vrednost
    best_eval.inlier_rmse = best_rmse   # njena natančnost
    return best_eval   # vrnemo končni, izboljšan rezultat


def transformation_error(estimated: np.ndarray, reference: np.ndarray) -> dict:
    """Razstavi odstopanje med ocenjeno in referenčno 4x4 rigidno
    transformacijo na napako rotacijskega kota (stopinje) in translacijsko
    napako (enake enote kot oblak točk), namesto surove razlike matrik, ki
    oboje zmeša v težko razumljive številke."""
    r_est, t_est = estimated[:3, :3], estimated[:3, 3]   # rotacijski in translacijski del ocenjene transformacije
    r_ref, t_ref = reference[:3, :3], reference[:3, 3]   # rotacijski in translacijski del referenčne transformacije

    r_delta = r_est @ r_ref.T   # relativna rotacija med ocenjeno in referenčno
    # Kot rotacije iz sledi (trace) rotacijske matrike: trace(R) = 1 + 2*cos(theta).
    cos_theta = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)   # kosinus napake kota, omejen na veljavno območje
    rotation_error_deg = np.degrees(np.arccos(cos_theta))   # napaka rotacije v stopinjah
    translation_error = np.linalg.norm(t_est - t_ref)   # napaka translacije (evklidska razdalja)

    return {
        "rotation_error_deg": rotation_error_deg,   # napaka rotacije v stopinjah
        "translation_error": translation_error,   # napaka translacije v enotah oblaka točk
        "frobenius_norm": np.linalg.norm(estimated - reference),   # skupna razlika matrik za grobo primerjavo
    }


def evaluate_registration(source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          transformation: np.ndarray,
                          threshold: float) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, transformation)   # Open3D-jeva vgrajena ocena ujemanja (fitness, inlier_rmse, ...)


def evaluate_target_coverage(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray,
                             threshold: float) -> dict:
    """PRAVA smer za PASS/FAIL: za vsako TOČKO SKENA (target) izmeri
    razdaljo do najbližje CAD površine (source) pod dano transformacijo -
    obratna smer poizvedbe od Open3D-jevega evaluate_registration/
    registration_icp fitness (icp_result.fitness zgoraj), ki poizveduje iz
    source-a proti target-u in deli s len(target.points).

    Zakaj je source->target fitness slaba PASS/FAIL metrika: target je
    NUJNO delen pogled (kamera zaradi okluzije in omejenega vidnega polja
    vidi le del CAD površine), zato velik del gostega source-a nikoli ne
    dobi bližnjega target ujemanja - to zniža fitness tudi pri POPOLNOMA
    pravilni pozi, kar dela fiksen prag na fitness nezanesljiv (napačno
    zavrne dobre poze na močno okludiranih skenih). target->source (ta
    funkcija) meri nasprotno, fizikalno bolj neposredno vprašanje: ali
    vsaka dejansko izmerjena točka leži na delu? Če je poza pravilna, mora
    skoraj vsaka realna (ne-šumna/ne-ozadje) točka skena ležati blizu CAD
    površine, POPOLNOMA NEODVISNO od tega, kolikšen del te površine je
    sken sploh zajel - ta metrika je torej robustna na delno vidljivost,
    za razliko od source->target fitness.

    Implementirano z isto compute_point_cloud_distance operacijo, ki jo že
    uporablja verify_asymmetric_feature_alignment (transformiraj source v
    target-ov okvir, izmeri razdaljo od target do njega), le na CELOTNEM
    target-u namesto le na asimetrični podmnožici."""
    source_transformed = copy.deepcopy(source)   # ne spreminjamo originala
    source_transformed.transform(transformation)   # source (CAD) v target-ov (svetovni/sken) okvir
    dist = np.asarray(target.compute_point_cloud_distance(source_transformed))   # razdalja vsake target točke do najbližje CAD površine

    return {
        "n_points": len(dist),   # koliko target (sken) točk smo preverili
        "mean": float(dist.mean()),   # povprečna razdalja do CAD površine
        "median": float(np.median(dist)),   # mediana razdalje
        "fraction_within_threshold": float((dist <= threshold).mean()),   # delež target točk, ki dejansko ležijo na delu
    }


def identify_asymmetric_points(pcd: o3d.geometry.PointCloud,
                               up_axis: int,
                               center: np.ndarray,
                               test_angles_deg: tuple[float, ...] = (30, 60, 90, 120, 150, 180,
                                                                     210, 240, 270, 300, 330),
                               top_fraction: float = 0.15) -> np.ndarray:
    """Samodejno najde razlikovalne/asimetrične točke dela (locator pini,
    konektor - značilnosti, na katere se docstring yaw_sweep_registration
    že zanaša za ločevanje pravega yaw od ponavljajoče se geometrijske
    prevare), iz lastne samo-simetrije CAD modela, namesto da bi jih bilo
    treba ročno označiti.

    Za vsakega od peščice testnih zasukov okoli up_axis to izmeri razdaljo
    vsake točke do njenega najbližjega soseda v ZASUKANI kopiji istega
    oblaka. Točka na ponavljajoči se/osnovni geometriji (npr. ena od več
    skoraj identičnih montažnih izboklin) pristane blizu NEKE zasukane
    kopije same sebe - to geometrijsko pomeni "ponavljajoče se". Točka na
    resnično asimetrični značilnosti tega nikoli ne stori, pri nobenem
    testiranem kotu - ostane daleč od vsake zasukane kopije. Obdržanje
    zgornjega dela `top_fraction` po tej vedno-daleč razdalji vrne natanko
    tiste točke, ki jih napačen-a-verjeten yaw ne bi uspel poravnati, kar
    je bistvo njihovega ločenega preverjanja po registraciji.

    center naj bo lastni centroid oblaka točk, v kakršnemkoli okvirju že
    je `pcd` - to meri NARAVNO samo-simetrijo oblike, zato mora biti
    ovrednoteno pred kakršnokoli registracijsko transformacijo, ne na že
    poravnanem rezultatu.
    """
    axis = np.zeros(3)
    axis[up_axis] = 1.0   # os vrtenja za teste simetrije
    points = np.asarray(pcd.points)   # točke kot numpy tabela
    centered = points - center   # točke, premaknjene tako, da je center v izhodišču (za vrtenje okoli center-a)

    min_dist = np.full(len(points), np.inf)   # najmanjša razdalja vsake točke do kakšne zasukane kopije, začetno neskončno
    for angle in test_angles_deg:   # preizkusimo vsak testni kot posebej
        R = rotation_about_axis(axis, angle)   # rotacijska matrika za ta testni kot
        rotated_points = (R @ centered.T).T + center   # zasukana kopija oblaka točk
        rotated_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rotated_points))   # zasukana kopija kot oblak točk
        dist = np.asarray(pcd.compute_point_cloud_distance(rotated_pcd))   # razdalja vsake izvorne točke do najbližje v zasukani kopiji
        min_dist = np.minimum(min_dist, dist)   # obdržimo najmanjšo razdaljo, videno doslej, čez vse teste

    cutoff = np.quantile(min_dist, 1.0 - top_fraction)   # prag - meja za zgornji del po razdalji
    return min_dist >= cutoff   # True za točke, ki so daleč od vseh zasukanih kopij (asimetrične)


_ASYMMETRIC_MASK_CACHE: dict[str, np.ndarray] = {}   # v-pomnilniku predpomnilnik, glej identify_asymmetric_points_cached


def identify_asymmetric_points_cached(pcd: o3d.geometry.PointCloud,
                                      up_axis: int,
                                      center: np.ndarray,
                                      top_fraction: float,
                                      cache_key: Optional[str]) -> np.ndarray:
    """Ovojnica okoli identify_asymmetric_points s preprostim in-memory
    predpomnilnikom (velja za življenjsko dobo tega Python procesa, ne na
    disk). Maska je odvisna SAMO od CAD geometrije (source, PRED
    kakršnokoli registracijsko transformacijo) - ne od poze, šuma ali
    kamere - zato jo je nesmiselno na novo računati (2-5s) pri VSAKEM
    klicu run_registration znotraj istega procesa, ko se isti CAD/crop
    parametri ne spremenijo (npr. camera_benchmark.py pokliče
    run_registration na stotine/tisoče-krat z istim CAD-om in top_fraction,
    le drugo pozo/šum/kamero vsakič).

    cache_key naj bo run_registration-ov lastni source_cache_key (že
    izračunan za preprocess_point_cloud) + asymmetric_check_top_fraction -
    ta dva skupaj enolično določata `source` in s tem tudi masko. Klicatelj
    naj poda cache_key=None, kadar cache NI varen (npr. use_cad_cache=False
    - sveže Poisson-disk vzorčenje ni deterministično med zagoni, glej
    load_cad_model docstring, zato bi predpomnjena maska lahko ustrezala
    DRUGAČNEMU dejanskemu oblaku točk kot trenutni `source`)."""
    if cache_key is not None and cache_key in _ASYMMETRIC_MASK_CACHE:
        return _ASYMMETRIC_MASK_CACHE[cache_key]
    mask = identify_asymmetric_points(pcd, up_axis, center, top_fraction=top_fraction)
    if cache_key is not None:
        _ASYMMETRIC_MASK_CACHE[cache_key] = mask
    return mask


def verify_asymmetric_feature_alignment(source: o3d.geometry.PointCloud,
                                        target: o3d.geometry.PointCloud,
                                        transformation: np.ndarray,
                                        asymmetric_mask: np.ndarray,
                                        residual_threshold: float) -> dict:
    """Varovalka, ki teče PO končni registraciji: ali se lastna
    razlikovalna geometrija dela (identify_asymmetric_points) dejansko
    ujema s target pod končno transformacijo, ne le osnovna oblika na
    splošno?

    Skupna fitness/inlier_rmse lahko izgledata v redu, medtem ko se
    napačen-a-verjeten yaw (prevara s ponavljajočo se izboklino, glej
    docstring yaw_sweep_registration) kljub temu prevleče skozi - osnovna
    geometrija, ki prevladuje v teh agregatnih metrikah, se po konstrukciji
    ujema v obeh primerih. Preverjanje residuala specifično na asimetrični
    podmnožici je veliko bolj ciljana varovalka: te točke pristanejo blizu
    target le, če je orientacija dejansko pravilna.
    """
    source_pts = np.asarray(source.points)[asymmetric_mask]   # obdržimo samo asimetrične/razlikovalne točke source-a
    asym_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))   # te točke kot samostojen oblak
    asym_pcd.transform(transformation)   # uporabimo končno transformacijo nanje
    dist = np.asarray(asym_pcd.compute_point_cloud_distance(target))   # razdalja vsake do najbližje točke v target

    return {
        "n_points": len(dist),   # koliko asimetričnih točk smo preverili
        "mean": float(dist.mean()),   # povprečna razdalja
        "median": float(np.median(dist)),   # mediana razdalje
        "max": float(dist.max()),   # največja (najslabša) razdalja
        "fraction_within_threshold": float((dist <= residual_threshold).mean()),   # delež točk znotraj sprejemljivega praga
    }


def decide_registration_outcome(icp_result: o3d.pipelines.registration.RegistrationResult,
                                asym_stats: dict,
                                target_coverage: dict,
                                skipped_icp: bool,
                                min_fitness: float,
                                max_inlier_rmse: float,
                                min_asymmetric_fraction: float,
                                min_target_coverage: float) -> dict:
    """Eksplicitna PASS/FAIL odločitev o končni registraciji, vrnjena kot
    del rezultata (ne le izpisana na stdout).

    Prejšnje "varovalke" v tej datoteki so bile zgolj asimetrične: nizka
    coarse fitness je preskočila ICP, a je funkcija kljub temu VRNILA
    (verjetno slabo) transformacijo; nizek delež poravnanih asimetričnih
    značilnosti je zgolj izpisal WARNING. Robot, ki to skripto kliče kot
    korak v svojem ciklu prijemanja, ne bere stdout-a - potrebuje en sam,
    nedvoumen booleov signal (in razloge, če je False), na katerega se
    lahko dejansko odloči, ali sme prijeti del. Vsak prag spodaj lahko SAM
    prevesi odločitev na FAIL, ne glede na to, kako dobri so preostali -
    to ni "povprečna ocena", je AND čez vse varnostne pogoje.

    Glavna varovalka za dejansko prekrivanje sken<->CAD je min_target_coverage
    (glej evaluate_target_coverage) - target->source, robustna na okluzijo.
    min_fitness na icp_result.fitness (source->target, Open3D-jeva
    definicija - glej evaluate_registration) je namerno samo NIZKA
    zdrava-pamet meja (privzeto majhna), ne primarna odločitvena meja:
    ta fitness je po konstrukciji nizji na močno okludiranih skenih tudi
    pri popolnoma pravilni pozi, zato bi visok fiksen prag nanjo (prejšnji
    privzetek 0.6) zavračal dobre poze - natanko napaka, ki jo je
    prakticen zagon te datoteke pokazal.

    OPOZORILO: vsi privzeti pragovi (min_fitness/max_inlier_rmse/
    min_asymmetric_fraction/min_target_coverage) so razumni PLACEHOLDER-ji,
    NE empirično umerjene vrednosti. Pravilna umeritev: Monte Carlo čez
    znane dobre poze (ta datoteka že podpira --translation_x/y/z/
    --rotation_x/y/z_deg in naključen šum za tak namen) da porazdelitev
    vsake metrike pri PRAVILNI registraciji - prag postavi pod spodnji rep
    te porazdelitve, in preveri, da namerno napačni yaw kandidati (npr.
    --yaw_step_deg z zamaknjenim začetkom) padejo jasno pod njega, preden
    kateremukoli od teh privzetkov zaupaš v produkciji.
    """
    reasons = []
    if skipped_icp:
        reasons.append(
            "coarse (yaw sweep) fitness was below --min_coarse_fitness_for_icp, "
            "so ICP was skipped entirely - the returned pose is only the coarse "
            "seed and is not reliable enough to act on")
    if icp_result.fitness < min_fitness:
        reasons.append(f"fitness {icp_result.fitness:.4f} is below the minimum "
                       f"required {min_fitness:.4f} (sanity floor only - see "
                       f"target_coverage below for the real occlusion-robust overlap check)")
    if icp_result.inlier_rmse > max_inlier_rmse:
        reasons.append(f"inlier_rmse {icp_result.inlier_rmse:.4f} is above the maximum "
                       f"allowed {max_inlier_rmse:.4f}")
    if asym_stats["fraction_within_threshold"] < min_asymmetric_fraction:
        reasons.append(
            f"only {asym_stats['fraction_within_threshold']:.2%} of the part's distinctive/"
            f"asymmetric features aligned within threshold (minimum required "
            f"{min_asymmetric_fraction:.2%}) - the result may have converged to a symmetric "
            f"look-alike orientation (e.g. a repeated-boss match) rather than the true pose")
    if target_coverage["fraction_within_threshold"] < min_target_coverage:
        reasons.append(
            f"only {target_coverage['fraction_within_threshold']:.2%} of the scanned (target) "
            f"points lie on the CAD surface under this pose (minimum required "
            f"{min_target_coverage:.2%}) - a correct pose should place nearly every genuine "
            f"scanned point on the part regardless of occlusion, so a low value here means the "
            f"scan doesn't actually sit on the part as posed")

    return {
        "accepted": len(reasons) == 0,   # True <=> robot je varen za prijem dela na tej pozi
        "reasons": reasons,   # prazen seznam, če je accepted True; sicer vsak razlog za FAIL
        "fitness": icp_result.fitness,
        "inlier_rmse": icp_result.inlier_rmse,
        "asymmetric_fraction": asym_stats["fraction_within_threshold"],
        "target_coverage_fraction": target_coverage["fraction_within_threshold"],
    }


def run_registration(cad_path: Path,
                     scan_path: Optional[Path] = None,
                     extra_scan_paths: Optional[list[Path]] = None,
                     use_real_scan: bool = False,
                     voxel_size: float = 1.0,
                     sample_point_count: int = 200000,
                     filter_real_scan: bool = True,
                     filter_grazing_incidence: bool = True,
                     flying_pixel_filter_max_incidence_deg: float = 80.0,
                     grazing_incidence_normal_radius_factor: float = 2.0,
                     disable_occlusion: bool = False,
                     depth_noise_at_1m: float = 0.1,
                     noise_distance_power: float = 2.0,
                     noise_reference_distance_m: float = 1.0,
                     max_incidence_deg: float = 75.0,
                     distance_bias_permille: float = 0.0,
                     global_planarity_mm: float = 0.0,
                     global_planarity_spatial_scale_px: Optional[float] = None,
                     outlier_probability: float = 0.01,
                     outlier_std_multiplier: float = 15.0,
                     flying_pixel_depth_jump_mm: float = 3.0,
                     flying_pixel_probability: float = 0.3,
                     quantization_step_at_1m: float = 0.05,
                     noise_spatial_correlation_px: float = 1.5,
                     top_fraction: float = 0.35,
                     up_axis: int = 2,
                     flip_up_direction: bool = False,
                     coarse_distance_factor: float = 3.0,
                     yaw_step_deg: float = 6.0,
                     yaw_sweep_refine_iterations: int = 5,
                     icp_distance_factor: float = 2.0,
                     icp_voxel_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
                     icp_max_iterations: int = 100,
                     robust_kernel_k_factor: float = 1.0,
                     min_coarse_fitness_for_icp: float = 0.1,
                     use_cad_cache: bool = True,
                     use_preprocess_cache: bool = True,
                     cache_key_tag: str = "",
                     camera_fov_deg: float = 40.0,
                     camera_width_px: int = 640,
                     camera_height_px: int = 480,
                     table_size_x: float = 500.0,
                     table_size_y: float = 500.0,
                     table_center: Optional[np.ndarray] = None,
                     add_table_background: bool = False,
                     remove_table_background_flag: Optional[bool] = None,
                     table_removal_margin_mm: float = 10.0,
                     tilt_normal_top_band_fraction: float = 0.5,
                     height_percentile: float = 99.5,
                     yaw_sweep_n_jobs: int = -1,
                     asymmetric_check_top_fraction: float = 0.15,
                     asymmetric_check_residual_factor: float = 3.0,
                     target_coverage_distance_factor: float = 2.0,
                     min_accept_fitness: float = 0.1,
                     max_accept_inlier_rmse: Optional[float] = None,
                     min_accept_asymmetric_fraction: float = 0.5,
                     min_accept_target_coverage: float = 0.85) -> tuple[
                         o3d.geometry.PointCloud,
                         o3d.geometry.PointCloud,
                         o3d.pipelines.registration.RegistrationResult,
                         o3d.pipelines.registration.RegistrationResult,
                         dict]:
    if remove_table_background_flag is None:
        # Privzeto vklopljeno za PRAVE skene, izklopljeno za simulacijo:
        # simulirani sken nima mize v oblaku, razen če je add_table_background
        # eksplicitno True (v tem primeru remove_table_background_flag
        # ostane uporabnikova eksplicitna izbira, ne ta samodejni privzetek).
        # Pravi sken pa domala vedno zajame vsaj nekaj mize/ozadja okoli
        # dela, remove_table_background_flag=False (prejšnji trdi privzetek)
        # pa je to ozadje puščal v target - PCA/RANSAC ocena naklona bi
        # lahko ocenila normalo MIZE namesto zgornje ploskve dela, naklon
        # dela pa bi se v celoti izgubil.
        remove_table_background_flag = use_real_scan

    print("Phase 1: Loading CAD model...")
    source = load_cad_model(cad_path, sample_point_count, use_cache=use_cad_cache)   # naložimo CAD kot oblak točk
    full_part_height = float(source.get_max_bound()[up_axis] - source.get_min_bound()[up_axis])   # znana polna višina CELEGA dela, PRED obrezovanjem - potrebna za remove_table_background
    source = crop_top_region(source, top_fraction=top_fraction, up_axis=up_axis,
                             flip_up_direction=flip_up_direction)   # obrežemo na zgornje "sealing" območje
    cropped_region_height = float(source.get_max_bound()[up_axis] - source.get_min_bound()[up_axis])   # dejanska (izmerjena, ne le top_fraction*full_part_height) višina te regije - za remove_table_background, da zoži pas tudi od spodaj
    print(f"  Cropped CAD model to top {top_fraction:.0%} along axis {up_axis} "
          f"(flip_up_direction={flip_up_direction}) "
          f"(the seal region the camera actually scans): {len(source.points)} points remain")
    # Obrezovanje spremeni source iz zaprte površine v enostransko zgornjo
    # ploščo, enako kot target - orient_normals_consistent_tangent_plane-ova
    # predpostavka zaprte površine ne velja več, zato source namesto tega
    # potrebuje virtualno "gledanje navzdol od zgoraj" lokacijo kamere,
    # enako kot target.
    source_camera_location = top_camera_location(source, up_axis=up_axis,
                                                  flip_up_direction=flip_up_direction)   # virtualna kamera za orientacijo normal source-a

    print("Phase 2: Loading target scan...")
    if use_real_scan and scan_path is not None and scan_path.exists():
        all_scan_paths = [scan_path] + list(extra_scan_paths or [])   # primarni sken + morebitni dodatni zaporedni zajemi za temporalno povprečenje
        if len(all_scan_paths) > 1:
            target = load_and_merge_real_scans(all_scan_paths)   # združimo več zajemov istega statičnega dela
        else:
            target = load_real_scan(scan_path)   # naložimo pravi sken iz datoteke
        if filter_real_scan:
            target = filter_scan(target)   # odstranimo statistične osamelce
    else:
        cad_mesh = load_cad_mesh(cad_path)   # naložimo CAD kot mrežo (za ray casting)
        target = simulate_camera_scan(cad_mesh, disable_occlusion=disable_occlusion,
                                      up_axis=up_axis, flip_up_direction=flip_up_direction,
                                      top_fraction=top_fraction,
                                      camera_location=np.zeros(3),
                                      fov_deg=camera_fov_deg, width_px=camera_width_px,
                                      height_px=camera_height_px,
                                      depth_noise_at_1m=depth_noise_at_1m,
                                      noise_distance_power=noise_distance_power,
                                      noise_reference_distance_m=noise_reference_distance_m,
                                      max_incidence_deg=max_incidence_deg,
                                      distance_bias_permille=distance_bias_permille,
                                      global_planarity_mm=global_planarity_mm,
                                      global_planarity_spatial_scale_px=global_planarity_spatial_scale_px,
                                      outlier_probability=outlier_probability,
                                      outlier_std_multiplier=outlier_std_multiplier,
                                      flying_pixel_depth_jump_mm=flying_pixel_depth_jump_mm,
                                      flying_pixel_probability=flying_pixel_probability,
                                      quantization_step_at_1m=quantization_step_at_1m,
                                      noise_spatial_correlation_px=noise_spatial_correlation_px,
                                      add_table_background=add_table_background,
                                      table_size_x=table_size_x, table_size_y=table_size_y,
                                      table_center=table_center)   # simuliramo skeniran oblak točk
        print(f"  Simulated target has {len(target.points)} points "
              f"(source has {len(source.points)}) - overlap ratio "
              f"{len(target.points) / len(source.points):.2f}")

    if remove_table_background_flag:
        print("  Removing table/background points from target...")
        target = remove_table_background(target, up_axis=up_axis, flip_up_direction=flip_up_direction,
                                         part_height_mm=full_part_height,
                                         cropped_region_height_mm=cropped_region_height,
                                         margin_mm=table_removal_margin_mm)   # odstranimo domnevne mizne/ozadje točke IN zavese pod skenirano regijo

    # source (CAD) in target (sken) prikazana v svojih izvornih/svetovnih
    # koordinatah, brez uporabljene transformacije - njuna relativna
    # pozicija pred kakršnokoli registracijo, da je od začetka vidno, kaj
    # morata grobi (coarse) in ICP korak sploh rešiti.
    draw_registration_result(source, target, np.eye(4), "Pred poravnavo (CAD vs. scan)")

    # Kamera je VEDNO fiksirana v izhodišču svetovnega koordinatnega sistema
    # (0,0,0) - za realne IN simulirane skene enako, glej
    # build_reference_transform docstring. Prava kamera je fiksno montirana
    # in NI nujno neposredno nad delom (del lahko pristane odmaknjen vstran
    # na mizi, kamera pa se ne premakne z njim); simulirana kamera je zdaj
    # po definiciji na isti fiksni poziciji (glej simulate_camera_scan
    # zgoraj) - translation_x/y/z (SIMULATED_TRANSFORM) je tisto, kar se
    # premika GLEDE NA kamero, ne obratno. Prejšnja koda je za oba primera
    # dinamično izračunala "kamero nad target-om" (top_camera_location) -
    # to je tiho pomenilo drugačno kamero za vsako testno pozo, kar realna,
    # fiksno montirana kamera nikoli ne počne.
    target_camera_location = np.zeros(3)

    if filter_grazing_incidence:
        print("  Filtering grazing-incidence (flying pixel) points from target...")
        target = filter_grazing_incidence_points(
            target, target_camera_location,
            max_incidence_deg=flying_pixel_filter_max_incidence_deg,
            normal_radius=voxel_size * grazing_incidence_normal_radius_factor)   # odstranimo domnevne leteče piksle na robovih pred predobdelavo

    print("Phase 3: Preprocessing point clouds...")
    # Cache ključi identificirajo *vhod* (vse zgoraj, kar določa source/
    # target-ove točke, preden se predobdelava sploh začne);
    # preprocess_point_cloud sam vključi vsak parameter, ki dodatno vpliva
    # na njegov izhod.
    source_cache_key = (f"source_{cad_path.stem}_{sample_point_count}pts_top{top_fraction}_"
                       f"axis{up_axis}_flip{flip_up_direction}")   # cache ključ za source
    # remove_table_background_flag in filter_grazing_incidence oba
    # spremenita target (odstranita točke) PRED tem, ko
    # preprocess_point_cloud sploh vidi svoj cache_key, zato morata biti
    # del obeh spodnjih ključev - sicer bi vklop/izklop katerega od njiju
    # tiho servisiral predpomnjen target iz nasprotnega stanja.
    removal_tag = f"_notable{table_removal_margin_mm}" if remove_table_background_flag else ""
    grazing_tag = (f"_nograze{flying_pixel_filter_max_incidence_deg}_"
                  f"{grazing_incidence_normal_radius_factor}" if filter_grazing_incidence else "")
    if use_real_scan and scan_path is not None and scan_path.exists():
        # scan_path.stem sam po sebi NI dovolj: robot v produkciji vsak
        # cikel zapiše nov sken v isto pot (npr. "scan.ply"), zato bi ključ,
        # ki pozna samo ime datoteke, za vedno servisiral predpomnjeno
        # predobdelavo PRVEGA kdajkoli videnega skena na tej poti - tiha in
        # katastrofalna napaka, ker se program ne bi nikoli pritožil, samo
        # tiho vračal napačno pozo. mtime (nanosekunde, da se ne spremeni
        # samo znotraj iste sekunde) + velikost datoteke skupaj zaznata
        # vsako novo zapisano vsebino brez branja/hashiranja celotnega
        # (lahko velikega) oblaka točk ob vsakem zagonu. Če je scan_path
        # del več-skenskega merge-a (extra_scan_paths), mora ključ pokriti
        # mtime+velikost VSAKEGA prispevajočega skena, ne le prvega -
        # sprememba katerekoli od dodatnih datotek mora prav tako
        # neveljaviti predpomnilnik.
        scan_stats_tag = "_".join(
            f"{p.stem}_m{p.stat().st_mtime_ns}_s{p.stat().st_size}" for p in all_scan_paths)
        target_cache_key = (f"target_real_{scan_stats_tag}_"
                            f"filter{filter_real_scan}{removal_tag}{grazing_tag}")   # cache ključ za pravi (morda združen) sken
    else:
        # cache_key_tag vključi vse, kar spremeni sam SIMULATED_TRANSFORM
        # (translation_x/y/z/rotation_x/y/z_deg, vbrizgan v main()), a se
        # sicer ne odraža v teh parametrih - brez tega bi se dve različni
        # testni pozi zaleteli v isti zapis predpomnilnika in tiho
        # postregli predpomnjen target ene poze drugi.
        noise_params = (depth_noise_at_1m, noise_distance_power, noise_reference_distance_m,
                       max_incidence_deg, distance_bias_permille, global_planarity_mm,
                       global_planarity_spatial_scale_px,
                       outlier_probability, outlier_std_multiplier, flying_pixel_depth_jump_mm,
                       flying_pixel_probability, quantization_step_at_1m, noise_spatial_correlation_px)
        noise_digest = hashlib.md5(repr(noise_params).encode()).hexdigest()[:10]   # kratek hash vseh šumovnih parametrov
        table_tag = (f"_table{table_size_x}x{table_size_y}at{tuple(table_center) if table_center is not None else (0, 0)}"
                    if add_table_background else "")   # miza vpliva na target samo, če je add_table_background=True
        target_cache_key = (f"target_sim_{cad_path.stem}_top{top_fraction}_"
                            f"axis{up_axis}_flip{flip_up_direction}_noise{noise_digest}_occ{disable_occlusion}"
                            f"_fov{camera_fov_deg}_{camera_width_px}x{camera_height_px}{table_tag}{removal_tag}"
                            f"{grazing_tag}{cache_key_tag}")   # cache ključ za simuliran target

    # source = na vrh obrezan CAD model (enostranska ploskev, enako kot
    # target - ni več zaprta površina, glej source_camera_location zgoraj)
    # target = enostranski pogled kamere (pravi ali simuliran)
    source_down = preprocess_point_cloud(
        source, voxel_size, is_partial_view=True, camera_location=source_camera_location,
        cache_key=source_cache_key, use_cache=use_preprocess_cache)   # zmanjšan in normaliziran source
    target_down = preprocess_point_cloud(
        target, voxel_size, is_partial_view=True, camera_location=target_camera_location,
        cache_key=target_cache_key, use_cache=use_preprocess_cache)   # zmanjšan in normaliziran target

    print("Phase 4: Global registration (yaw sweep)...")
    # Izkorišča pravo omejitev postavitve (stik z mizo fiksira Z/roll/
    # pitch, prost je samo yaw, dejanski naklon stran od tega pa se prav
    # tako oceni in popravi) namesto reševanja slepe 6-DOF registracije -
    # glej docstring yaw_sweep_registration().
    coarse_result = yaw_sweep_registration(
        source_down, target_down, voxel_size,
        up_axis=up_axis,
        yaw_step_deg=yaw_step_deg,
        distance_threshold_factor=coarse_distance_factor,
        refine_iterations=yaw_sweep_refine_iterations,
        table_size_x=table_size_x,
        table_size_y=table_size_y,
        table_center=table_center,
        tilt_normal_top_band_fraction=tilt_normal_top_band_fraction,
        height_percentile=height_percentile,
        n_jobs=yaw_sweep_n_jobs)   # groba (yaw sweep) registracija
    print(f"  Yaw sweep fitness: {coarse_result.fitness:.4f}, "
          f"inlier_rmse: {coarse_result.inlier_rmse:.4f}")
    if not use_real_scan:   # pri simulaciji poznamo referenčno transformacijo, lahko preverimo napako
        err = transformation_error(coarse_result.transformation, SIMULATED_TRANSFORM)
        print(f"  Yaw sweep vs reference transform: "
              f"rotation_error={err['rotation_error_deg']:.2f} deg, "
              f"translation_error={err['translation_error']:.3f}, "
              f"frobenius_norm={err['frobenius_norm']:.3f}")
    draw_registration_result(source, target, coarse_result.transformation, "Yaw Sweep Result")   # vizualiziramo grob rezultat

    # Normale za ICP se izračunajo za vsak korak posebej znotraj
    # večnivojske zanke refine_registration (vsak korak znova zmanjša
    # gostoto in potrebuje normale, ki ustrezajo gostoti TEGA koraka), zato
    # tu ločen preračun s polno ločljivostjo ni potreben.

    skipped_icp = coarse_result.fitness < min_coarse_fitness_for_icp   # ali smo ICP preskočili zaradi prešibkega grobega rezultata (potrebno za PASS/FAIL odločitev spodaj)
    if skipped_icp:   # grob rezultat je preslab, da bi ga ICP lahko izboljšal
        # Point-to-plane ICP linearizira okoli začetnega približka in
        # predpostavi, da je ta že blizu rešitve - podati mu tako slab
        # grob začetni približek (običajno deset+ stopinj napake rotacije)
        # ne le, da ne konvergira, ampak lahko divergira v nesmisel
        # (transformacije z metrsko/kilometrsko translacijo). Preskočimo
        # ga, namesto da bi ustvarili zavajajočo "končno" matriko; pravi
        # problem, ki ga je treba rešiti, je fitness grobega koraka, ne ICP.
        print(f"  SKIPPING ICP: coarse fitness {coarse_result.fitness:.4f} is below "
              f"--min_coarse_fitness_for_icp={min_coarse_fitness_for_icp}. ICP can only "
              f"refine a seed that's already roughly correct - feeding it this one is "
              f"likely to diverge, not improve it.")
        icp_result = coarse_result   # obdržimo grob rezultat kot končnega
    else:
        # refine_registration sledi najbolje ocenjenemu kandidatu med
        # grobim začetnim približkom in vsakim ICP korakom pod enim
        # skupnim pragom (glej njen docstring), zato je icp_result tu
        # zagotovljeno vsaj tako dober kot coarse_result - ločeno naknadno
        # preverjanje zavrnitve ni potrebno.
        icp_result = refine_registration(source, target, coarse_result.transformation, voxel_size,
                                         icp_distance_factor=icp_distance_factor,
                                         icp_voxel_scales=icp_voxel_scales,
                                         source_camera_location=source_camera_location,
                                         target_camera_location=target_camera_location,
                                         icp_max_iterations=icp_max_iterations,
                                         robust_kernel_k_factor=robust_kernel_k_factor)   # natančna, večnivojska ICP refinacija
        print(f"  ICP fitness: {icp_result.fitness:.4f}, "
              f"inlier_rmse: {icp_result.inlier_rmse:.4f}")

    draw_registration_result(source, target, icp_result.transformation, "ICP Result")   # vizualiziramo končni rezultat
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

    evaluation_before = evaluate_registration(source, target, coarse_result.transformation, voxel_size * 1.5)   # ocena pred ICP
    evaluation_after = evaluate_registration(source, target, icp_result.transformation, voxel_size * 1.5)   # ocena po ICP
    print("Evaluation before ICP:", evaluation_before)
    print("Evaluation after ICP:", evaluation_after)

    print("Phase 5: Asymmetric-feature safety check...")
    # Skupna fitness/inlier_rmse lahko izgledata v redu, medtem ko je
    # rezultat dejansko konvergiral v ponavljajoče-se-geometrijsko podoben
    # yaw (glej docstring yaw_sweep_registration) - osnovna geometrija, ki
    # prevladuje v teh agregatnih metrikah, se po konstrukciji ujema v
    # obeh primerih. identify_asymmetric_points najde source-ove lastne
    # razlikovalne točke (locator pini, konektor) iz njegove samo-simetrije;
    # verify_asymmetric_feature_alignment nato preveri SPECIFIČNO residual
    # teh točk pod končno transformacijo, veliko bolj ciljana varovalka kot
    # agregatne metrike zgoraj.
    asymmetric_cache_key = (f"{source_cache_key}_asymtop{asymmetric_check_top_fraction}"
                           if use_cad_cache else None)   # glej identify_asymmetric_points_cached - None onemogoči cache, če CAD vzorčenje ni deterministično
    asymmetric_mask = identify_asymmetric_points_cached(
        source, up_axis, source.get_center(), top_fraction=asymmetric_check_top_fraction,
        cache_key=asymmetric_cache_key)   # katere source točke so razlikovalne/asimetrične
    residual_threshold = voxel_size * asymmetric_check_residual_factor   # prag "dobrega ujemanja" za te točke
    asym_stats = verify_asymmetric_feature_alignment(
        source, target, icp_result.transformation, asymmetric_mask, residual_threshold)   # dejanski residual po ICP
    print(f"  {asym_stats['n_points']} asymmetric/distinctive points identified "
          f"(top {asymmetric_check_top_fraction:.0%} by self-rotation distance): "
          f"mean_residual={asym_stats['mean']:.3f}, median_residual={asym_stats['median']:.3f}, "
          f"max_residual={asym_stats['max']:.3f}, "
          f"fraction_within_{residual_threshold:.2f}mm={asym_stats['fraction_within_threshold']:.2%}")

    print("Phase 6: Target->source coverage check...")
    # Obratna smer od icp_result.fitness (source->target, glej
    # evaluate_target_coverage docstring za razlago, zakaj je ta smer
    # robustna na okluzijo, source->target fitness pa ne) - to je GLAVNA
    # metrika, na kateri temelji spodnja PASS/FAIL odločitev.
    coverage_threshold = voxel_size * target_coverage_distance_factor   # prag "leži na delu" za target točke
    target_coverage = evaluate_target_coverage(source, target, icp_result.transformation, coverage_threshold)
    print(f"  {target_coverage['n_points']} scanned (target) points checked against CAD surface: "
          f"mean_dist={target_coverage['mean']:.3f}, median_dist={target_coverage['median']:.3f}, "
          f"fraction_within_{coverage_threshold:.2f}mm={target_coverage['fraction_within_threshold']:.2%}")

    print("Phase 7: PASS/FAIL accept decision...")
    # Nadomesti prejšnji asimetrični par varovalk (WARNING izpis tukaj +
    # tiho vrnjena, morda slaba transformacija na prešibki coarse fitness
    # zgoraj) z ENIM eksplicitnim, strojno berljivim rezultatom - glej
    # decide_registration_outcome docstring za razlog, zakaj mora biti to
    # del vrnjenega rezultata, ne le nekaj, izpisano na stdout, IN za
    # razlog, zakaj je target_coverage (ne icp_result.fitness) glavna
    # varovalka za prekrivanje.
    if max_accept_inlier_rmse is None:
        max_accept_inlier_rmse = voxel_size * 1.5   # privzeto: isti prag, pod katerim je bil icp_result sploh ocenjen (glej common_threshold v refine_registration)
    decision = decide_registration_outcome(
        icp_result, asym_stats, target_coverage, skipped_icp,
        min_fitness=min_accept_fitness,
        max_inlier_rmse=max_accept_inlier_rmse,
        min_asymmetric_fraction=min_accept_asymmetric_fraction,
        min_target_coverage=min_accept_target_coverage)
    if decision["accepted"]:
        print(f"  ACCEPT: fitness={decision['fitness']:.4f}, inlier_rmse={decision['inlier_rmse']:.4f}, "
              f"asymmetric_fraction={decision['asymmetric_fraction']:.2%}, "
              f"target_coverage={decision['target_coverage_fraction']:.2%} - all thresholds met, "
              f"robot may act on this pose")
    else:
        print(f"  REJECT: robot must NOT act on this pose. Reasons:")
        for reason in decision["reasons"]:
            print(f"    - {reason}")

    return source, target, coarse_result, icp_result, decision   # source, target, oba rezultata registracije (grob + ICP) in eksplicitna PASS/FAIL odločitev


# ---------------------------------------------------------------------------
# Samostojni (hitri, brez CAD/registracije) testi novih parametrov za
# kalibracijo šumovnega modela na resnične kamere (glej kamere.py):
# noise_reference_distance_m, distance_bias_permille, global_planarity_mm in
# build_reference_transform (--translation_x/y/z/--rotation_x/y/z_deg).
# Testirajo SAMO, da je vsak nov parameter matematično pravilno implementiran
# (npr. da noise_reference_distance_m res premakne referenčno točko šumovne
# formule, ne da je le neuporabljen argument) - ne testirajo celotnega
# registracijskega cevovoda, za kar služijo --translation_x/y/z/
# --rotation_x/y/z_deg empirični zagoni. Pognati jih je mogoče z
# `main_trial.py --self_test`.
# ---------------------------------------------------------------------------

def _test_noise_reference_distance() -> None:
    """Pri distance_m == noise_reference_distance_m in incidence_cos=1.0
    (naravnost gledanje, brez 1/cos faktorja) mora heteroscedastic_noise_std_mm
    vrniti natanko depth_noise_at_1m - to velja PO DEFINICIJI referenčne
    razdalje (glej docstring), ne glede na to, kateri noise_reference_distance_m
    izberemo. Preverja tudi, da privzeta vrednost (1.0) ohrani staro,
    nazaj-kompatibilno obnašanje."""
    for reference_m in (1.0, 0.442, 0.6, 0.05):
        std = heteroscedastic_noise_std_mm(
            distance_m=np.array([reference_m]), incidence_cos=np.array([1.0]),
            depth_noise_at_1m=0.05, noise_distance_power=2.0, max_incidence_deg=75.0,
            noise_reference_distance_m=reference_m)
        assert abs(std[0] - 0.05) < 1e-9, (
            f"Pri distance_m==noise_reference_distance_m ({reference_m}) mora "
            f"std==depth_noise_at_1m==0.05, dobil {std[0]}")
    # Pri DVAKRAT večji razdalji od reference in noise_distance_power=2.0
    # mora biti std natanko 4x večji (2^2) - preverja, da je razmerje
    # distance_m/noise_reference_distance_m (ne golo distance_m) tisto, kar
    # gre v np.power.
    std_at_2x = heteroscedastic_noise_std_mm(
        distance_m=np.array([0.884]), incidence_cos=np.array([1.0]),
        depth_noise_at_1m=0.05, noise_distance_power=2.0, max_incidence_deg=75.0,
        noise_reference_distance_m=0.442)
    assert abs(std_at_2x[0] - 0.05 * 4.0) < 1e-9, (
        f"Pri 2x referenčni razdalji in power=2.0 mora biti std 4x večji od "
        f"depth_noise_at_1m (0.20), dobil {std_at_2x[0]}")
    print("  OK: heteroscedastic_noise_std_mm pravilno uporabi noise_reference_distance_m")


def _test_distance_bias_permille() -> None:
    """apply_distance_bias_mm mora premakniti razdaljo za natanko
    bias_permille/1000 delež (v obe smeri - pozitiven/negativen bias) in
    ne spremeniti ničesar pri bias_permille=0 (privzeto, nazaj-kompatibilno)."""
    distance_mm = np.array([1000.0, 500.0, 250.0])
    unbiased = apply_distance_bias_mm(distance_mm, 0.0)
    assert np.allclose(unbiased, distance_mm), "bias_permille=0 ne sme spremeniti razdalje"
    biased = apply_distance_bias_mm(distance_mm, 1.25)
    expected = distance_mm * 1.00125
    assert np.allclose(biased, expected), f"Pričakovano {expected}, dobil {biased}"
    biased_neg = apply_distance_bias_mm(distance_mm, -2.5)
    expected_neg = distance_mm * 0.9975
    assert np.allclose(biased_neg, expected_neg), f"Pričakovano {expected_neg}, dobil {biased_neg}"
    print("  OK: apply_distance_bias_mm pravilno skalira razdaljo v obe smeri")


def _test_global_planarity_bias() -> None:
    """generate_global_planarity_bias_mm mora vrniti polje z razponom
    (max-min) natanko amplitude_mm, ničelno polje pri amplitude_mm=0
    (privzeto, nazaj-kompatibilno), in dovolj GLADKO (nizkofrekvenčno -
    sosednji piksli podobni, ne iid šum kot preostali šumovni členi)."""
    shape = (64, 64)
    zero_field = generate_global_planarity_bias_mm(shape, amplitude_mm=0.0, low_freq_sigma_px=10.0)
    assert np.allclose(zero_field, 0.0), "amplitude_mm=0 mora vrniti ničelno polje"

    field = generate_global_planarity_bias_mm(shape, amplitude_mm=0.25, low_freq_sigma_px=10.0)
    actual_range = field.max() - field.min()
    assert abs(actual_range - 0.25) < 1e-9, f"Razpon mora biti natanko 0.25mm, dobil {actual_range}"

    # Gladkost: povprečna razlika med sosednjima pikseloma mora biti veliko
    # manjša od skupnega razpona polja - iid šum (kot raw_noise pred
    # glajenjem) te lastnosti nima, gladko nizkofrekvenčno polje pa jo ima
    # po konstrukciji.
    neighbor_diff = np.abs(np.diff(field, axis=0)).mean()
    assert neighbor_diff < actual_range * 0.1, (
        f"Polje ni dovolj gladko/nizkofrekvenčno (povprečna sosednja razlika "
        f"{neighbor_diff:.4f} ni veliko manjša od razpona {actual_range:.4f})")
    print("  OK: generate_global_planarity_bias_mm vrne pravilno skalirano, gladko polje")


def _test_top_camera_location_offset_distance() -> None:
    """top_camera_location z eksplicitnim offset_distance mora postaviti
    kamero natanko na to razdaljo od roba oblaka (ne od privzetega,
    velikosti-oblaka-odvisnega offset_factor) - ta funkcija se v novi
    (kamera-vedno-v-izhodišču) zasnovi uporablja samo še za
    source_camera_location (virtualna "pogled od zgoraj" lokacija za
    orientacijo CAD-ovih normal, glej run_registration), ne več za pravo/
    simulirano kamero (ta je zdaj vedno np.zeros(3)), a mora ostati
    pravilno parametrizirana."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 5.0]])))
    for offset in (442.0, 1000.0, 250.0):
        loc = top_camera_location(pcd, up_axis=2, offset_distance=offset)
        expected_z = pcd.get_max_bound()[2] + offset
        assert abs(loc[2] - expected_z) < 1e-9, (
            f"offset_distance={offset}: pričakovano Z={expected_z}, dobil {loc[2]}")
    print("  OK: top_camera_location pravilno uporabi eksplicitno podan offset_distance")


def _test_build_reference_transform() -> None:
    """build_reference_transform mora pri privzetih (ničelnih) rotacijah
    vrniti identično rotacijo z natanko podano translacijo, in pravilno
    zavrteti znane enotske vektorje pri 90-stopinjskih rotacijah okoli
    posamezne osi - preverja, da je konstrukcija matrike (Rz@Ry@Rx +
    translacija v zadnjem stolpcu) dejansko pravilna, ne le da funkcija
    vrne matriko pravilne oblike."""
    identity_case = build_reference_transform(10.0, -20.0, -442.0, 0.0, 0.0, 0.0)
    assert np.allclose(identity_case[:3, :3], np.eye(3)), (
        f"Pri ničelnih rotacijah mora biti rotacijski del identiteta, dobil {identity_case[:3, :3]}")
    assert np.allclose(identity_case[:3, 3], [10.0, -20.0, -442.0]), (
        f"Translacija mora biti natanko podana, dobil {identity_case[:3, 3]}")
    assert np.allclose(identity_case[3, :], [0.0, 0.0, 0.0, 1.0]), "Zadnja vrstica mora biti [0,0,0,1]"

    # 90 stopinj okoli Z (yaw) mora preslikati enotski X vektor na Y.
    yaw_90 = build_reference_transform(0.0, 0.0, 0.0, 0.0, 0.0, 90.0)
    assert np.allclose(yaw_90[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9), (
        f"90 stopinj okoli Z mora preslikati X-os na Y-os, dobil {yaw_90[:3, :3] @ [1.0, 0.0, 0.0]}")

    # 90 stopinj okoli X (roll) mora preslikati enotski Y vektor na Z.
    roll_90 = build_reference_transform(0.0, 0.0, 0.0, 90.0, 0.0, 0.0)
    assert np.allclose(roll_90[:3, :3] @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9), (
        f"90 stopinj okoli X mora preslikati Y-os na Z-os, dobil {roll_90[:3, :3] @ [0.0, 1.0, 0.0]}")

    # 90 stopinj okoli Y (pitch) mora preslikati enotski Z vektor na X.
    pitch_90 = build_reference_transform(0.0, 0.0, 0.0, 0.0, 90.0, 0.0)
    assert np.allclose(pitch_90[:3, :3] @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-9), (
        f"90 stopinj okoli Y mora preslikati Z-os na X-os, dobil {pitch_90[:3, :3] @ [0.0, 0.0, 1.0]}")

    print("  OK: build_reference_transform pravilno sestavi rotacijo in translacijo")


def run_self_tests() -> None:
    """Pognati vse hitre, samostojne teste novih parametrov za kalibracijo
    na resnične kamere (glej modulski komentar zgoraj in kamere.py)."""
    print("Running self-tests for new camera calibration parameters...")
    _test_noise_reference_distance()
    _test_distance_bias_permille()
    _test_global_planarity_bias()
    _test_top_camera_location_offset_distance()
    _test_build_reference_transform()
    print("All self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAD-to-scan point cloud registration")
    parser.add_argument("--self_test", action="store_true",
                        help="Run the fast, standalone self-tests for the new camera-calibration "
                             "noise parameters (noise_reference_distance_m, distance_bias_permille, "
                             "global_planarity_mm, build_reference_transform - see "
                             "run_self_tests()) and exit, instead of running registration. Does "
                             "not need --cad/--scan or any other argument")
    parser.add_argument("--cad", default="eHDS S Housing + CC s pottingom, fine.STL",
                        help="Path to the CAD mesh file")
    parser.add_argument("--scan", default="scan.ply",
                        help="Path to the real scan point cloud file")
    parser.add_argument("--extra_scans", nargs="*", default=[],
                        help="Paths to additional scan files of the SAME static placement to "
                             "merge with --scan for temporal noise averaging (see "
                             "load_and_merge_real_scans) - if the robot/sensor can capture "
                             "several (e.g. 3-5) consecutive scans of the part sitting still on "
                             "the table before it needs a final pose, merging them and letting "
                             "Phase 3's voxel downsampling average each voxel's points cuts "
                             "uncorrelated sensor noise by roughly sqrt(N) for free. No inter-scan "
                             "registration is done or needed - the sensor is fixed and the part "
                             "doesn't move between captures, so all scans already share one "
                             "coordinate frame")
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
    parser.add_argument("--depth_noise_at_1m", type=float, default=0.1,
                        help="Depth-noise std (mm) at --noise_reference_distance_m distance and "
                             "0deg incidence (straight-on) - despite the flag's name, this is "
                             "measured at --noise_reference_distance_m, not necessarily 1m (see "
                             "that flag). Scales up with both distance and incidence angle - see "
                             "--noise_distance_power/--max_incidence_deg - matching a real "
                             "triangulation sensor's behavior instead of a single flat std "
                             "everywhere")
    parser.add_argument("--noise_distance_power", type=float, default=2.0,
                        help="Exponent for how depth-noise std grows with distance "
                             "(distance_m/--noise_reference_distance_m ** this, calibrated to "
                             "equal --depth_noise_at_1m exactly at that reference distance). Also "
                             "used for --quantization_step_at_1m's distance scaling")
    parser.add_argument("--noise_reference_distance_m", type=float, default=1.0,
                        help="Distance (m) at which --depth_noise_at_1m was actually measured/"
                             "calibrated. Default 1.0 preserves old behavior, but real "
                             "triangulation sensor datasheets (see kamere.py) report noise at "
                             "their own working distance (e.g. 0.442m for Photoneo PhoXi S), "
                             "never at 1m - extrapolating a temporal-noise measurement taken over "
                             "a narrow working range (e.g. 384-520mm) out to 1m with the default "
                             "--noise_distance_power=2.0 can badly misestimate the true noise, "
                             "since the real trend over that narrow range is often much flatter "
                             "than quadratic. Set this to the sensor's own reference distance and "
                             "--depth_noise_at_1m to the noise measured there instead")
    parser.add_argument("--max_incidence_deg", type=float, default=75.0,
                        help="Incidence angle (degrees off straight-on) beyond which the "
                             "1/cos(incidence) noise-growth factor is clamped, so it can't blow "
                             "up right at grazing incidence where cos -> 0")
    parser.add_argument("--distance_bias_permille", type=float, default=0.0,
                        help="Systematic (not per-pixel-random) multiplicative distance "
                             "calibration error, as a fraction (permille) of the measured "
                             "distance, constant for the whole simulated capture - see "
                             "apply_distance_bias_mm() and kamere.py's 'Relative distance "
                             "accuracy'/'Dimension Trueness Error' spec entries, which report "
                             "this as an overall (mostly systematic, not random) accuracy figure "
                             "that the noise model previously had no parameter for. 0 (default) "
                             "disables it. Deterministic per run (like --translation_x/y/z), not "
                             "randomly redrawn, so you can sweep it to test tolerance to a known "
                             "systematic bias (e.g. --distance_bias_permille 1.25 for Photoneo's "
                             "spec)")
    parser.add_argument("--global_planarity_mm", type=float, default=0.0,
                        help="Amplitude (peak-to-valley range, mm) of a smooth, low-frequency "
                             "systematic distortion field added across the whole simulated depth "
                             "image, constant for the whole capture - see "
                             "generate_global_planarity_bias_mm() and kamere.py's 'Global "
                             "planarity'/'Global Planarity Trueness Error' spec entries. The "
                             "existing noise model (heteroscedastic gaussian, outliers, flying "
                             "pixels, quantization) is entirely random and zero-mean per pixel; "
                             "none of it represents this kind of smooth systematic bow/warp "
                             "across the field of view. 0 (default) disables it")
    parser.add_argument("--global_planarity_spatial_scale_px", type=float, default=None,
                        help="Gaussian-blur sigma (pixels) controlling how 'low-frequency' (smooth "
                             "over how large an area) the --global_planarity_mm distortion field "
                             "is. Default (unset): min(--camera_width_px, --camera_height_px) / 4")
    parser.add_argument("--outlier_probability", type=float, default=0.01,
                        help="Fraction of pixels that get a gross depth error (multipath, weak "
                             "SNR, bad stereo/structured-light match) instead of ordinary sensor "
                             "noise - a genuinely different error mechanism, not just a bit more "
                             "of the same jitter")
    parser.add_argument("--outlier_std_multiplier", type=float, default=15.0,
                        help="How much wider an outlier pixel's error distribution is than that "
                             "same pixel's ordinary noise std - see --outlier_probability")
    parser.add_argument("--flying_pixel_depth_jump_mm", type=float, default=3.0,
                        help="Depth difference (mm) between neighboring pixels that counts as a "
                             "discontinuity (part edge, pin edge, seam) eligible to become a "
                             "'flying pixel'. Set relative to your real edge features' scale "
                             "(e.g. 2-5mm for a pin edge)")
    parser.add_argument("--flying_pixel_probability", type=float, default=0.3,
                        help="Probability that a pixel sitting on a depth discontinuity (see "
                             "--flying_pixel_depth_jump_mm) becomes a flying pixel - an "
                             "interpolated, physically-nonexistent depth between the near and "
                             "far surface, the characteristic 'floating point cloud at edges' "
                             "artifact real depth sensors produce and plain Gaussian noise never does")
    parser.add_argument("--quantization_step_at_1m", type=float, default=0.05,
                        help="Depth quantization step (mm) at 1m distance - the sensor's ADC/"
                             "disparity resolution reports depth in discrete steps, not "
                             "continuously. Scales with distance the same way "
                             "--depth_noise_at_1m does (via --noise_distance_power)")
    parser.add_argument("--noise_spatial_correlation_px", type=float, default=1.5,
                        help="Gaussian-blur sigma (pixels) used to spatially correlate depth "
                             "noise between neighboring pixels, matching a real sensor's "
                             "matching/correlation window (which mixes several neighboring "
                             "pixels rather than measuring each independently) - roughly the "
                             "window's own size in pixels")
    parser.add_argument("--disable_occlusion", action="store_true",
                        help="Skip ray casting and densely sample the whole CAD mesh instead, "
                             "so the simulated scan keeps full coverage - to check whether "
                             "partial visibility (occlusion + field of view) is what's causing "
                             "the recovered transform to diverge from the reference")
    parser.add_argument("--camera_fov_deg", type=float, default=40.0,
                        help="Horizontal field of view (degrees) of the simulated depth camera "
                             "used for ray casting the target scan. Narrower = less of the "
                             "table around the part is captured, denser coverage of the part "
                             "itself for a given --camera_width_px/--camera_height_px")
    parser.add_argument("--camera_width_px", type=int, default=640,
                        help="Simulated camera image width in pixels (= number of rays cast "
                             "across the field of view). Higher = denser target point cloud, "
                             "slower ray casting")
    parser.add_argument("--camera_height_px", type=int, default=480,
                        help="Simulated camera image height in pixels, see --camera_width_px")
    parser.add_argument("--translation_x", type=float, default=0.0,
                        help="X translation (mm) of the ground-truth reference transform "
                             "(SIMULATED_TRANSFORM, see build_reference_transform) - the part's "
                             "position relative to the camera, which is ALWAYS fixed at the world "
                             "origin (0,0,0) for both real and simulated runs. Default 0 (part "
                             "centered under the camera in X) - set nonzero to empirically test "
                             "how far off-center (within the camera's field of view) the part can "
                             "sit and still be registered correctly")
    parser.add_argument("--translation_y", type=float, default=0.0,
                        help="Same as --translation_x but for the Y axis")
    parser.add_argument("--translation_z", type=float, default=-500.0,
                        help="Z translation (mm) of the ground-truth reference transform - the "
                             "part's distance from the camera (at the origin), so this must be "
                             "NEGATIVE (camera looks down at the part below it). Default -500.0 is "
                             "a generic placeholder; set it to MINUS the actual camera's sweet-spot/"
                             "focal working distance instead (e.g. -442 for Photoneo PhoXi S, -600 "
                             "for Zivid 2+ M60, -500 for Mech-Eye PRO S at its 500mm focal "
                             "configuration - see kamere.py). kontrolna_plosca.py's "
                             "resolve_camera_params() does this automatically for the selected camera")
    parser.add_argument("--rotation_x_deg", type=float, default=0.0,
                        help="Rotation (degrees) of the ground-truth reference transform about "
                             "the world X axis (roll), applied before Y/Z rotation - see "
                             "build_reference_transform. Default 0 (part lying flat, no tilt) - "
                             "set nonzero to empirically test how much roll/pitch tilt (part not "
                             "lying perfectly flat on the table) the pipeline can tolerate. "
                             "yaw_sweep_registration estimates and corrects tilt from target's "
                             "surface normal, tested to hold up to ~25-27 deg combined tilt before "
                             "breaking down around 30")
    parser.add_argument("--rotation_y_deg", type=float, default=0.0,
                        help="Same as --rotation_x_deg but about the world Y axis (pitch)")
    parser.add_argument("--rotation_z_deg", type=float, default=0.0,
                        help="Same as --rotation_x_deg but about the world Z axis (yaw) - applied "
                             "last, after X and Y rotation. Note this is the GROUND-TRUTH yaw the "
                             "pipeline must recover, unrelated to --yaw_step_deg's search "
                             "resolution below")
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
    parser.add_argument("--yaw_step_deg", type=float, default=6.0,
                        help="Degrees between candidate yaw angles in the sweep (halves the "
                             "candidate count, and with it the sweep's runtime, at each doubling). "
                             "Smaller = finer coarse search but slower; the multi-scale ICP stage "
                             "refines the winning angle to full precision afterward regardless. "
                             "Default raised from 3.0 to 6.0 after empirically comparing both "
                             "across several off-center/rotated test poses (same camera noise "
                             "seed): final rotation/translation error after ICP was "
                             "indistinguishable between 3.0 and 6.0 in every case that converged "
                             "at all (a pose that failed to converge failed almost identically "
                             "at both step sizes, so coarsening isn't what caused it) - the ICP "
                             "refinement stage does in fact recover the coarser yaw discretization "
                             "as expected. Lower this back to 3.0 (or finer) if you see the coarse "
                             "sweep repeatedly landing on the wrong side of a near-symmetric "
                             "feature for your specific part")
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
    parser.add_argument("--yaw_sweep_n_jobs", type=int, default=-1,
                        help="Parallel workers for evaluating yaw sweep candidates (they're "
                             "fully independent - see yaw_sweep_registration/_evaluate_yaw_candidate). "
                             "-1 (default) uses all CPU cores via joblib, 1 forces serial "
                             "evaluation (falls back to serial automatically either way if "
                             "joblib isn't installed). Empirically gives the same winning "
                             "candidate as serial (verified over repeated runs). Measured speedup "
                             "is workload-dependent: a single one-off run pays a one-time "
                             "process-pool startup cost that can outweigh the savings from just "
                             "120 cheap candidates (measured roughly break-even, occasionally "
                             "slightly slower than serial); repeated run_registration calls in "
                             "the same process (e.g. camera_benchmark.py) reuse the same pool "
                             "and saw ~1.7x speedup on the yaw sweep step specifically")
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
    parser.add_argument("--robust_kernel_k_factor", type=float, default=1.0,
                        help="Scale (as a multiple of each ICP stage's own voxel size) of the "
                             "Tukey robust kernel (TukeyLoss) used in the point-to-plane ICP "
                             "estimation in refine_registration. Without a robust kernel, "
                             "outliers/flying pixels that fall within max_correspondence_distance "
                             "(so they DO become correspondences) pull the solution with full, "
                             "unbounded weight, same as a clean point; Tukey downweights a "
                             "residual once it exceeds k and zeroes it out entirely above 2k, "
                             "effectively excluding outliers instead of trusting them equally. "
                             "Lower this (e.g. 0.5) for noisier scans to reject outliers harder, "
                             "raise it if a clean scan's legitimate near-edge points are being "
                             "excessively downweighted")
    parser.add_argument("--min_coarse_fitness_for_icp", type=float, default=0.1,
                        help="Skip ICP entirely if the coarse (yaw sweep) fitness is below "
                             "this. ICP's linearized solver can diverge to nonsense (meter/"
                             "km-scale translations) when seeded with a badly wrong pose, so "
                             "it's safer to skip it than trust its output on a bad seed")
    parser.add_argument("--table_size_x", type=float, default=500.0,
                        help="Width (mm, X axis) of the known table/work-envelope area the "
                             "part can be placed in. yaw_sweep_registration rejects any yaw "
                             "candidate whose implied placement would put the part's center "
                             "outside this area - a physical prior that keeps an implausible "
                             "placement from winning purely on ICP fitness. Set to the real "
                             "robot's actual work envelope")
    parser.add_argument("--table_size_y", type=float, default=500.0,
                        help="Depth (mm, Y axis) of the known table/work-envelope area, see "
                             "--table_size_x")
    parser.add_argument("--table_center_x", type=float, default=0.0,
                        help="X coordinate (mm, world/CAD frame) of the table area's center - "
                             "together with --table_size_x/--table_size_y defines the "
                             "rectangular region a candidate placement must fall within")
    parser.add_argument("--table_center_y", type=float, default=0.0,
                        help="Y coordinate (mm, world/CAD frame) of the table area's center, "
                             "see --table_center_x")
    parser.add_argument("--add_table_background", action="store_true",
                        help="Add a flat table plane (sized/positioned by --table_size_x/y and "
                             "--table_center_x/y - the same table used for the physical prior) "
                             "into the ray casting scene, so rays that miss the part hit the "
                             "table instead of nothing - the simulated scan then includes "
                             "background/table points around the part, to see how the pipeline "
                             "handles that instead of a clean part-only scan")
    parser.add_argument("--remove_table_background", dest="remove_table_background_flag",
                        action="store_true", default=None,
                        help="Remove likely table/background points from target before "
                             "registration, using the part's own known height (from source/CAD) "
                             "instead of 'largest plane' detection (e.g. RANSAC segment_plane) - "
                             "the latter can mistakenly remove the part itself if it occupies a "
                             "large flat area of the field of view. Assumes the lowest (or "
                             "highest, see --up_axis/flip) observed points in target are the "
                             "table, since the part cannot extend below the table it rests on. "
                             "Default: on automatically for --use_real_scan (a real scan almost "
                             "always includes some table/background, and leaving it in can make "
                             "the tilt-normal estimate lock onto the table instead of the part's "
                             "top surface), off for simulated scans (no table in the cloud unless "
                             "--add_table_background is also set) - pass --no_remove_table_background "
                             "to force it off even for a real scan")
    parser.add_argument("--no_remove_table_background", dest="remove_table_background_flag",
                        action="store_false",
                        help="Force table/background removal off, overriding the --use_real_scan "
                             "default described under --remove_table_background")
    parser.add_argument("--table_removal_margin_mm", type=float, default=10.0,
                        help="Margin (mm) added on both ends of the kept height band in "
                             "--remove_table_background: excludes points within this distance "
                             "of the detected table level, and points more than the part's own "
                             "height plus this margin above it")
    parser.add_argument("--no_filter_grazing_incidence", dest="filter_grazing_incidence",
                        action="store_false", default=True,
                        help="Disable the grazing-incidence (flying pixel) point filter, on by "
                             "default for both real and simulated scans. Removes points whose "
                             "local normal is nearly perpendicular to the camera view direction "
                             "(incidence angle above --flying_pixel_filter_max_incidence_deg) - "
                             "at such a steep angle the sensor's matching window straddles two "
                             "depth levels at once and commonly reports an interpolated, "
                             "physically-nonexistent depth (see simulate_camera_scan's own "
                             "flying-pixel noise model). Unlike --filter_real_scan's statistical "
                             "outlier removal, this targets the actual physical cause "
                             "(incidence angle) instead of the symptom (isolated distance), which "
                             "matters because flying pixels form dense clusters along edges, not "
                             "isolated outliers")
    parser.add_argument("--flying_pixel_filter_max_incidence_deg", type=float, default=80.0,
                        help="Incidence angle (degrees off straight-on) above which a point is "
                             "removed by --filter_grazing_incidence")
    parser.add_argument("--grazing_incidence_normal_radius_factor", type=float, default=2.0,
                        help="Normal-estimation radius for --filter_grazing_incidence, as a "
                             "multiple of --voxel_size, applied to the RAW (not yet downsampled) "
                             "scan")
    parser.add_argument("--asymmetric_check_top_fraction", type=float, default=0.15,
                        help="Fraction of source's points (by self-rotation distance, see "
                             "identify_asymmetric_points) treated as the part's distinctive/"
                             "asymmetric features for the post-ICP safety check. Lower = only "
                             "the most distinctive points count, stricter/noisier check; higher "
                             "= includes more borderline bulk-geometry points, looser check")
    parser.add_argument("--asymmetric_check_residual_factor", type=float, default=3.0,
                        help="A distinctive/asymmetric point counts as 'aligned' after ICP if "
                             "its distance to target is within this multiple of --voxel_size. "
                             "The safety check warns if fewer than half of them meet that bar")
    parser.add_argument("--tilt_normal_top_band_fraction", type=float, default=0.5,
                        help="Fraction of target's points (by height percentile along --up_axis) "
                             "fed into the RANSAC plane fit that estimates the part's tilt "
                             "(replaces a plain PCA-over-everything normal estimate, which has no "
                             "robust weighting and can lock onto the table instead of the part's "
                             "top surface if the table isn't removed first - see "
                             "--remove_table_background)")
    parser.add_argument("--height_percentile", type=float, default=99.5,
                        help="Percentile (0-100) used instead of a plain max() when comparing "
                             "source/target height along --up_axis to estimate the Z-translation "
                             "in the yaw sweep. A single flying-pixel/outlier (up to "
                             "--outlier_std_multiplier times the ordinary noise std in this "
                             "pipeline's own sensor model) can move a plain max() by several mm; "
                             "a high percentile treats the 'top' as a band of the topmost points "
                             "instead of one single point")
    parser.add_argument("--target_coverage_distance_factor", type=float, default=2.0,
                        help="Distance threshold for the target->source coverage check, as a "
                             "multiple of --voxel_size: a scanned point counts as 'on the part' "
                             "if it's within this distance of the CAD surface under the final "
                             "pose. This is the primary occlusion-robust overlap check used for "
                             "ACCEPT/REJECT - see --min_accept_target_coverage and "
                             "evaluate_target_coverage()'s docstring")
    parser.add_argument("--min_accept_target_coverage", type=float, default=0.85,
                        help="Minimum fraction of scanned (target) points that must lie on the "
                             "CAD surface (within --target_coverage_distance_factor * voxel_size) "
                             "for the PASS/FAIL accept decision to be ACCEPT. This is the PRIMARY "
                             "overlap gate: unlike --min_accept_fitness (source->target, "
                             "depressed by occlusion even for a correct pose), target->source "
                             "coverage measures whether the points actually measured lie on the "
                             "part, regardless of how much of the part the scan happened to see. "
                             "PLACEHOLDER default, not empirically calibrated - see "
                             "decide_registration_outcome()'s docstring for how to calibrate it "
                             "(Monte Carlo over known-good poses using this file's own "
                             "--translation_x/y/z/--rotation_x/y/z_deg simulation harness) before "
                             "trusting it in production")
    parser.add_argument("--min_accept_fitness", type=float, default=0.1,
                        help="Minimum final ICP fitness (Open3D's source->target overlap ratio) "
                             "required for ACCEPT. Intentionally just a LOW sanity floor, not the "
                             "primary overlap gate - this fitness is depressed by occlusion alone "
                             "(target is necessarily a partial view of the part) even under a "
                             "perfectly correct pose, so a high fixed threshold here would reject "
                             "good poses on heavily-occluded scans. See "
                             "--min_accept_target_coverage for the actual (occlusion-robust) "
                             "overlap gate")
    parser.add_argument("--max_accept_inlier_rmse", type=float, default=None,
                        help="Maximum final ICP inlier_rmse allowed for ACCEPT. Default: "
                             "1.5 * --voxel_size, the same common threshold icp_result's own "
                             "fitness/inlier_rmse were evaluated under in refine_registration")
    parser.add_argument("--min_accept_asymmetric_fraction", type=float, default=0.5,
                        help="Minimum fraction of the part's distinctive/asymmetric features "
                             "(see --asymmetric_check_top_fraction/--asymmetric_check_residual_factor) "
                             "that must align within threshold for ACCEPT - guards against a "
                             "registration that converged to a symmetric look-alike orientation "
                             "(e.g. a repeated mounting boss) despite good overall fitness")
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
    return parser.parse_args()


def main() -> None:
    # Python zahteva, da globalna deklaracija imena prehiti VSAKO dodelitev
    # vanj v isti funkciji - ker main() spodaj prepiše modulski
    # SIMULATED_TRANSFORM (iz --translation_x/y/z/--rotation_x/y/z_deg),
    # mora biti `global` deklariran tukaj, pred to dodelitvijo.
    global SIMULATED_TRANSFORM
    np.random.seed(42)   # nastavimo seme za ponovljive naključne rezultate
    base_path = Path(__file__).resolve().parent   # mapa, kjer leži ta skripta
    args = parse_args()   # razberemo argumente ukazne vrstice

    if args.self_test:   # samo poženemo hitre samostojne teste novih parametrov in končamo, brez CAD/registracije
        run_self_tests()
        return

    cad_path = base_path / args.cad   # polna pot do CAD datoteke
    real_scan_path = base_path / args.scan   # polna pot do datoteke s pravim skenom

    # Referenčna (ground-truth) transformacija - poza dela GLEDE NA kamero,
    # ki je vedno fiksirana v izhodišču (0,0,0), za realne IN simulirane
    # skene enako (glej build_reference_transform in run_registration).
    # Prejšnja različica je imela fiksno hardcodirano SIMULATED_TRANSFORM
    # (naklon ~30 stopinj, X/Y zamik) plus ločen aditiven --test_tilt_deg/
    # --test_translation_x/y mehanizem za vbrizganje dodatnih motenj nanjo -
    # zdaj namesto tega --translation_x/y/z/--rotation_x/y/z_deg NEPOSREDNO
    # in POPOLNOMA definirajo referenčno pozo v enem koraku, brez ločenega
    # "osnovnega" stanja.
    #
    # translation_x/y/z sama po sebi merita glede na CAD-ov lastni izvor
    # koordinat, ki NI nujno poravnan s centrom skenirane "sealing" regije
    # (glej compute_native_reference_point docstring) - brez popravka
    # spodaj translation_x=y=0 ne bi postavil dela na sredino kamerinega
    # pogleda, translation_z pa ne bi pomenil prave razdalje do skenirane
    # površine, kar bi simulirano vidno polje (scanning area) naredilo
    # ožje/širše in napačno centrirano glede na kamerino specifikacijo
    # (glej kamere.py). Popravek se izračuna SAMO za simulacijo - pri
    # pravem skenu SIMULATED_TRANSFORM ni uporabljen za nič.
    rotation_only = build_reference_transform(0.0, 0.0, 0.0, args.rotation_x_deg,
                                              args.rotation_y_deg, args.rotation_z_deg)
    rotation_matrix = rotation_only[:3, :3]
    requested_translation = np.array([args.translation_x, args.translation_y, args.translation_z])
    if args.use_real_scan:
        corrected_translation = requested_translation   # SIMULATED_TRANSFORM se pri pravem skenu ne uporablja - popravek ni potreben
    else:
        reference_mesh = load_cad_mesh(cad_path)   # NEtransformirana CAD mreža, samo za izračun referenčne točke
        reference_point = compute_native_reference_point(
            reference_mesh, top_fraction=args.top_fraction, up_axis=args.up_axis,
            flip_up_direction=args.flip_up_direction)
        corrected_translation = requested_translation - rotation_matrix @ reference_point
        print(f"Native CAD reference point (center of the scanned sealing region, "
              f"pre-rotation/translation): {reference_point} - --translation_x/y/z is "
              f"relative to THIS point, not the CAD file's raw coordinate origin")

    SIMULATED_TRANSFORM = build_reference_transform(
        translation_x=corrected_translation[0], translation_y=corrected_translation[1],
        translation_z=corrected_translation[2], rotation_x_deg=args.rotation_x_deg,
        rotation_y_deg=args.rotation_y_deg, rotation_z_deg=args.rotation_z_deg)
    print(f"Reference (ground-truth) transform - part pose relative to the camera "
          f"(fixed at the origin):\n{SIMULATED_TRANSFORM}\n")

    _source, _target, _coarse_result, _icp_result, decision = run_registration(
        cad_path=cad_path,
        scan_path=real_scan_path,
        extra_scan_paths=[base_path / p for p in args.extra_scans],
        use_real_scan=args.use_real_scan,
        voxel_size=args.voxel_size,
        sample_point_count=args.sample_points,
        filter_grazing_incidence=args.filter_grazing_incidence,
        flying_pixel_filter_max_incidence_deg=args.flying_pixel_filter_max_incidence_deg,
        grazing_incidence_normal_radius_factor=args.grazing_incidence_normal_radius_factor,
        depth_noise_at_1m=args.depth_noise_at_1m,
        noise_distance_power=args.noise_distance_power,
        noise_reference_distance_m=args.noise_reference_distance_m,
        max_incidence_deg=args.max_incidence_deg,
        distance_bias_permille=args.distance_bias_permille,
        global_planarity_mm=args.global_planarity_mm,
        global_planarity_spatial_scale_px=args.global_planarity_spatial_scale_px,
        outlier_probability=args.outlier_probability,
        outlier_std_multiplier=args.outlier_std_multiplier,
        flying_pixel_depth_jump_mm=args.flying_pixel_depth_jump_mm,
        flying_pixel_probability=args.flying_pixel_probability,
        quantization_step_at_1m=args.quantization_step_at_1m,
        noise_spatial_correlation_px=args.noise_spatial_correlation_px,
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
        robust_kernel_k_factor=args.robust_kernel_k_factor,
        min_coarse_fitness_for_icp=args.min_coarse_fitness_for_icp,
        use_cad_cache=args.use_cad_cache,
        use_preprocess_cache=args.use_preprocess_cache,
        cache_key_tag=(f"_tx{args.translation_x}_ty{args.translation_y}_tz{args.translation_z}"
                      f"_rx{args.rotation_x_deg}_ry{args.rotation_y_deg}_rz{args.rotation_z_deg}"),
        camera_fov_deg=args.camera_fov_deg,
        camera_width_px=args.camera_width_px,
        camera_height_px=args.camera_height_px,
        table_size_x=args.table_size_x,
        table_size_y=args.table_size_y,
        table_center=np.array([args.table_center_x, args.table_center_y]),
        add_table_background=args.add_table_background,
        remove_table_background_flag=args.remove_table_background_flag,
        table_removal_margin_mm=args.table_removal_margin_mm,
        tilt_normal_top_band_fraction=args.tilt_normal_top_band_fraction,
        height_percentile=args.height_percentile,
        yaw_sweep_n_jobs=args.yaw_sweep_n_jobs,
        asymmetric_check_top_fraction=args.asymmetric_check_top_fraction,
        asymmetric_check_residual_factor=args.asymmetric_check_residual_factor,
        target_coverage_distance_factor=args.target_coverage_distance_factor,
        min_accept_fitness=args.min_accept_fitness,
        max_accept_inlier_rmse=args.max_accept_inlier_rmse,
        min_accept_asymmetric_fraction=args.min_accept_asymmetric_fraction,
        min_accept_target_coverage=args.min_accept_target_coverage)

    if not decision["accepted"]:
        # Neničeln izhodni exit-code je signal, ki ga robotov klicoči
        # proces (ta skripta se navadno kliče kot podproces enega koraka
        # v ciklu prijemanja) dejansko lahko preveri, ne da bi moral
        # razčlenjevati stdout - glej decide_registration_outcome zgoraj.
        sys.exit(1)


if __name__ == "__main__":   # skripta se poganja neposredno (ne uvožena kot modul)
    main()   # zaženemo glavni tok programa
