"""camera_benchmark.py - statistična primerjava treh kamer (glej kamere.py)
na natančnost registracije: veliko naključnih semen na majhnem naboru
reprezentativnih pozicij/zasukov, namesto ene same (od sreče šumovnega
vzorca odvisne) meritve na pogoj - glej DOCSTRING SPODAJ za natančen
eksperimentalni načrt in POMEMBNO OPOZORILO o tem, kaj je od zahtevanih
"obveznih popravkov" že vgrajeno v main_trial.py in zato TU NI ponovljeno.

EKSPERIMENTALNI NAČRT (glej konfiguracijske konstante spodaj):
  Za vsako kamero v kamere.CAMERAS:
    - 7 testnih pozicij v ravnini mize (center, 4 vogali, sredini levega/
      desnega roba), izpeljanih iz KAMERINEGA LASTNEGA scanning_area_mm pri
      working_distance_mm, zamaknjenih od roba za polmer CAD dela (izračunan
      iz naloženega CAD-a po crop_top_region) + varnostni rob, da del NIKOLI
      ne zapusti vidnega polja - testiramo izven-sredinsko natančnost, ne
      "del manjka".
    - 4 yaw koti, namerno zamaknjeni za pol koraka yaw-sweep preiskovanja
      (yaw_step_deg=3.0 -> zamik 1.5 stopinje): 1.5, 91.5, 181.5, 271.5 -
      testira najslabši primer diskretizacije yaw, namesto referenc, ki
      pristanejo točno na testiranem kandidatu.
    - N_SEEDS naključnih semen na (pozicija x yaw) kombinacijo.
  Skupaj 7 x 4 x N_SEEDS zagonov na kamero.

POMEMBNO - RAZLIKA OD PRVOTNIH NAVODIL (main_trial.py se je od takrat
spremenil, glej spodaj za natančno stanje):
  1. "Fiksna lokacija/orientacija kamere" - main_trial.py TO ŽE POČNE SAMO
     OD SEBE, brez potrebe po monkeypatchu: kamera je VEDNO na (0,0,0)
     (run_registration nima več fixed_camera_location parametra - odstranjen),
     IN simulate_camera_scan že cilja žarke v FIKSNI smeri (naravnost
     navzdol, vzdolž -up_axis), NE na mesh.get_center() - to natanko
     popravlja bug, ki ga opisujejo prvotna navodila. Zato TA skripta NE
     kliče top_camera_location niti ne monkeypatcha ray-aiminga - ni
     potrebno (potrjeno ločeno, izven te datoteke, prek main_trial.
     simulate_camera_scan neposredno - glej validacijsko poročilo).
  2. Referenčna (ground-truth) transformacija se sestavi z
     main_trial.build_reference_transform() +
     main_trial.compute_native_reference_point() (glej
     build_ground_truth_transform spodaj) - to je nova, dosledna pot, ki jo
     main_trial.main() sam uporablja za --translation_x/y/z/--rotation_z_deg,
     ne ročno sestavljanje matrike.
  Vse OSTALO iz prvotnih navodil (SPEC_OVERRIDES za Mech-Eye, skaliranje
  noise_spatial_correlation_px z ločljivostjo, brez-vizualizacije,
  ponovljivost/seed, robustnost/try-except, use_preprocess_cache=False,
  Excel izhod s 3 listi + resume, smoke test) JE implementirano spodaj kot
  zahtevano.
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import kamere
import kontrolna_plosca as kp
import main_trial as m

BASE_PATH = Path(__file__).resolve().parent
CAD_FILE = "eHDS S Housing + CC s pottingom, fine.STL"
OUTPUT_FILE = BASE_PATH / "camera_benchmark.xlsx"


# =============================================================================
# KONFIGURACIJA EKSPERIMENTA
# =============================================================================
RESOLUTION_SCALE = 0.25          # skalira vsako kamerino depth_map_resolution_px navzdol (hitrost vs. natančnost)
N_SEEDS = 13                     # naključnih semen na (pozicija x yaw) kombinacijo
SEED_BASE = 1000                 # dejanska semena so SEED_BASE, SEED_BASE+1, ...
YAW_ANGLES_DEG = [1.5, 91.5, 181.5, 271.5]   # namerno zamaknjeno za pol yaw_step_deg (3.0/2)
POSITION_MARGIN_EXTRA_MM = 10.0  # dodaten varnostni rob čez CAD-ov lastni XY polmer

TOP_FRACTION = 0.35
UP_AXIS = 2
FLIP_UP_DIRECTION = False

# Manjkajoči specs podatki za Mech-Eye - KONZERVATIVNE OCENE, primerljive z
# ostalima dvema kamerama (glej kamere.py - global_planarity_mm in
# relative_distance_accuracy_permille NISTA bila podana v prejeti
# specifikaciji, torej bi se sicer privzeto simulirala kot 0, kar bi Mech-Eye
# dalo nepošteno prednost "po izpustitvi"). Mutira kamere.CAMERAS SAMO v tem
# procesu (v pomnilniku) - ne spreminja kamere.py na disku.
SPEC_OVERRIDES = {
    "mecheye_pro_s": {
        "global_planarity_mm": 0.15,               # OCENA - primerljivo s Photoneo (0.22) / Zivid (0.10)
        "relative_distance_accuracy_permille": 1.5,   # OCENA - primerljivo s Photoneo (1.25) / Zivid (2.0)
    },
}

# Parametri, enaki za VSE zagone (glej main_trial.py --help za razlago
# vsakega) - vse, kar je kamera-specifično (FOV, ločljivost, šum ...), se
# doda posebej v run_single() prek kontrolna_plosca.resolve_camera_params().
COMMON_RUN_KWARGS = dict(
    voxel_size=1.0,
    sample_point_count=200000,
    add_table_background=True,
    remove_table_background_flag=True,
    use_cad_cache=True,          # isti CAD za vse zagone - vzorčenje se izračuna samo enkrat
    use_preprocess_cache=False,  # vsak target je unikaten (nov seed) - predpomnjenje bi samo kopičilo tisoče datotek brez koristi
    top_fraction=TOP_FRACTION,
    up_axis=UP_AXIS,
    flip_up_direction=FLIP_UP_DIRECTION,
    table_size_x=500.0,   # eksplicitno (ne le zanašanje na run_registration-ov privzetek) - glej table_half_extent_mm sanity check spodaj
    table_size_y=500.0,
    # Eksplicitno pripeto na 3.0 (ne zanašanje na main_trial.py-jev lasten
    # privzetek, ki se je medtem spremenil na 6.0) - YAW_ANGLES_DEG zgoraj
    # je namenoma zamaknjen za natanko POL yaw_step_deg (1.5 = 3.0/2), da
    # testira najslabši primer diskretizacije za TA korak; sprememba
    # yaw_step_deg tukaj brez ustreznega preračuna YAW_ANGLES_DEG bi to
    # razmerje pokvarila.
    yaw_step_deg=3.0,
)

RUNS_COLUMNS = [
    "camera", "position", "tx_mm", "ty_mm", "yaw_deg", "seed", "status",
    "rot_error_deg", "trans_error_mm", "coarse_rot_error_deg", "coarse_trans_error_mm",
    "fitness", "inlier_rmse", "asymmetric_fraction", "target_coverage",
    "accepted", "reject_reasons", "estimated_fields", "runtime_s", "error_message",
]


def apply_spec_overrides() -> None:
    """Vpiše SPEC_OVERRIDES v kamere.CAMERAS (v pomnilniku, samo za ta
    proces) - klicati ENKRAT, pred kakršnimkoli klicem
    kontrolna_plosca.resolve_camera_params()."""
    for camera_name, overrides in SPEC_OVERRIDES.items():
        kamere.CAMERAS[camera_name].update(overrides)
        print(f"  SPEC_OVERRIDES uporabljen za '{camera_name}': {overrides}")


def compute_cad_reference_and_radius(cad_path: Path) -> tuple[np.ndarray, float]:
    """Izračuna (a) referenčno točko (center zgornje "sealing" regije v
    CAD-ovem lastnem, NEtransformiranem okviru - glej
    main_trial.compute_native_reference_point) in (b) njen XY polmer
    (polovica diagonale bounding boxa te regije) - ista fizična regija za
    VSE tri kamere, saj gre za isti CAD del."""
    mesh = m.load_cad_mesh(cad_path)
    reference_point = m.compute_native_reference_point(
        mesh, top_fraction=TOP_FRACTION, up_axis=UP_AXIS, flip_up_direction=FLIP_UP_DIRECTION)
    cropped = m.crop_top_region(mesh, top_fraction=TOP_FRACTION, up_axis=UP_AXIS,
                                flip_up_direction=FLIP_UP_DIRECTION)
    half_extent_xy = (np.asarray(cropped.get_max_bound())[:2] - np.asarray(cropped.get_min_bound())[:2]) / 2.0
    radius = float(np.linalg.norm(half_extent_xy))   # polovica diagonale = konzervativen (nekoliko precenjen) polmer
    return reference_point, radius


def compute_positions(scanning_area_mm: dict, margin_mm: float) -> list[tuple[str, float, float]]:
    """7 testnih pozicij (center, 4 vogali, sredini levega/desnega roba)
    znotraj kamerinega scanning_area_mm, zamaknjenih od roba za margin_mm
    (CAD-ov lastni XY polmer + varnostni rob), da del NIKOLI ne zapusti
    vidnega polja - testiramo natančnost izven-sredinske registracije, ne
    "del manjka"."""
    usable_x = scanning_area_mm["width"] / 2.0 - margin_mm
    usable_y = scanning_area_mm["height"] / 2.0 - margin_mm
    if usable_x <= 0 or usable_y <= 0:
        raise ValueError(
            f"margin_mm={margin_mm:.1f} je prevelik za scanning_area_mm={scanning_area_mm} "
            f"(usable_x={usable_x:.1f}, usable_y={usable_y:.1f}) - zmanjšaj CAD polmer ali "
            f"POSITION_MARGIN_EXTRA_MM")
    return [
        ("center", 0.0, 0.0),
        ("corner_NE", usable_x, usable_y),
        ("corner_NW", -usable_x, usable_y),
        ("corner_SE", usable_x, -usable_y),
        ("corner_SW", -usable_x, -usable_y),
        ("edge_E", usable_x, 0.0),
        ("edge_W", -usable_x, 0.0),
    ]


def build_ground_truth_transform(reference_point: np.ndarray, tx: float, ty: float,
                                 translation_z: float, yaw_deg: float) -> np.ndarray:
    """Sestavi referenčno (ground-truth) transformacijo za en zagon - ISTA
    logika, ki jo main_trial.main() uporabi za --translation_x/y/z/
    --rotation_z_deg (glej main_trial.compute_native_reference_point
    docstring za razlago, zakaj je popravek za referenčno točko potreben:
    CAD-ov lastni izvor koordinat ni poravnan s centrom skenirane regije)."""
    rotation_only = m.build_reference_transform(0.0, 0.0, 0.0, 0.0, 0.0, yaw_deg)
    rotation_matrix = rotation_only[:3, :3]
    requested = np.array([tx, ty, translation_z])
    corrected = requested - rotation_matrix @ reference_point
    return m.build_reference_transform(corrected[0], corrected[1], corrected[2], 0.0, 0.0, yaw_deg)


def load_existing_runs() -> tuple[list[dict], set]:
    """Prebere obstoječi Excel (če obstaja) za podporo nadaljevanju -
    vrne (seznam že opravljenih vrstic, množica (camera,tx,ty,yaw,seed)
    ključev, ki jih je treba PRESKOČITI)."""
    if not OUTPUT_FILE.exists():
        return [], set()
    existing = pd.read_excel(OUTPUT_FILE, sheet_name="runs")
    done = set()
    for row in existing.itertuples(index=False):
        done.add((row.camera, round(float(row.tx_mm), 3), round(float(row.ty_mm), 3),
                  round(float(row.yaw_deg), 3), int(row.seed)))
    print(f"Najdena obstoječa datoteka z {len(existing)} vrsticami - te kombinacije bodo preskočene.")
    return existing.to_dict("records"), done


def build_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sestavi povzetek po kameri in po (kamera x pozicija) - štetje,
    štetje napak, mean/median/p95/max napak translacije/rotacije, mean
    pokritosti, delež sprejetih (accept rate)."""
    def stats(group: pd.DataFrame) -> pd.Series:
        ok = group[group["status"] == "ok"]
        return pd.Series({
            "count": len(group),
            "error_count": int((group["status"] == "error").sum()),
            "trans_error_mm_mean": ok["trans_error_mm"].mean(),
            "trans_error_mm_median": ok["trans_error_mm"].median(),
            "trans_error_mm_p95": ok["trans_error_mm"].quantile(0.95) if len(ok) else np.nan,
            "trans_error_mm_max": ok["trans_error_mm"].max(),
            "rot_error_deg_mean": ok["rot_error_deg"].mean(),
            "rot_error_deg_median": ok["rot_error_deg"].median(),
            "rot_error_deg_p95": ok["rot_error_deg"].quantile(0.95) if len(ok) else np.nan,
            "rot_error_deg_max": ok["rot_error_deg"].max(),
            "target_coverage_mean": ok["target_coverage"].mean(),
            "accept_rate": ok["accepted"].mean() if len(ok) else np.nan,
        })

    by_camera = df.groupby("camera", group_keys=False)[df.columns].apply(stats).reset_index()
    by_camera_position = df.groupby(["camera", "position"], group_keys=False)[df.columns].apply(stats).reset_index()
    return {"by_camera": by_camera, "by_camera_position": by_camera_position}


def save_excel(runs_rows: list[dict], config_rows: list[dict]) -> None:
    """Prepiše celoten Excel (runs/summary/config) - pri tej velikosti
    (do ~1100 vrstic) je popoln prepis vsak zagon dovolj hiter in enostavno
    zanesljiv (glej modulski docstring - to je bilo eksplicitno zahtevano)."""
    runs_df = pd.DataFrame(runs_rows, columns=RUNS_COLUMNS)
    summary = build_summary(runs_df) if len(runs_df) else {
        "by_camera": pd.DataFrame(), "by_camera_position": pd.DataFrame()}

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        runs_df.to_excel(writer, sheet_name="runs", index=False)

        summary["by_camera"].to_excel(writer, sheet_name="summary", index=False, startrow=0)
        note_row = len(summary["by_camera"]) + 2
        pd.DataFrame([[
            "OPOMBA: accept_rate uporablja NEUMERJENE placeholder pragove "
            "(glej main_trial.decide_registration_outcome) - kamere razvrščajte "
            "po trans_error_mm/rot_error_deg, NE po accept_rate."]]).to_excel(
            writer, sheet_name="summary", index=False, header=False,
            startrow=note_row, startcol=0)
        by_cam_pos_row = note_row + 3
        summary["by_camera_position"].to_excel(
            writer, sheet_name="summary", index=False, startrow=by_cam_pos_row)

        pd.DataFrame(config_rows).to_excel(writer, sheet_name="config", index=False)


def run_single(camera_name: str, position_label: str, tx: float, ty: float, yaw_deg: float,
              seed: int, reference_point: np.ndarray, camera_params: dict, cad_path: Path,
              estimated_fields: str, cache_key_tag: str) -> dict:
    """Požene EN zagon registracije in vrne eno vrstico rezultatov (slovar,
    ustreza RUNS_COLUMNS). Nikoli ne vrže naprej - napake ujame in zabeleži
    kot status='error', da ena neuspešna kombinacija ne prekine sweep-a."""
    row: dict = {
        "camera": camera_name, "position": position_label, "tx_mm": tx, "ty_mm": ty,
        "yaw_deg": yaw_deg, "seed": seed, "estimated_fields": estimated_fields,
    }

    translation_z = camera_params["translation_z"]
    m.SIMULATED_TRANSFORM = build_ground_truth_transform(reference_point, tx, ty, translation_z, yaw_deg)

    np.random.seed(seed)   # ponovljivost - takoj PRED klicem run_registration
    start = time.time()
    try:
        _source, _target, coarse_result, icp_result, decision = m.run_registration(
            cad_path=cad_path,
            camera_fov_deg=camera_params["camera_fov_deg"],
            camera_width_px=camera_params["camera_width_px"],
            camera_height_px=camera_params["camera_height_px"],
            depth_noise_at_1m=camera_params["depth_noise_at_1m"],
            noise_reference_distance_m=camera_params["noise_reference_distance_m"],
            distance_bias_permille=camera_params.get("distance_bias_permille", 0.0),
            global_planarity_mm=camera_params.get("global_planarity_mm", 0.0),
            noise_spatial_correlation_px=camera_params.get("noise_spatial_correlation_px", 1.5),
            cache_key_tag=cache_key_tag,
            **COMMON_RUN_KWARGS,
        )
        runtime_s = time.time() - start
        icp_err = m.transformation_error(icp_result.transformation, m.SIMULATED_TRANSFORM)
        coarse_err = m.transformation_error(coarse_result.transformation, m.SIMULATED_TRANSFORM)
        row.update({
            "status": "ok",
            "rot_error_deg": icp_err["rotation_error_deg"],
            "trans_error_mm": icp_err["translation_error"],
            "coarse_rot_error_deg": coarse_err["rotation_error_deg"],
            "coarse_trans_error_mm": coarse_err["translation_error"],
            "fitness": decision["fitness"],
            "inlier_rmse": decision["inlier_rmse"],
            "asymmetric_fraction": decision["asymmetric_fraction"],
            "target_coverage": decision["target_coverage_fraction"],
            "accepted": decision["accepted"],
            "reject_reasons": "; ".join(decision["reasons"]),
            "runtime_s": runtime_s,
            "error_message": "",
        })
    except Exception as exc:   # noqa: BLE001 - namerno ujamemo vse, glej docstring
        runtime_s = time.time() - start
        row.update({
            "status": "error",
            "rot_error_deg": np.nan, "trans_error_mm": np.nan,
            "coarse_rot_error_deg": np.nan, "coarse_trans_error_mm": np.nan,
            "fitness": np.nan, "inlier_rmse": np.nan, "asymmetric_fraction": np.nan,
            "target_coverage": np.nan, "accepted": False, "reject_reasons": "",
            "runtime_s": runtime_s, "error_message": f"{type(exc).__name__}: {exc}",
        })
        print(f"    NAPAKA: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    return row


def main(smoke_test: bool = False) -> None:
    print("=" * 70)
    print(f"camera_benchmark.py {'(SMOKE TEST)' if smoke_test else '(FULL SWEEP)'}")
    print("=" * 70)

    apply_spec_overrides()

    cad_path = BASE_PATH / CAD_FILE
    reference_point, part_radius_mm = compute_cad_reference_and_radius(cad_path)
    margin_mm = part_radius_mm + POSITION_MARGIN_EXTRA_MM
    print(f"CAD part XY radius (native frame): {part_radius_mm:.1f}mm, "
          f"margin used (radius + {POSITION_MARGIN_EXTRA_MM}mm): {margin_mm:.1f}mm")

    m.draw_registration_result = lambda *args, **kwargs: None   # brez vizualizacijskih oken za ves sweep

    cameras = list(kamere.CAMERAS.keys())
    yaw_list = list(YAW_ANGLES_DEG)
    seeds = [SEED_BASE + i for i in range(N_SEEDS)]

    if smoke_test:
        cameras = cameras[:1]
        yaw_list = yaw_list[:1]
        seeds = seeds[:2]

    existing_rows, done_set = load_existing_runs()
    all_rows = list(existing_rows)

    table_half_extent_mm = COMMON_RUN_KWARGS.get("table_size_x", 500.0) / 2.0   # privzetek run_registration, glej fizični prior v yaw_sweep_registration

    plan = []
    config_rows = []
    for camera_name in cameras:
        spec = kamere.CAMERAS[camera_name]
        camera_params = kp.resolve_camera_params(camera_name, resolution_scale=RESOLUTION_SCALE)
        estimated_fields = ",".join(sorted(SPEC_OVERRIDES.get(camera_name, {}).keys()))
        positions = compute_positions(spec["scanning_area_mm"], margin_mm)
        if smoke_test:
            positions = positions[:1]   # samo "center"

        for _, tx, ty in positions:
            if abs(tx) >= table_half_extent_mm or abs(ty) >= table_half_extent_mm:
                raise ValueError(
                    f"{camera_name}: pozicija ({tx:.1f},{ty:.1f}) presega privzeto mizo "
                    f"({table_half_extent_mm:.0f}mm polovična stranica) - yaw_sweep_registration "
                    f"bo zavrnil VSE kandidate (fizični prior). Zmanjšaj POSITION_MARGIN_EXTRA_MM "
                    f"ali povečaj table_size_x/y v COMMON_RUN_KWARGS.")

        config_rows.append({
            "camera": camera_name,
            "working_distance_mm": spec["working_distance_mm"],
            "camera_fov_deg": camera_params.get("camera_fov_deg"),
            "camera_width_px": camera_params.get("camera_width_px"),
            "camera_height_px": camera_params.get("camera_height_px"),
            "depth_noise_at_1m": camera_params.get("depth_noise_at_1m"),
            "noise_reference_distance_m": camera_params.get("noise_reference_distance_m"),
            "distance_bias_permille": camera_params.get("distance_bias_permille"),
            "global_planarity_mm": camera_params.get("global_planarity_mm"),
            "noise_spatial_correlation_px": camera_params.get("noise_spatial_correlation_px"),
            "estimated_fields": estimated_fields,
            "resolution_scale": RESOLUTION_SCALE,
        })

        for position_label, tx, ty in positions:
            for yaw_deg in yaw_list:
                for seed in seeds:
                    plan.append((camera_name, position_label, tx, ty, yaw_deg, seed,
                               camera_params, estimated_fields))

    config_rows.append({
        "camera": "(global settings)",
        "resolution_scale": RESOLUTION_SCALE,
        "n_seeds": N_SEEDS,
        "seed_base": SEED_BASE,
        "yaw_angles_deg": str(YAW_ANGLES_DEG),
        "position_margin_extra_mm": POSITION_MARGIN_EXTRA_MM,
        "cad_part_radius_mm": part_radius_mm,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    total = len(plan)
    print(f"Planned {total} runs across {len(cameras)} camera(s), "
          f"{len(yaw_list)} yaw angle(s), {len(seeds)} seed(s)")

    completed_this_session = 0
    elapsed_times: list[float] = []
    for i, (camera_name, position_label, tx, ty, yaw_deg, seed, camera_params,
           estimated_fields) in enumerate(plan, start=1):
        key = (camera_name, round(tx, 3), round(ty, 3), round(yaw_deg, 3), seed)
        if key in done_set:
            continue

        print(f"[{i}/{total}] {camera_name} {position_label} yaw={yaw_deg} seed={seed} ...",
              end=" ", flush=True)
        cache_key_tag = f"_bench_{camera_name}_{position_label}_yaw{yaw_deg}_seed{seed}"
        row = run_single(camera_name, position_label, tx, ty, yaw_deg, seed, reference_point,
                         camera_params, cad_path, estimated_fields, cache_key_tag)
        all_rows.append(row)
        completed_this_session += 1
        elapsed_times.append(row["runtime_s"])

        if row["status"] == "ok":
            print(f"-> trans_err={row['trans_error_mm']:.2f}mm rot_err={row['rot_error_deg']:.2f}deg "
                  f"({row['runtime_s']:.1f}s)")
        else:
            print(f"-> ERROR ({row['runtime_s']:.1f}s)")

        avg_time = sum(elapsed_times) / len(elapsed_times)
        remaining = max(total - i, 0)
        eta_s = avg_time * remaining
        print(f"    progress {i}/{total} | avg {avg_time:.1f}s/run | "
              f"ETA {eta_s / 3600:.1f}h ({eta_s / 60:.0f} min)")

        save_excel(all_rows, config_rows)

    print("=" * 70)
    print(f"{'Smoke test' if smoke_test else 'Sweep'} done - "
          f"{completed_this_session} run(s) executed this session, "
          f"{len(all_rows)} total row(s) in {OUTPUT_FILE.name}")
    print("=" * 70)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
