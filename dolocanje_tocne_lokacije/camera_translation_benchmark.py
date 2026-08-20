"""camera_translation_benchmark.py - kako KONSISTENTNA (ponovljiva) je
registracija posamezne kamere, ko se del premika navzven (stran od
kamerine optične osi v (0,0,0)), vse do roba kamerinega LISTED (dokumentiranega
specifikacijskega, kamere.py-jevega scanning_area_mm) vidnega polja.

Sestrska skripta camera_benchmark.py testira natančnost na peščici FIKSNIH
pozicij znotraj kamerinega scanning_area_mm; ta skripta namesto tega dela en
sam, gost, NAVZVEN rastoč sweep vzdolž ene same smeri, dokler pozicija ne bi
presegla dokumentirano vidno polje kamere - odgovarja na "kako hitro/
enakomerno se natančnost degradira, ko se del oddaljuje od centra", ne "kako
natančna je registracija na nekaj reprezentativnih pozicijah".

EKSPERIMENTALNI NAČRT (glej konfiguracijske konstante spodaj):
  Za vsako kamero v CAMERAS_TO_TEST (VSE iz kamere.CAMERAS RAZEN
  "mecheye_pro_s" - eksplicitno preskočena po uporabnikovi zahtevi):
    - Del začne NATANKO POD kamero (distance_mm=0, torej translation_x=y=0),
      nato se premika navzven vzdolž ENE SAME, fiksne smeri
      (OUTWARD_DIRECTION_DEG, privzeto 0 stopinj = vzdolž +X osi) v korakih
      po STEP_MM (privzeto 10mm): distance_mm = 0, 10, 20, 30, ...
    - Na VSAKI poziciji (distance_mm) se izvede N_ITERATIONS_PER_POSITION
      (privzeto 100) neodvisnih simuliranih skenov - vsak z DRUGAČNIM yaw
      kotom (enakomerno razporejenih čez 0-360 stopinj, torej i * 360/100 =
      i * 3.6 stopinj) IN drugačnim naključnim semenom šuma (SEED_BASE+i) -
      torej resnično 100 RAZLIČNIH skenov na pozicijo, ne 100 ponovitev
      istega vhoda.
    - Po vseh 100 ponovitvah se pozicija premakne za STEP_MM navzven in
      postopek se ponovi.
    - Sweep za posamezno kamero se USTAVI, ko bi naslednja pozicija
      (distance_mm + del-radij + varnostni rob) presegla kamerino LISTED
      scanning_area_mm - glej compute_max_outward_distance_mm(). To je
      UMETNA (analitična, iz specifikacije izračunana), ne empirično
      zaznana meja - namenoma: add_table_background=True (glej
      COMMON_RUN_KWARGS) je tu VKLJUČEN, torej ray casting v resnici
      pogosto zadene mizo tudi, ko je del že zunaj kamerinega
      dokumentiranega vidnega polja (fizično bi kamera v resnici še vedno
      nekaj videla - samo ne dela), zato dejanski ray-miss ni zanesljiv
      signal za to skripto (za razliko od prejšnje različice te datoteke,
      ki je add_table_background namenoma izklopila iz istega razloga v
      obratni smeri). Meja je torej "kamera po specifikaciji dela na tej
      razdalji ne bi smela videti", ne "ray casting je dejansko nehal
      zadevati kaj koli".

  "Konsistentnost" pomeni STD (ne le mean/median) napake translacije/
  rotacije med teh 100 ponovitev na vsaki (kamera, distance_mm) kombinaciji
  - glej "summary" list v Excel izhodu (build_summary spodaj).

PONOVNA UPORABA IZ camera_benchmark.py (uvoženo kot modul, NE podvojeno):
  compute_cad_reference_and_radius, build_ground_truth_transform - ista
  logika za referenčno CAD točko/pozo, ki jo main_trial.main() sam uporablja
  za --translation_x/y/z/--rotation_z_deg.

RESUME: kot camera_benchmark.py - obstoječi Excel se prebere, že opravljene
(camera, distance_mm, iteration) kombinacije se preskočijo. Meja (število
pozicij) se vsakič izračuna analitično iz kamere.py, torej je enaka na vsak
zagon - ni odvisna od tega, kaj je bilo že opravljeno.

OPOZORILO O ČASU IZVAJANJA: pri privzetih nastavitvah je to ZNATNO več
zagonov kot camera_benchmark.py (za vsako kamero grobo
(kamerin_scanning_area_polovica_mm / STEP_MM) pozicij x 100 ponovitev, npr.
~15-25 pozicij x 100 = 1500-2500 zagonov na kamero). Za hitro sanity-check
pred polnim zagonom uporabite `python camera_translation_benchmark.py
--smoke-test` (1 kamera, 3 pozicije, 5 ponovitev na pozicijo). Excel se
periodično shranjuje (glej SAVE_EVERY_N_RUNS) - prekinjen zagon se lahko
varno nadaljuje s ponovnim zagonom iste skripte.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import kamere
import kontrolna_plosca as kp
import main_trial as m
import camera_benchmark as cb

BASE_PATH = Path(__file__).resolve().parent
CAD_FILE = "eHDS S Housing + CC s pottingom, fine.STL"
OUTPUT_FILE = BASE_PATH / "camera_translation_benchmark.xlsx"


# =============================================================================
# KONFIGURACIJA EKSPERIMENTA
# =============================================================================
RESOLUTION_SCALE = 0.25
N_ITERATIONS_PER_POSITION = 100
STEP_MM = 10.0
OUTWARD_DIRECTION_DEG = 0.0   # 0 = vzdolž +X osi - spremenite za drugo smer (npr. 45.0 za diagonalo NE)
SEED_BASE = 5000
POSITION_MARGIN_EXTRA_MM = 10.0   # dodaten varnostni rob čez CAD-ov lastni XY polmer (glej compute_max_outward_distance_mm) - ista logika/privzetek kot camera_benchmark.py
MAX_POSITIONS_SAFETY_CAP = 80   # varovalka proti neskončni zanki (80*10mm=800mm) - nobena testirana kamera naj tega ne doseže; če se to zgodi, WARNING izpiše, da preveriš nastavitve
SAVE_EVERY_N_RUNS = 20   # Excel prepis je O(trenutno_število_vrstic) - pri tisočih zagonih bi shranjevanje po vsakem posamičnem zagonu (kot camera_benchmark.py, ki ima ~10x manj vrstic) postalo prevladujoč strošek; namesto tega shranimo vsakih SAVE_EVERY_N_RUNS zagonov + vedno ob koncu vsake kamere/skripte

TOP_FRACTION = 0.35
UP_AXIS = 2
FLIP_UP_DIRECTION = False

CAMERAS_TO_TEST = [name for name in kamere.CAMERAS if name != "mecheye_pro_s"]   # Mech-Eye eksplicitno preskočena (uporabnikova zahteva)

# Parametri, enaki za VSE zagone te skripte (glej main_trial.py --help za
# razlago vsakega) - vse, kar je kamera-specifično (FOV, ločljivost, šum
# ...), se doda posebej v run_single() prek kontrolna_plosca.resolve_camera_params().
COMMON_RUN_KWARGS = dict(
    voxel_size=1.0,
    sample_point_count=200000,
    add_table_background=True,        # VKLJUČENO (glej modulski docstring) - miza pod delom v ray-casting sceni, kot pri camera_benchmark.py
    remove_table_background_flag=True,   # odstrani mizne/ozadje točke iz target-a pred registracijo, glej main_trial.py --help
    use_cad_cache=True,           # isti CAD za vse zagone - vzorčenje se izračuna samo enkrat
    use_preprocess_cache=False,   # vsak zagon je unikaten (nova pozicija/yaw/seed) - predpomnjenje bi samo kopičilo tisoče datotek brez koristi
    top_fraction=TOP_FRACTION,
    up_axis=UP_AXIS,
    flip_up_direction=FLIP_UP_DIRECTION,
    table_size_x=1000.0,   # fizični prior za yaw_sweep_registration IN velikost mizne mreže v ray-casting sceni - namenoma velik, da ne umetno omeji dosega pred kamerino LISTED FOV mejo (glej compute_max_outward_distance_mm)
    table_size_y=1000.0,
)

RUNS_COLUMNS = [
    "camera", "distance_mm", "direction_deg", "iteration", "tx_mm", "ty_mm", "yaw_deg", "seed",
    "status", "rot_error_deg", "trans_error_mm", "coarse_rot_error_deg", "coarse_trans_error_mm",
    "fitness", "inlier_rmse", "asymmetric_fraction", "target_coverage",
    "accepted", "reject_reasons", "runtime_s", "error_message",
]


def compute_max_outward_distance_mm(scanning_area_mm: dict, direction_deg: float, margin_mm: float) -> float:
    """Analitično izračuna največjo razdaljo (mm) vzdolž direction_deg, pri
    kateri del (zmanjšan za margin_mm - CAD-ov lastni XY polmer + varnostni
    rob, glej POSITION_MARGIN_EXTRA_MM) ŠE OSTANE znotraj kamerinega LISTED
    (specifikacijskega) pravokotnega scanning_area_mm, centriranega na
    (0,0). Za pravokotno vidno polje [-W/2,W/2] x [-H/2,H/2] in smer
    (cos,sin) je omejujoča tista os, ki jo pri danem kotu prva doseže -
    d_max = min(W/2/|cos|, H/2/|sin|) (os z |komponenta|~0 je izpuščena, ker
    je za njo meja neskončna)."""
    half_w = scanning_area_mm["width"] / 2.0
    half_h = scanning_area_mm["height"] / 2.0
    theta = np.radians(direction_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    candidates = []
    if abs(cos_t) > 1e-9:
        candidates.append(half_w / abs(cos_t))
    if abs(sin_t) > 1e-9:
        candidates.append(half_h / abs(sin_t))
    boundary_mm = min(candidates) if candidates else float("inf")   # candidates prazen samo, če je direction_deg degenerirana (ne more se zgoditi za realen kot)
    return max(boundary_mm - margin_mm, 0.0)


def load_existing_runs() -> tuple[list[dict], set]:
    """Prebere obstoječi Excel (če obstaja) za podporo nadaljevanju - vrne
    (seznam že opravljenih vrstic, množica (camera,distance_mm,iteration)
    ključev, ki jih je treba PRESKOČITI)."""
    if not OUTPUT_FILE.exists():
        return [], set()
    existing = pd.read_excel(OUTPUT_FILE, sheet_name="runs")
    done = set()
    for row in existing.itertuples(index=False):
        done.add((row.camera, round(float(row.distance_mm), 3), int(row.iteration)))
    print(f"Najdena obstoječa datoteka z {len(existing)} vrsticami - te kombinacije bodo preskočene.")
    return existing.to_dict("records"), done


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Sestavi povzetek po (kamera x distance_mm) - štetje, štetje napak,
    mean/STD/median/p95/max napak translacije/rotacije, mean pokritosti,
    delež sprejetih. STD je tu GLAVNA metrika ("konsistentnost" = nizka
    variabilnost med 100 ponovitvami na isti poziciji), ne le dodatek."""
    def stats(group: pd.DataFrame) -> pd.Series:
        ok = group[group["status"] == "ok"]
        return pd.Series({
            "count": len(group),
            "error_count": int((group["status"] == "error").sum()),
            "trans_error_mm_mean": ok["trans_error_mm"].mean(),
            "trans_error_mm_std": ok["trans_error_mm"].std(),
            "trans_error_mm_median": ok["trans_error_mm"].median(),
            "trans_error_mm_p95": ok["trans_error_mm"].quantile(0.95) if len(ok) else np.nan,
            "trans_error_mm_max": ok["trans_error_mm"].max(),
            "rot_error_deg_mean": ok["rot_error_deg"].mean(),
            "rot_error_deg_std": ok["rot_error_deg"].std(),
            "rot_error_deg_median": ok["rot_error_deg"].median(),
            "rot_error_deg_p95": ok["rot_error_deg"].quantile(0.95) if len(ok) else np.nan,
            "rot_error_deg_max": ok["rot_error_deg"].max(),
            "target_coverage_mean": ok["target_coverage"].mean(),
            "accept_rate": ok["accepted"].mean() if len(ok) else np.nan,
        })

    return df.groupby(["camera", "distance_mm"], group_keys=False)[df.columns].apply(stats).reset_index()


def save_excel(runs_rows: list[dict], config_rows: list[dict]) -> None:
    """Prepiše celoten Excel (runs/summary/config) - glej SAVE_EVERY_N_RUNS
    za razlog, zakaj se to NE kliče po vsakem posameznem zagonu (za razliko
    od camera_benchmark.py, glej modulski docstring)."""
    runs_df = pd.DataFrame(runs_rows, columns=RUNS_COLUMNS)
    summary_df = build_summary(runs_df) if len(runs_df) else pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        runs_df.to_excel(writer, sheet_name="runs", index=False)

        summary_df.to_excel(writer, sheet_name="summary", index=False, startrow=0)
        note_row = len(summary_df) + 2
        pd.DataFrame([[
            "OPOMBA: 'konsistentnost' berite prek trans_error_mm_std/rot_error_deg_std "
            "(variabilnost med 100 ponovitvami na isti distance_mm) - accept_rate uporablja "
            "NEUMERJENE placeholder pragove (glej main_trial.decide_registration_outcome)."]]).to_excel(
            writer, sheet_name="summary", index=False, header=False,
            startrow=note_row, startcol=0)

        pd.DataFrame(config_rows).to_excel(writer, sheet_name="config", index=False)


def run_single(camera_name: str, distance_mm: float, direction_deg: float, iteration: int,
              tx: float, ty: float, yaw_deg: float, seed: int, reference_point: np.ndarray,
              camera_params: dict, cad_path: Path, cache_key_tag: str) -> dict:
    """Požene EN zagon registracije in vrne eno vrstico rezultatov (slovar,
    ustreza RUNS_COLUMNS). Nikoli ne vrže naprej - napake ujame in zabeleži
    kot status='error', da ena neuspešna kombinacija ne prekine sweep-a (za
    razliko od prejšnje različice te datoteke, tu status NE določa, kdaj se
    sweep ustavi - meja je vnaprej izračunana, glej compute_max_outward_distance_mm)."""
    row: dict = {
        "camera": camera_name, "distance_mm": distance_mm, "direction_deg": direction_deg,
        "iteration": iteration, "tx_mm": tx, "ty_mm": ty, "yaw_deg": yaw_deg, "seed": seed,
    }

    translation_z = camera_params["translation_z"]
    m.SIMULATED_TRANSFORM = cb.build_ground_truth_transform(reference_point, tx, ty, translation_z, yaw_deg)

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
    return row


def main(smoke_test: bool = False) -> None:
    print("=" * 70)
    print(f"camera_translation_benchmark.py {'(SMOKE TEST)' if smoke_test else '(FULL SWEEP)'}")
    print("=" * 70)

    cad_path = BASE_PATH / CAD_FILE
    reference_point, part_radius_mm = cb.compute_cad_reference_and_radius(cad_path)
    margin_mm = part_radius_mm + POSITION_MARGIN_EXTRA_MM
    print(f"CAD part XY radius (native frame): {part_radius_mm:.1f}mm, "
          f"margin used (radius + {POSITION_MARGIN_EXTRA_MM}mm): {margin_mm:.1f}mm")

    m.draw_registration_result = lambda *args, **kwargs: None   # brez vizualizacijskih oken za ves sweep

    n_iterations = N_ITERATIONS_PER_POSITION
    max_positions_cap = MAX_POSITIONS_SAFETY_CAP
    cameras = list(CAMERAS_TO_TEST)
    if smoke_test:
        cameras = cameras[:1]
        n_iterations = 5
        max_positions_cap = 3

    print(f"Testing cameras: {cameras} (skipped: mecheye_pro_s, per user request)")
    print(f"Direction: {OUTWARD_DIRECTION_DEG} deg, step: {STEP_MM}mm, "
          f"{n_iterations} iterations/position, resolution_scale={RESOLUTION_SCALE}")

    existing_rows, done_set = load_existing_runs()
    all_rows = list(existing_rows)

    yaw_angles = [i * (360.0 / n_iterations) for i in range(n_iterations)]   # enakomerno razporejeni čez 0-360 stopinj
    seeds = [SEED_BASE + i for i in range(n_iterations)]
    direction_rad = np.radians(OUTWARD_DIRECTION_DEG)
    dir_cos, dir_sin = np.cos(direction_rad), np.sin(direction_rad)

    config_rows = [{
        "camera": "(global settings)",
        "resolution_scale": RESOLUTION_SCALE,
        "n_iterations_per_position": n_iterations,
        "step_mm": STEP_MM,
        "outward_direction_deg": OUTWARD_DIRECTION_DEG,
        "seed_base": SEED_BASE,
        "cad_part_radius_mm": part_radius_mm,
        "position_margin_mm": margin_mm,
        "cameras_tested": ",".join(cameras),
        "cameras_skipped": "mecheye_pro_s",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }]

    completed_this_session = 0
    unsaved_since_last_write = 0
    elapsed_times: list[float] = []

    def maybe_save(force: bool = False) -> None:
        nonlocal unsaved_since_last_write
        if force or unsaved_since_last_write >= SAVE_EVERY_N_RUNS:
            save_excel(all_rows, config_rows)
            unsaved_since_last_write = 0

    for camera_name in cameras:
        spec = kamere.CAMERAS[camera_name]
        camera_params = kp.resolve_camera_params(camera_name, resolution_scale=RESOLUTION_SCALE)

        max_distance_mm = compute_max_outward_distance_mm(spec["scanning_area_mm"], OUTWARD_DIRECTION_DEG, margin_mm)
        n_positions = int(max_distance_mm // STEP_MM) + 1   # +1 za distance_mm=0 (center)
        n_positions = min(n_positions, max_positions_cap)
        if n_positions >= max_positions_cap:
            print(f"  WARNING: {camera_name}: izračunano število pozicij doseže "
                  f"MAX_POSITIONS_SAFETY_CAP ({max_positions_cap}) - preveri scanning_area_mm/STEP_MM")

        print(f"\n--- {camera_name} --- "
              f"scanning_area_mm={spec['scanning_area_mm']}, max_distance_mm={max_distance_mm:.1f}, "
              f"{n_positions} position(s) planned ({n_positions * n_iterations} runs)")

        for position_index in range(n_positions):
            distance_mm = position_index * STEP_MM
            tx = distance_mm * dir_cos
            ty = distance_mm * dir_sin

            for iteration in range(n_iterations):
                key = (camera_name, round(distance_mm, 3), iteration)
                if key in done_set:
                    continue
                cache_key_tag = f"_trbench_{camera_name}_d{distance_mm:.0f}_it{iteration}_seed{seeds[iteration]}"
                row = run_single(camera_name, distance_mm, OUTWARD_DIRECTION_DEG, iteration, tx, ty,
                                 yaw_angles[iteration], seeds[iteration], reference_point, camera_params,
                                 cad_path, cache_key_tag)
                all_rows.append(row)
                completed_this_session += 1
                unsaved_since_last_write += 1
                elapsed_times.append(row["runtime_s"])
                done_set.add(key)
                maybe_save()

                if iteration % 20 == 0 or iteration == n_iterations - 1:
                    avg_time = sum(elapsed_times[-50:]) / len(elapsed_times[-50:])
                    print(f"    [{camera_name}] distance={distance_mm:.0f}mm iteration={iteration + 1}/{n_iterations} "
                          f"| avg {avg_time:.1f}s/run (last 50) | {completed_this_session} run(s) this session")

        maybe_save(force=True)   # vedno shrani ob koncu kamere, ne glede na SAVE_EVERY_N_RUNS
        config_rows.append({
            "camera": camera_name,
            "scanning_area_mm": str(spec["scanning_area_mm"]),
            "max_distance_mm": max_distance_mm,
            "n_positions": n_positions,
        })

    maybe_save(force=True)

    print("=" * 70)
    print(f"{'Smoke test' if smoke_test else 'Sweep'} done - "
          f"{completed_this_session} run(s) executed this session, "
          f"{len(all_rows)} total row(s) in {OUTPUT_FILE.name}")
    print("=" * 70)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
