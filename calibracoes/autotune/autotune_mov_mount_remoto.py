import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import json_output_path
from controle.mov_mount_remoto import (
    DEFAULT_BASE_URL,
    MAX_TEMPO_MOV,
    SETTLE_AFTER_STOP_S,
    TOLERANCIA_GRAUS,
    TelescopeClient,
    calc_error_az,
)


RESULTS_JSON = json_output_path("autotune_mov_mount_remoto_resultados.json")

TEST_TIMEOUT_S = 45.0
TEST_REPEATS = 1
CONTROL_PERIOD_S = 0.45
VEL_MAX_AZ = 0.70
VEL_MAX_ALT = 0.55
VEL_MIN = 0.010
CMD_ZERO_SNAP = 0.004

KP_AZ_CANDIDATES = [0.50, 0.65, 0.80]
KP_ALT_CANDIDATES = [0.25, 0.35, 0.45, 0.55]
KD_AZ_CANDIDATES = [0.00, 0.02, 0.04]
KD_ALT_CANDIDATES = [0.00, 0.01, 0.02]

TEST_MOVES = [
    ("Az +0.20", 0.20, 0.0),
    ("Az -0.20", -0.20, 0.0),
    ("Alt +0.15", 0.0, 0.15),
    ("Alt -0.15", 0.0, -0.15),
]

FAILURE_PENALTY = 120.0
OVERSHOOT_PENALTY = 8.0
FINAL_ERROR_WEIGHT = 80.0


@dataclass(frozen=True)
class RemotePIDParams:
    kp_az: float
    kp_alt: float
    kd_az: float
    kd_alt: float


@dataclass
class TrialResult:
    move_name: str
    success: bool
    elapsed_s: float
    final_err_az_deg: float
    final_err_alt_deg: float
    overshoots_az: int
    overshoots_alt: int
    reason: str


@dataclass
class CandidateResult:
    params: RemotePIDParams
    score: float
    success_count: int
    total_tests: int
    median_time_s: float | None
    mean_overshoots: float
    trials: list[TrialResult]


def apply_min_velocity(cmd: float) -> float:
    if abs(cmd) < CMD_ZERO_SNAP:
        return 0.0
    if 0.0 < abs(cmd) < VEL_MIN:
        return float(VEL_MIN * np.sign(cmd))
    return float(cmd)


def axis_command(error: float, prev_error: float | None, dt: float, kp: float, kd: float, vmax: float) -> float:
    derivative = 0.0 if prev_error is None or dt <= 0 else (error - prev_error) / dt
    cmd = (kp * error) + (kd * derivative)
    cmd = float(np.clip(cmd, -vmax, vmax))
    return apply_min_velocity(cmd)


def _stop_axes(telescope: TelescopeClient) -> None:
    telescope.stop()
    time.sleep(SETTLE_AFTER_STOP_S)


def run_trial(
    telescope: TelescopeClient,
    params: RemotePIDParams,
    move_name: str,
    delta_az: float,
    delta_alt: float,
) -> TrialResult:
    az0, alt0 = telescope.read_altaz()
    target_az = (az0 + delta_az) % 360.0
    target_alt = float(np.clip(alt0 + delta_alt, -90.0, 90.0))

    prev_err_az = None
    prev_err_alt = None
    last_err_az = None
    last_err_alt = None
    overshoots_az = 0
    overshoots_alt = 0
    reason = "timeout"
    success = False
    t0 = time.perf_counter()
    last_t = t0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - t0
            dt = max(now - last_t, 1e-3)
            last_t = now

            az, alt = telescope.read_altaz()
            err_az = calc_error_az(target_az, az)
            err_alt = target_alt - alt

            if abs(err_az) <= TOLERANCIA_GRAUS and abs(err_alt) <= TOLERANCIA_GRAUS:
                reason = "stabilized"
                success = True
                break
            if elapsed >= min(TEST_TIMEOUT_S, MAX_TEMPO_MOV):
                reason = "timeout"
                break

            if last_err_az is not None and err_az * last_err_az < 0:
                overshoots_az += 1
            if last_err_alt is not None and err_alt * last_err_alt < 0:
                overshoots_alt += 1

            cmd_az = 0.0 if abs(err_az) <= TOLERANCIA_GRAUS else axis_command(
                err_az,
                prev_err_az,
                dt,
                params.kp_az,
                params.kd_az,
                VEL_MAX_AZ,
            )
            cmd_alt = 0.0 if abs(err_alt) <= TOLERANCIA_GRAUS else axis_command(
                err_alt,
                prev_err_alt,
                dt,
                params.kp_alt,
                params.kd_alt,
                VEL_MAX_ALT,
            )

            telescope.move_axis(0, cmd_az)
            telescope.move_axis(1, cmd_alt)

            prev_err_az = err_az
            prev_err_alt = err_alt
            last_err_az = err_az
            last_err_alt = err_alt
            time.sleep(CONTROL_PERIOD_S)
    finally:
        _stop_axes(telescope)

    azf, altf = telescope.read_altaz()
    final_err_az = calc_error_az(target_az, azf)
    final_err_alt = target_alt - altf
    elapsed = time.perf_counter() - t0
    return TrialResult(
        move_name=move_name,
        success=success,
        elapsed_s=float(elapsed),
        final_err_az_deg=float(final_err_az),
        final_err_alt_deg=float(final_err_alt),
        overshoots_az=overshoots_az,
        overshoots_alt=overshoots_alt,
        reason=reason,
    )


def score_trials(trials: list[TrialResult], total_tests: int) -> tuple[float, int, float | None, float]:
    success_count = sum(1 for trial in trials if trial.success)
    success_times = [trial.elapsed_s for trial in trials if trial.success]
    median_time = float(np.median(success_times)) if success_times else None
    overshoots = [trial.overshoots_az + trial.overshoots_alt for trial in trials]
    mean_overshoots = float(np.mean(overshoots)) if overshoots else 0.0

    score = 0.0
    for trial in trials:
        final_error = math.hypot(trial.final_err_az_deg, trial.final_err_alt_deg)
        score += FINAL_ERROR_WEIGHT * final_error
        score += OVERSHOOT_PENALTY * (trial.overshoots_az + trial.overshoots_alt)
        if trial.success:
            score += trial.elapsed_s
        else:
            score += FAILURE_PENALTY + trial.elapsed_s
    score += FAILURE_PENALTY * max(total_tests - len(trials), 0)
    return score, success_count, median_time, mean_overshoots


def evaluate_candidate(telescope: TelescopeClient, params: RemotePIDParams) -> CandidateResult:
    print(
        "\nCandidato "
        f"KpAz={params.kp_az:.3f} KpAlt={params.kp_alt:.3f} "
        f"KdAz={params.kd_az:.3f} KdAlt={params.kd_alt:.3f}"
    )
    trials: list[TrialResult] = []
    total_tests = len(TEST_MOVES) * TEST_REPEATS

    for repeat_idx in range(1, TEST_REPEATS + 1):
        for move_name, delta_az, delta_alt in TEST_MOVES:
            print(f"  {move_name} rep {repeat_idx}/{TEST_REPEATS}")
            trial = run_trial(telescope, params, move_name, delta_az, delta_alt)
            trials.append(trial)
            err = math.hypot(trial.final_err_az_deg, trial.final_err_alt_deg)
            status = "OK" if trial.success else "FALHA"
            print(
                f"    {status} | t={trial.elapsed_s:.2f}s | "
                f"err={err:.5f}deg | overs={trial.overshoots_az + trial.overshoots_alt} | "
                f"{trial.reason}"
            )
            if not trial.success:
                break
        if trials and not trials[-1].success:
            break

    score, success_count, median_time, mean_overshoots = score_trials(trials, total_tests)
    print(
        f"  Score={score:.3f} | sucessos={success_count}/{total_tests} | "
        f"mediana={'n/a' if median_time is None else f'{median_time:.2f}s'} | "
        f"overshoot_medio={mean_overshoots:.2f}"
    )
    return CandidateResult(
        params=params,
        score=score,
        success_count=success_count,
        total_tests=total_tests,
        median_time_s=median_time,
        mean_overshoots=mean_overshoots,
        trials=trials,
    )


def candidate_grid() -> list[RemotePIDParams]:
    candidates = []
    for kp_az in KP_AZ_CANDIDATES:
        for kp_alt in KP_ALT_CANDIDATES:
            for kd_az in KD_AZ_CANDIDATES:
                for kd_alt in KD_ALT_CANDIDATES:
                    candidates.append(RemotePIDParams(kp_az, kp_alt, kd_az, kd_alt))
    return candidates


def save_results(results: list[CandidateResult]) -> None:
    ranking = sorted(
        results,
        key=lambda item: (
            -item.success_count,
            item.score,
            item.median_time_s if item.median_time_s is not None else float("inf"),
        ),
    )
    payload = {
        "timestamp_epoch": time.time(),
        "config": {
            "base_url": DEFAULT_BASE_URL,
            "test_timeout_s": TEST_TIMEOUT_S,
            "test_repeats": TEST_REPEATS,
            "control_period_s": CONTROL_PERIOD_S,
            "test_moves": TEST_MOVES,
        },
        "ranking": [
            {
                "params": asdict(item.params),
                "score": item.score,
                "success_count": item.success_count,
                "total_tests": item.total_tests,
                "median_time_s": item.median_time_s,
                "mean_overshoots": item.mean_overshoots,
                "trials": [asdict(trial) for trial in item.trials],
            }
            for item in ranking
        ],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Podio autotune remoto ===")
    for idx, item in enumerate(ranking[:8], start=1):
        median = "n/a" if item.median_time_s is None else f"{item.median_time_s:.2f}s"
        print(
            f"{idx}. KpAz={item.params.kp_az:.3f} KpAlt={item.params.kp_alt:.3f} "
            f"KdAz={item.params.kd_az:.3f} KdAlt={item.params.kd_alt:.3f} | "
            f"score={item.score:.3f} | sucessos={item.success_count}/{item.total_tests} | "
            f"mediana={median} | overs={item.mean_overshoots:.2f}"
        )
    print(f"\nResultados salvos em: {RESULTS_JSON}")


def main() -> None:
    print("=== Autotune PID para mount remoto ===")
    print("Use com o mount livre para movimentos pequenos. Ctrl+C para parar.")
    base_url = input(f"URL do mount remoto [{DEFAULT_BASE_URL}]: ").strip() or DEFAULT_BASE_URL
    max_candidates = input("Max candidatos para testar [12]: ").strip()
    max_candidates = 12 if not max_candidates else int(max_candidates)

    telescope = TelescopeClient(base_url)
    telescope.ensure_ready()
    az, alt = telescope.read_altaz()
    print(f"Pos inicial: Az={az:.6f} deg | Alt={alt:.6f} deg")

    seeds = [
        RemotePIDParams(0.65, 0.40, 0.02, 0.00),
        RemotePIDParams(0.65, 0.35, 0.02, 0.00),
        RemotePIDParams(0.80, 0.35, 0.02, 0.00),
        RemotePIDParams(0.50, 0.35, 0.00, 0.00),
        RemotePIDParams(0.65, 0.45, 0.00, 0.00),
    ]
    candidates = seeds + [candidate for candidate in candidate_grid() if candidate not in seeds]

    results: list[CandidateResult] = []
    try:
        for params in candidates[:max_candidates]:
            results.append(evaluate_candidate(telescope, params))
    except KeyboardInterrupt:
        print("\nAutotune interrompido pelo usuario.")
    finally:
        telescope.stop()

    if results:
        save_results(results)


if __name__ == "__main__":
    main()
