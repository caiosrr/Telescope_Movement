import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from artifact_paths import display_path, json_output_path
from mov_simultaneo import (
    MAX_CORRECOES,
    TOLERANCIA_GRAUS,
    VEL_MAX_LIMITE,
    VEL_MIN_LIMITE,
    PID,
    ensure_connected,
    ensure_not_tracking,
    ensure_unparked,
    move_axis,
    read_altaz,
    calc_error,
)

# ===== Configuracao da busca =====
SAFE_MAX_DELTA_DEG = 20.0
SAFE_AZ_MIN_DEG = 270.0
SAFE_AZ_MAX_DEG = 30.0
SAFE_ALT_MIN_DEG = -30
SAFE_ALT_MAX_DEG = 30
TEST_TIMEOUT_S = 25.0
TEST_REPEATS = 1
TWIDDLE_MAX_ITERS = 5
TWIDDLE_MIN_DP_SUM = 0.01
DP_SHRINK = 0.60
DP_GROW = 1.20
RESULTS_JSON = json_output_path("autotune_mov_simultaneo_resultados.json")

# Mantemos Ki fixo para reduzir risco e dimensionalidade.
KI_FIXED = 0.0001
INITIAL_KP = 1.19
INITIAL_KD = 0.08
INITIAL_DP_KP = 0.08
INITIAL_DP_KD = 0.015
KP_BOUNDS = (0.10, 2.00)
KD_BOUNDS = (0.00, 0.50)

FAILURE_PENALTY = 100.0
SKIPPED_TEST_PENALTY = 60.0
OVERSHOOT_PENALTY = 3.0
FIRST_OVERSHOOT_WEIGHT = 0.20
TIMEOUT_PENALTY_SCALE = 1.0


@dataclass(frozen=True)
class TunedParams:
    kp: float
    kd: float


@dataclass(frozen=True)
class TestMove:
    name: str
    delta_az_deg: float
    delta_alt_deg: float


@dataclass
class MoveTrialResult:
    move_name: str
    success: bool
    elapsed_s: float
    final_err_az_deg: float
    final_err_alt_deg: float
    overshoots_az: int
    overshoots_alt: int
    reason: str


@dataclass
class CandidateEvaluation:
    kp: float
    kd: float
    score: float
    success_count: int
    total_tests: int
    median_time_s: float | None
    mean_overshoots: float
    trials: list[MoveTrialResult]


TEST_MOVES = [
    TestMove("Az +5", +5.0, 0.0),
    TestMove("Az -5", -5.0, 0.0),
    TestMove("Alt +2", 0.0, +2.0),
    TestMove("Alt -2", 0.0, -2.0),
]


def _validate_test_moves():
    for move in TEST_MOVES:
        if abs(move.delta_az_deg) > SAFE_MAX_DELTA_DEG or abs(move.delta_alt_deg) > SAFE_MAX_DELTA_DEG:
            raise ValueError(
                f"Movimento de teste {move.name} excede o limite seguro de {SAFE_MAX_DELTA_DEG} deg."
            )


def _az_in_safe_window(az_deg: float) -> bool:
    az = az_deg % 360.0
    az_min = SAFE_AZ_MIN_DEG % 360.0
    az_max = SAFE_AZ_MAX_DEG % 360.0
    if az_min <= az_max:
        return az_min <= az <= az_max
    return az >= az_min or az <= az_max


def _clip_params(params: TunedParams) -> TunedParams:
    kp = float(np.clip(params.kp, KP_BOUNDS[0], KP_BOUNDS[1]))
    kd = float(np.clip(params.kd, KD_BOUNDS[0], KD_BOUNDS[1]))
    return TunedParams(kp=kp, kd=kd)


def _overshoot_cost(overshoots_total: int) -> float:
    if overshoots_total <= 0:
        return 0.0
    if overshoots_total == 1:
        return FIRST_OVERSHOOT_WEIGHT * OVERSHOOT_PENALTY
    return (FIRST_OVERSHOOT_WEIGHT * OVERSHOOT_PENALTY) + ((overshoots_total - 1) * OVERSHOOT_PENALTY)


def _run_move_trial(mount: bool, move: TestMove, params: TunedParams) -> MoveTrialResult:
    az0, alt0 = read_altaz()
    alvo_az = (az0 + move.delta_az_deg) % 360
    alvo_alt = alt0 + move.delta_alt_deg

    if not _az_in_safe_window(az0):
        raise RuntimeError(
            f"Posicao inicial em azimute ({az0:.4f} deg) esta fora da janela segura "
            f"[{SAFE_AZ_MIN_DEG}, {SAFE_AZ_MAX_DEG}] deg."
        )

    if not _az_in_safe_window(alvo_az):
        raise RuntimeError(
            f"Movimento {move.name} sairia da janela segura de azimute "
            f"[{SAFE_AZ_MIN_DEG}, {SAFE_AZ_MAX_DEG}] deg."
        )

    if alvo_alt > SAFE_ALT_MAX_DEG or alvo_alt < SAFE_ALT_MIN_DEG:
        raise RuntimeError(
            f"Movimento {move.name} sairia da faixa segura de altitude "
            f"[{SAFE_ALT_MIN_DEG}, {SAFE_ALT_MAX_DEG}] deg."
        )

    pid_az = PID(
        kp=params.kp,
        ki=KI_FIXED,
        kd=params.kd,
        setpoint=alvo_az,
        output_limits=(-VEL_MAX_LIMITE, VEL_MAX_LIMITE),
        integral_limit=5,
    )
    pid_alt = PID(
        kp=params.kp,
        ki=KI_FIXED,
        kd=params.kd,
        setpoint=alvo_alt,
        output_limits=(-VEL_MAX_LIMITE, VEL_MAX_LIMITE),
        integral_limit=5,
    )

    t0 = time.perf_counter()
    tempo_decorrido = 0.0
    error_last_az = None
    error_last_alt = None
    overshoots_az = 0
    overshoots_alt = 0
    reason = "timeout"
    success = False

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:
                az, alt = read_altaz()
                tempo_decorrido = time.perf_counter() - t0

                cmd_az, error_az = pid_az.update(0, az)
                cmd_alt, error_alt = pid_alt.update(1, alt)

                err_abs_az = abs(error_az)
                err_abs_alt = abs(error_alt)

                if tempo_decorrido > TEST_TIMEOUT_S:
                    reason = "timeout"
                    break

                az_ok = err_abs_az < TOLERANCIA_GRAUS
                alt_ok = err_abs_alt < TOLERANCIA_GRAUS
                if az_ok and alt_ok:
                    reason = "stabilized"
                    success = True
                    executor.submit(move_axis, 0, 0.0, mount).result()
                    executor.submit(move_axis, 1, 0.0, mount).result()
                    break

                if error_last_az is not None and error_az * error_last_az < 0:
                    overshoots_az += 1
                    if overshoots_az > MAX_CORRECOES:
                        reason = "too_many_overshoots_az"
                        cmd_az = 0.0
                        break
                    pid_az.reset()
                    cmd_az = 0.0

                if error_last_alt is not None and error_alt * error_last_alt < 0:
                    overshoots_alt += 1
                    if overshoots_alt > MAX_CORRECOES:
                        reason = "too_many_overshoots_alt"
                        cmd_alt = 0.0
                        break
                    pid_alt.reset()
                    cmd_alt = 0.0

                if az_ok:
                    cmd_az = 0.0
                elif 0 < abs(cmd_az) < VEL_MIN_LIMITE:
                    cmd_az = VEL_MIN_LIMITE * np.sign(cmd_az)

                if alt_ok:
                    cmd_alt = 0.0
                elif 0 < abs(cmd_alt) < VEL_MIN_LIMITE:
                    cmd_alt = VEL_MIN_LIMITE * np.sign(cmd_alt)

                future_az = executor.submit(move_axis, 0, cmd_az, mount)
                future_alt = executor.submit(move_axis, 1, cmd_alt, mount)
                future_az.result()
                future_alt.result()

                error_last_az = error_az
                error_last_alt = error_alt

                max_err = max(err_abs_az, err_abs_alt)
                if max_err > 1.0:
                    dt_sleep = 0.1
                elif max_err > 0.1:
                    dt_sleep = 0.05
                else:
                    dt_sleep = 0.01

                time.sleep(dt_sleep)
        finally:
            move_axis(0, 0.0, mount)
            move_axis(1, 0.0, mount)

    azf, altf = read_altaz()
    final_err_az = float(calc_error(0, alvo_az, azf))
    final_err_alt = float(calc_error(1, alvo_alt, altf))
    return MoveTrialResult(
        move_name=move.name,
        success=success,
        elapsed_s=tempo_decorrido,
        final_err_az_deg=final_err_az,
        final_err_alt_deg=final_err_alt,
        overshoots_az=overshoots_az,
        overshoots_alt=overshoots_alt,
        reason=reason,
    )


def _score_trials(trials: list[MoveTrialResult], total_expected: int) -> tuple[float, int, float | None, float]:
    success_count = sum(1 for t in trials if t.success)
    success_times = [t.elapsed_s for t in trials if t.success]
    median_time = float(np.median(success_times)) if success_times else None
    mean_overshoots = float(
        np.mean([t.overshoots_az + t.overshoots_alt for t in trials])
    ) if trials else 0.0

    score = 0.0
    for trial in trials:
        overshoot_total = trial.overshoots_az + trial.overshoots_alt
        overshoot_cost = _overshoot_cost(overshoot_total)
        if trial.success:
            score += trial.elapsed_s
            score += overshoot_cost
        else:
            score += FAILURE_PENALTY
            score += TIMEOUT_PENALTY_SCALE * trial.elapsed_s
            score += 2.0 * overshoot_cost

    skipped = max(total_expected - len(trials), 0)
    score += SKIPPED_TEST_PENALTY * skipped
    return score, success_count, median_time, mean_overshoots


def evaluate_candidate(mount: bool, params: TunedParams) -> CandidateEvaluation:
    params = _clip_params(params)
    print(f"\nAvaliando candidato: Kp={params.kp:.4f}, Kd={params.kd:.4f}")

    trials: list[MoveTrialResult] = []
    total_expected = len(TEST_MOVES) * TEST_REPEATS

    for repeat_idx in range(1, TEST_REPEATS + 1):
        for move in TEST_MOVES:
            print(f"  Teste {move.name} | repeticao {repeat_idx}/{TEST_REPEATS}")
            trial = _run_move_trial(mount, move, params)
            trials.append(trial)
            err_total = math.hypot(trial.final_err_az_deg, trial.final_err_alt_deg)
            status = "SUCESSO" if trial.success else "FALHA"
            print(
                f"    -> {status} em {trial.elapsed_s:.2f}s | "
                f"overshoots={trial.overshoots_az + trial.overshoots_alt} | "
                f"err_final={err_total:.5f}deg | motivo={trial.reason}"
            )
            if not trial.success:
                break
        if trials and not trials[-1].success:
            break

    score, success_count, median_time, mean_overshoots = _score_trials(trials, total_expected)
    median_str = f"{median_time:.3f}s" if median_time is not None else "n/a"
    print(
        f"  Score={score:.3f} | sucessos={success_count}/{total_expected} | "
        f"mediana={median_str} | overshoot_medio={mean_overshoots:.2f}"
    )
    return CandidateEvaluation(
        kp=params.kp,
        kd=params.kd,
        score=score,
        success_count=success_count,
        total_tests=total_expected,
        median_time_s=median_time,
        mean_overshoots=mean_overshoots,
        trials=trials,
    )


def _params_equal(a: TunedParams, b: TunedParams) -> bool:
    return abs(a.kp - b.kp) < 1e-9 and abs(a.kd - b.kd) < 1e-9


def twiddle_search(mount: bool):
    current = _clip_params(TunedParams(INITIAL_KP, INITIAL_KD))
    dp = {"kp": INITIAL_DP_KP, "kd": INITIAL_DP_KD}
    history: list[CandidateEvaluation] = []

    best_eval = evaluate_candidate(mount, current)
    history.append(best_eval)

    for iteration in range(1, TWIDDLE_MAX_ITERS + 1):
        if (dp["kp"] + dp["kd"]) < TWIDDLE_MIN_DP_SUM:
            print("\nParando: soma dos passos do twiddle ficou pequena o suficiente.")
            break

        print("\n" + "=" * 72)
        print(
            f"Iteracao {iteration}/{TWIDDLE_MAX_ITERS} | "
            f"best Kp={current.kp:.4f} Kd={current.kd:.4f} | "
            f"dp_kp={dp['kp']:.4f} dp_kd={dp['kd']:.4f}"
        )
        print("=" * 72)

        for field_name in ("kp", "kd"):
            improved = False

            trial_up = _clip_params(
                TunedParams(
                    kp=current.kp + (dp[field_name] if field_name == "kp" else 0.0),
                    kd=current.kd + (dp[field_name] if field_name == "kd" else 0.0),
                )
            )
            if not _params_equal(trial_up, current):
                eval_up = evaluate_candidate(mount, trial_up)
                history.append(eval_up)
                if eval_up.score < best_eval.score:
                    current = trial_up
                    best_eval = eval_up
                    dp[field_name] *= DP_GROW
                    improved = True

            if improved:
                continue

            trial_down = _clip_params(
                TunedParams(
                    kp=current.kp - (dp[field_name] if field_name == "kp" else 0.0),
                    kd=current.kd - (dp[field_name] if field_name == "kd" else 0.0),
                )
            )
            if not _params_equal(trial_down, current):
                eval_down = evaluate_candidate(mount, trial_down)
                history.append(eval_down)
                if eval_down.score < best_eval.score:
                    current = trial_down
                    best_eval = eval_down
                    dp[field_name] *= DP_GROW
                    improved = True

            if not improved:
                dp[field_name] *= DP_SHRINK

    return best_eval, history


def print_podium(history: list[CandidateEvaluation]):
    unique = {}
    for evaluation in history:
        key = (round(evaluation.kp, 6), round(evaluation.kd, 6))
        existing = unique.get(key)
        if existing is None or evaluation.score < existing.score:
            unique[key] = evaluation

    ranking = sorted(
        unique.values(),
        key=lambda item: (
            -item.success_count,
            item.score,
            item.median_time_s if item.median_time_s is not None else float("inf"),
            item.mean_overshoots,
        ),
    )

    print("\n=== Podio Autotune mov_simultaneo ===")
    for idx, item in enumerate(ranking[:5], start=1):
        median_str = f"{item.median_time_s:.3f}s" if item.median_time_s is not None else "n/a"
        print(
            f"{idx}. Kp={item.kp:.4f} Kd={item.kd:.4f} | "
            f"sucessos={item.success_count}/{item.total_tests} | "
            f"score={item.score:.3f} | mediana={median_str} | "
            f"overshoot_medio={item.mean_overshoots:.2f}"
        )


def save_results(path: str, best_eval: CandidateEvaluation, history: list[CandidateEvaluation]):
    payload = {
        "timestamp_epoch": time.time(),
        "config": {
            "safe_max_delta_deg": SAFE_MAX_DELTA_DEG,
            "safe_az_min_deg": SAFE_AZ_MIN_DEG,
            "safe_az_max_deg": SAFE_AZ_MAX_DEG,
            "safe_alt_min_deg": SAFE_ALT_MIN_DEG,
            "safe_alt_max_deg": SAFE_ALT_MAX_DEG,
            "test_timeout_s": TEST_TIMEOUT_S,
            "test_repeats": TEST_REPEATS,
            "twiddle_max_iters": TWIDDLE_MAX_ITERS,
            "ki_fixed": KI_FIXED,
            "overshoot_penalty": OVERSHOOT_PENALTY,
            "first_overshoot_weight": FIRST_OVERSHOOT_WEIGHT,
            "initial_kp": INITIAL_KP,
            "initial_kd": INITIAL_KD,
            "initial_dp_kp": INITIAL_DP_KP,
            "initial_dp_kd": INITIAL_DP_KD,
            "test_moves": [asdict(move) for move in TEST_MOVES],
        },
        "best": {
            "kp": best_eval.kp,
            "kd": best_eval.kd,
            "score": best_eval.score,
            "success_count": best_eval.success_count,
            "total_tests": best_eval.total_tests,
            "median_time_s": best_eval.median_time_s,
            "mean_overshoots": best_eval.mean_overshoots,
        },
        "history": [
            {
                "kp": evaluation.kp,
                "kd": evaluation.kd,
                "score": evaluation.score,
                "success_count": evaluation.success_count,
                "total_tests": evaluation.total_tests,
                "median_time_s": evaluation.median_time_s,
                "mean_overshoots": evaluation.mean_overshoots,
                "trials": [asdict(trial) for trial in evaluation.trials],
            }
            for evaluation in history
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    _validate_test_moves()
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    print("=== Autotune Seguro do mov_simultaneo ===")
    print("Busca local tipo twiddle em Kp/Kd, com Ki fixo e limites rigidos de seguranca.\n")
    mount = None

    try:
        mount = bool(int(input("mount 1, simulador 0: ").strip()))

        best_eval, history = twiddle_search(mount)
        print_podium(history)
        save_results(RESULTS_JSON, best_eval, history)
        print(f"\nResultados salvos em {display_path(RESULTS_JSON)}")
    except KeyboardInterrupt:
        print("\nAutotune interrompido pelo usuario.")
    finally:
        if mount is not None:
            try:
                move_axis(0, 0.0, mount)
                move_axis(1, 0.0, mount)
            except Exception:
                pass


if __name__ == "__main__":
    main()
