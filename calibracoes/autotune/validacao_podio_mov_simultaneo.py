import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import display_path, json_candidates, json_output_path
from calibracoes.autotune.autotune_mov_simultaneo import (
    MoveTrialResult,
    TestMove,
    TunedParams,
    _run_move_trial,
    ensure_connected,
    ensure_not_tracking,
    ensure_unparked,
    move_axis,
)

SOURCE_RESULTS_JSON = "autotune_mov_simultaneo_resultados.json"
RESULTS_JSON = json_output_path("validacao_podio_mov_simultaneo_resultados.json")
TOP_N = 3
VALIDATION_REPEATS = 1

FAILURE_PENALTY = 150.0
SKIPPED_TEST_PENALTY = 80.0
OVERSHOOT_PENALTY = 4.0
FIRST_OVERSHOOT_WEIGHT = 0.20

VALIDATION_MOVES = [
    TestMove("Az +1", +1.0, 0.0),
    TestMove("Az -1", -1.0, 0.0),
    TestMove("Az +3", +3.0, 0.0),
    TestMove("Az -3", -3.0, 0.0),
    TestMove("Alt +0.5", 0.0, +0.5),
    TestMove("Alt -0.5", 0.0, -0.5),
    TestMove("Alt +1.0", 0.0, +1.0),
    TestMove("Alt -1.0", 0.0, -1.0),
    TestMove("Diag +3/+1", +3.0, +1.0),
    TestMove("Diag -3/+1", -3.0, +1.0),
    TestMove("Diag +3/-1", +3.0, -1.0),
    TestMove("Diag -3/-1", -3.0, -1.0),
]


@dataclass
class ValidationTrialResult:
    candidate_name: str
    repeat_idx: int
    move_name: str
    success: bool
    elapsed_s: float
    overshoots_total: int
    reason: str
    final_err_az_deg: float
    final_err_alt_deg: float


@dataclass
class CandidateValidation:
    candidate_name: str
    kp: float
    kd: float
    score: float
    success_count: int
    total_tests: int
    median_time_s: float | None
    mean_overshoots: float
    trials: list[ValidationTrialResult]


def _load_top_candidates(path: str, top_n: int) -> list[tuple[str, TunedParams]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not history:
        raise RuntimeError(f"Nenhum historico encontrado em {path}")

    unique = {}
    for item in history:
        key = (round(float(item["kp"]), 6), round(float(item["kd"]), 6))
        existing = unique.get(key)
        if existing is None or float(item["score"]) < float(existing["score"]):
            unique[key] = item

    ranking = sorted(
        unique.values(),
        key=lambda item: (
            -int(item["success_count"]),
            float(item["score"]),
            float(item["median_time_s"]) if item["median_time_s"] is not None else float("inf"),
            float(item["mean_overshoots"]),
        ),
    )

    selected = []
    for idx, item in enumerate(ranking[:top_n], start=1):
        kp = float(item["kp"])
        kd = float(item["kd"])
        name = f"top{idx}_kp{kp:.4f}_kd{kd:.4f}"
        selected.append((name, TunedParams(kp=kp, kd=kd)))
    return selected


def _convert_trial(candidate_name: str, repeat_idx: int, trial: MoveTrialResult) -> ValidationTrialResult:
    return ValidationTrialResult(
        candidate_name=candidate_name,
        repeat_idx=repeat_idx,
        move_name=trial.move_name,
        success=trial.success,
        elapsed_s=trial.elapsed_s,
        overshoots_total=trial.overshoots_az + trial.overshoots_alt,
        reason=trial.reason,
        final_err_az_deg=trial.final_err_az_deg,
        final_err_alt_deg=trial.final_err_alt_deg,
    )


def _overshoot_cost(overshoots_total: int) -> float:
    if overshoots_total <= 0:
        return 0.0
    if overshoots_total == 1:
        return FIRST_OVERSHOOT_WEIGHT * OVERSHOOT_PENALTY
    return (FIRST_OVERSHOOT_WEIGHT * OVERSHOOT_PENALTY) + ((overshoots_total - 1) * OVERSHOOT_PENALTY)


def _score_trials(trials: list[ValidationTrialResult], total_expected: int) -> tuple[float, int, float | None, float]:
    success_count = sum(1 for trial in trials if trial.success)
    success_times = [trial.elapsed_s for trial in trials if trial.success]
    median_time = statistics.median(success_times) if success_times else None
    mean_overshoots = statistics.mean([trial.overshoots_total for trial in trials]) if trials else 0.0

    score = 0.0
    for trial in trials:
        overshoot_cost = _overshoot_cost(trial.overshoots_total)
        if trial.success:
            score += trial.elapsed_s
            score += overshoot_cost
        else:
            score += FAILURE_PENALTY
            score += trial.elapsed_s
            score += 2.0 * overshoot_cost

    skipped = max(total_expected - len(trials), 0)
    score += SKIPPED_TEST_PENALTY * skipped
    return score, success_count, median_time, mean_overshoots


def evaluate_candidate(mount: bool, candidate_name: str, params: TunedParams) -> CandidateValidation:
    print(f"\nValidando {candidate_name}: Kp={params.kp:.4f}, Kd={params.kd:.4f}")
    trials: list[ValidationTrialResult] = []
    total_expected = len(VALIDATION_MOVES) * VALIDATION_REPEATS

    for repeat_idx in range(1, VALIDATION_REPEATS + 1):
        for move in VALIDATION_MOVES:
            print(f"  Teste {move.name} | repeticao {repeat_idx}/{VALIDATION_REPEATS}")
            trial_raw = _run_move_trial(mount, move, params)
            trial = _convert_trial(candidate_name, repeat_idx, trial_raw)
            trials.append(trial)
            status = "SUCESSO" if trial.success else "FALHA"
            print(
                f"    -> {status} em {trial.elapsed_s:.2f}s | "
                f"overshoots={trial.overshoots_total} | motivo={trial.reason}"
            )

    score, success_count, median_time, mean_overshoots = _score_trials(trials, total_expected)
    median_str = f"{median_time:.3f}s" if median_time is not None else "n/a"
    print(
        f"  Score={score:.3f} | sucessos={success_count}/{total_expected} | "
        f"mediana={median_str} | overshoot_medio={mean_overshoots:.2f}"
    )

    return CandidateValidation(
        candidate_name=candidate_name,
        kp=params.kp,
        kd=params.kd,
        score=score,
        success_count=success_count,
        total_tests=total_expected,
        median_time_s=median_time,
        mean_overshoots=mean_overshoots,
        trials=trials,
    )


def print_podium(results: list[CandidateValidation]):
    ranking = sorted(
        results,
        key=lambda item: (
            -item.success_count,
            item.score,
            item.median_time_s if item.median_time_s is not None else float("inf"),
            item.mean_overshoots,
        ),
    )

    print("\n=== Podio Validacao do mov_simultaneo ===")
    for idx, item in enumerate(ranking, start=1):
        median_str = f"{item.median_time_s:.3f}s" if item.median_time_s is not None else "n/a"
        print(
            f"{idx}. {item.candidate_name} | "
            f"sucessos={item.success_count}/{item.total_tests} | "
            f"score={item.score:.3f} | mediana={median_str} | "
            f"overshoot_medio={item.mean_overshoots:.2f}"
        )


def save_results(path: str, source_path: str, results: list[CandidateValidation]):
    payload = {
        "timestamp_epoch": time.time(),
        "source_results_json": display_path(source_path),
        "config": {
            "top_n": TOP_N,
            "validation_repeats": VALIDATION_REPEATS,
            "validation_moves": [asdict(move) for move in VALIDATION_MOVES],
            "failure_penalty": FAILURE_PENALTY,
            "overshoot_penalty": OVERSHOOT_PENALTY,
            "first_overshoot_weight": FIRST_OVERSHOOT_WEIGHT,
        },
        "results": [
            {
                "candidate_name": item.candidate_name,
                "kp": item.kp,
                "kd": item.kd,
                "score": item.score,
                "success_count": item.success_count,
                "total_tests": item.total_tests,
                "median_time_s": item.median_time_s,
                "mean_overshoots": item.mean_overshoots,
                "trials": [asdict(trial) for trial in item.trials],
            }
            for item in results
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    mount = None
    try:
        mount = bool(int(input("mount 1, simulador 0: ").strip()))
        source_candidates = json_candidates(SOURCE_RESULTS_JSON)
        for candidate in source_candidates:
            if candidate.exists():
                source_path = candidate
                break
        else:
            source_path = source_candidates[0]
        top_candidates = _load_top_candidates(source_path, TOP_N)

        print("=== Validacao do Podio mov_simultaneo ===")
        print("Top 3 do autotune principal testados em amplitudes menores e movimentos diagonais.\n")

        results = [
            evaluate_candidate(mount, candidate_name, params)
            for candidate_name, params in top_candidates
        ]
        print_podium(results)
        save_results(RESULTS_JSON, source_path, results)
        print(f"\nResultados salvos em {display_path(RESULTS_JSON)}")
    except KeyboardInterrupt:
        print("\nValidacao do podio interrompida pelo usuario.")
    finally:
        if mount is not None:
            try:
                move_axis(0, 0.0, mount)
                move_axis(1, 0.0, mount)
            except Exception:
                pass


if __name__ == "__main__":
    main()
