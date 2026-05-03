import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from artifact_paths import display_path, json_output_path, matrix_candidates
from Center_of_Mass import centro_camera, centro_massa
from PID_controll import ensure_connected, ensure_not_tracking, ensure_unparked
from Tracker import (
    calcular_cm_corrigido,
    capture_frame,
    connect_camera,
    disconnect_camera,
    reset_camera_roi,
    set_camera_roi,
)
from mov_simultaneo import VEL_MAX_LIMITE, VEL_MIN_LIMITE, move_axis, move_axes_pid_2d

# ===== Configuracao do experimento =====
WINDOW_SIZE = 200
EXPOSURE_SECONDS = 32e-6
TOLERANCIA_PX = 2.0
RECENTER_SETTLE_S = 1.0
TRACKER_TIMEOUT_S = 10.0
CONTROL_HZ = 30.0
SIGNAL_TIMEOUT_S = 0.45
VEL_MAX_TESTE = min(1.2, VEL_MAX_LIMITE)
MEASUREMENT_ALPHA = 0.45
CMD_ACCEL_LIMIT = 0.60
CMD_KEEPALIVE_S = 0.15
MIN_CMD_DELTA_TO_SEND = 2e-4
CMD_ZERO_SNAP = 0.35 * VEL_MIN_LIMITE
DERIVATIVE_ALPHA = 0.70
ENABLE_RUNAWAY_BRAKE = True
RUNAWAY_MARGIN_PX = 1.0
RUNAWAY_FRAMES = 4
RUNAWAY_HOLD_S = 0.40

CM_RECENTER_MAX_ITERS = 8
REPEAT_COUNT = 3
RESULTS_JSON = json_output_path("calibracao_pid_tracker_resultados.json")


@dataclass(frozen=True)
class PIDCandidate:
    name: str
    kp_az: float
    kp_alt: float
    kd_az: float
    kd_alt: float


@dataclass(frozen=True)
class Perturbation:
    name: str
    delta_az_deg: float
    delta_alt_deg: float


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: bool = False
    has_signal: bool = False
    measurement_seq: int = 0
    measurement_ts: float = 0.0
    dx_px: float = 0.0
    dy_px: float = 0.0
    dx_filt_px: float = 0.0
    dy_filt_px: float = 0.0
    err_az_deg: float = 0.0
    err_alt_deg: float = 0.0
    cmd_az_deg_s: float = 0.0
    cmd_alt_deg_s: float = 0.0
    runaway_events: int = 0


@dataclass
class TrialResult:
    candidate: str
    perturbation: str
    repeat_idx: int
    success: bool
    settle_time_s: float | None
    timeout_s: float
    center_entries: int
    runaway_events: int
    signal_loss_seen: bool
    final_dx_px: float | None
    final_dy_px: float | None
    reason: str


PID_CANDIDATES = [
    PIDCandidate("base_kp1.0_kd0.18", 1.0, 1.0, 0.18, 0.18),
    PIDCandidate("kp1.2_kd0.18", 1.2, 1.2, 0.18, 0.18),
    PIDCandidate("kp1.0_kd0.24", 1.0, 1.0, 0.24, 0.24),
    PIDCandidate("kp0.8_kd0.18", 0.8, 0.8, 0.18, 0.18),
    PIDCandidate("kp1.2_kd0.24", 1.2, 1.2, 0.24, 0.24),
]

PERTURBATIONS = [
    Perturbation("+Az", +0.02, 0.0),
    Perturbation("-Az", -0.02, 0.0),
    Perturbation("+Alt", 0.0, +0.02),
    Perturbation("-Alt", 0.0, -0.02),
]


class MeasurementPD:
    def __init__(self, kp, kd, output_limits, derivative_alpha=0.70):
        self.kp = kp
        self.kd = kd
        self.min_output, self.max_output = output_limits
        self.derivative_alpha = derivative_alpha
        self.reset()

    def reset(self):
        self._last_error = None
        self._last_t = None
        self._d_filt = 0.0

    def _clip(self, value):
        if self.min_output is not None and value < self.min_output:
            return self.min_output
        if self.max_output is not None and value > self.max_output:
            return self.max_output
        return value

    def update(self, error, timestamp):
        if self._last_t is None or timestamp <= self._last_t:
            self._last_error = error
            self._last_t = timestamp
            self._d_filt = 0.0
            return self._clip(self.kp * error)

        dt = max(timestamp - self._last_t, 1e-3)
        deriv = (error - self._last_error) / dt
        self._d_filt = (self.derivative_alpha * self._d_filt) + ((1.0 - self.derivative_alpha) * deriv)

        output = self._clip((self.kp * error) + (self.kd * self._d_filt))
        self._last_error = error
        self._last_t = timestamp
        return output


def _apply_min_velocity(cmd):
    if abs(cmd) < CMD_ZERO_SNAP:
        return 0.0
    if 0.0 < abs(cmd) < VEL_MIN_LIMITE:
        return float(VEL_MIN_LIMITE * np.sign(cmd))
    return float(cmd)


def _slew_limit(current, target, max_delta):
    delta = target - current
    if abs(delta) <= max_delta:
        return float(target)
    return float(current + (np.sign(delta) * max_delta))


def _pixel_error_to_mount_error(dx_px, dy_px, A_inv):
    vec_px = np.array([-dx_px, -dy_px], dtype=float)
    err_vec = A_inv @ vec_px
    return float(err_vec[0]), float(err_vec[1])


def _control_loop_pid(
    state: SharedState,
    A_inv: np.ndarray,
    usar_mount: bool,
    candidate: PIDCandidate,
):
    pid_az = MeasurementPD(
        kp=candidate.kp_az,
        kd=candidate.kd_az,
        output_limits=(-VEL_MAX_TESTE, VEL_MAX_TESTE),
        derivative_alpha=DERIVATIVE_ALPHA,
    )
    pid_alt = MeasurementPD(
        kp=candidate.kp_alt,
        kd=candidate.kd_alt,
        output_limits=(-VEL_MAX_TESTE, VEL_MAX_TESTE),
        derivative_alpha=DERIVATIVE_ALPHA,
    )

    dt_target = 1.0 / CONTROL_HZ
    last_loop_t = time.perf_counter()
    last_seq = -1
    target_cmd_az = 0.0
    target_cmd_alt = 0.0
    cmd_az = 0.0
    cmd_alt = 0.0
    last_sent_az = None
    last_sent_alt = None
    last_sent_az_t = 0.0
    last_sent_alt_t = 0.0
    err_az = 0.0
    err_alt = 0.0
    prev_radius_px = None
    runaway_count = 0
    brake_until = 0.0

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:
                loop_t0 = time.perf_counter()
                dt_loop = max(loop_t0 - last_loop_t, 1e-4)
                last_loop_t = loop_t0

                with state.lock:
                    stop = state.stop
                    has_signal = state.has_signal
                    seq = state.measurement_seq
                    measurement_ts = state.measurement_ts
                    dx_filt = state.dx_filt_px
                    dy_filt = state.dy_filt_px

                if stop:
                    break

                measurement_age = (loop_t0 - measurement_ts) if measurement_ts else 1e9
                signal_ok = has_signal and (measurement_age <= SIGNAL_TIMEOUT_S)

                if not signal_ok:
                    err_az = 0.0
                    err_alt = 0.0
                    target_cmd_az = 0.0
                    target_cmd_alt = 0.0
                    pid_az.reset()
                    pid_alt.reset()
                    prev_radius_px = None
                    runaway_count = 0
                elif seq != last_seq:
                    last_seq = seq
                    radius_px = float(np.hypot(dx_filt, dy_filt))

                    if abs(dx_filt) <= TOLERANCIA_PX and abs(dy_filt) <= TOLERANCIA_PX:
                        err_az = 0.0
                        err_alt = 0.0
                    else:
                        err_az, err_alt = _pixel_error_to_mount_error(dx_filt, dy_filt, A_inv)

                    target_cmd_az = pid_az.update(err_az, measurement_ts)
                    target_cmd_alt = pid_alt.update(err_alt, measurement_ts)

                    if ENABLE_RUNAWAY_BRAKE:
                        cmd_norm = float(np.hypot(cmd_az, cmd_alt))
                        if (
                            prev_radius_px is not None
                            and cmd_norm >= VEL_MIN_LIMITE
                            and radius_px > (prev_radius_px + RUNAWAY_MARGIN_PX)
                            and radius_px > (2.0 * TOLERANCIA_PX)
                        ):
                            runaway_count += 1
                        else:
                            runaway_count = 0

                        prev_radius_px = radius_px

                        if runaway_count >= RUNAWAY_FRAMES:
                            target_cmd_az = 0.0
                            target_cmd_alt = 0.0
                            cmd_az = 0.0
                            cmd_alt = 0.0
                            pid_az.reset()
                            pid_alt.reset()
                            brake_until = loop_t0 + RUNAWAY_HOLD_S
                            runaway_count = 0
                            with state.lock:
                                state.runaway_events += 1

                if loop_t0 < brake_until:
                    target_cmd_az = 0.0
                    target_cmd_alt = 0.0

                target_cmd_az = float(np.clip(target_cmd_az, -VEL_MAX_TESTE, VEL_MAX_TESTE))
                target_cmd_alt = float(np.clip(target_cmd_alt, -VEL_MAX_TESTE, VEL_MAX_TESTE))

                max_step = CMD_ACCEL_LIMIT * dt_loop
                cmd_az = _slew_limit(cmd_az, target_cmd_az, max_step)
                cmd_alt = _slew_limit(cmd_alt, target_cmd_alt, max_step)

                if abs(target_cmd_az) < 1e-12 and abs(cmd_az) < VEL_MIN_LIMITE:
                    cmd_az = 0.0
                if abs(target_cmd_alt) < 1e-12 and abs(cmd_alt) < VEL_MIN_LIMITE:
                    cmd_alt = 0.0

                cmd_az = _apply_min_velocity(float(np.clip(cmd_az, -VEL_MAX_TESTE, VEL_MAX_TESTE)))
                cmd_alt = _apply_min_velocity(float(np.clip(cmd_alt, -VEL_MAX_TESTE, VEL_MAX_TESTE)))

                send_az = (
                    last_sent_az is None
                    or abs(cmd_az - last_sent_az) >= MIN_CMD_DELTA_TO_SEND
                    or (loop_t0 - last_sent_az_t) >= CMD_KEEPALIVE_S
                )
                send_alt = (
                    last_sent_alt is None
                    or abs(cmd_alt - last_sent_alt) >= MIN_CMD_DELTA_TO_SEND
                    or (loop_t0 - last_sent_alt_t) >= CMD_KEEPALIVE_S
                )

                future_az = None
                future_alt = None
                if send_az:
                    future_az = executor.submit(move_axis, 0, cmd_az, usar_mount)
                if send_alt:
                    future_alt = executor.submit(move_axis, 1, cmd_alt, usar_mount)

                if future_az is not None:
                    future_az.result()
                    last_sent_az = cmd_az
                    last_sent_az_t = loop_t0
                if future_alt is not None:
                    future_alt.result()
                    last_sent_alt = cmd_alt
                    last_sent_alt_t = loop_t0

                with state.lock:
                    state.err_az_deg = err_az
                    state.err_alt_deg = err_alt
                    state.cmd_az_deg_s = cmd_az
                    state.cmd_alt_deg_s = cmd_alt

                elapsed = time.perf_counter() - loop_t0
                if elapsed < dt_target:
                    time.sleep(dt_target - elapsed)
        finally:
            move_axis(0, 0.0, usar_mount)
            move_axis(1, 0.0, usar_mount)


def centralize_with_center_of_mass(A_inv: np.ndarray, usar_mount: bool) -> tuple[bool, dict]:
    reset_camera_roi()

    last_dx = None
    last_dy = None
    for idx in range(1, CM_RECENTER_MAX_ITERS + 1):
        frame = capture_frame(EXPOSURE_SECONDS)
        cm = centro_massa(frame)
        if cm is None:
            return False, {"reason": "no_signal", "iterations": idx - 1}

        x_cm, y_cm, _, toca_borda = cm
        cx, cy = centro_camera(frame)
        dx = float(x_cm - cx)
        dy = float(y_cm - cy)
        last_dx = dx
        last_dy = dy

        print(
            f"  CM recenter passo {idx}/{CM_RECENTER_MAX_ITERS}: "
            f"dx={dx:+.1f}px dy={dy:+.1f}px"
        )

        if abs(dx) <= TOLERANCIA_PX and abs(dy) <= TOLERANCIA_PX and not toca_borda:
            return True, {
                "reason": "centered",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
            }

        vec_px = np.array([-dx, -dy], dtype=float)
        d_az_deg, d_alt_deg = A_inv @ vec_px
        move_axes_pid_2d(usar_mount, float(d_az_deg), float(d_alt_deg))

    return False, {
        "reason": "max_iterations",
        "iterations": CM_RECENTER_MAX_ITERS,
        "dx_px": last_dx,
        "dy_px": last_dy,
    }


def run_tracker_trial(
    A_inv: np.ndarray,
    usar_mount: bool,
    candidate: PIDCandidate,
    timeout_s: float,
) -> TrialResult:
    state = SharedState()
    ctrl_thread = threading.Thread(
        target=_control_loop_pid,
        args=(state, A_inv, usar_mount, candidate),
        daemon=True,
    )
    ctrl_thread.start()

    t_start = time.perf_counter()
    center_hold_start = None
    center_entries = 0
    was_centered = False
    signal_loss_seen = False
    final_dx = None
    final_dy = None
    reason = "timeout"
    success = False
    settle_time_s = None

    try:
        while True:
            t_now = time.perf_counter()
            if (t_now - t_start) >= timeout_s:
                reason = "timeout"
                break

            frame_window = capture_frame(EXPOSURE_SECONDS)
            cm = calcular_cm_corrigido(frame_window)

            if cm is None:
                with state.lock:
                    state.has_signal = False
                    state.measurement_seq += 1
                    state.measurement_ts = t_now
                    runaway_events = state.runaway_events

                signal_loss_seen = True
                center_hold_start = None
                was_centered = False
                final_dx = None
                final_dy = None
                time.sleep(0.001)
                continue

            x_cm_local, y_cm_local = cm
            dx = float(x_cm_local - (WINDOW_SIZE / 2))
            dy = float(y_cm_local - (WINDOW_SIZE / 2))
            final_dx = dx
            final_dy = dy

            with state.lock:
                if state.measurement_seq == 0 or not state.has_signal:
                    dx_filt = dx
                    dy_filt = dy
                else:
                    dx_filt = (MEASUREMENT_ALPHA * dx) + ((1.0 - MEASUREMENT_ALPHA) * state.dx_filt_px)
                    dy_filt = (MEASUREMENT_ALPHA * dy) + ((1.0 - MEASUREMENT_ALPHA) * state.dy_filt_px)

                state.dx_px = dx
                state.dy_px = dy
                state.dx_filt_px = float(dx_filt)
                state.dy_filt_px = float(dy_filt)
                state.has_signal = True
                state.measurement_seq += 1
                state.measurement_ts = t_now
                runaway_events = state.runaway_events

            is_centered = (abs(dx) < TOLERANCIA_PX) and (abs(dy) < TOLERANCIA_PX)
            if is_centered and not was_centered:
                center_entries += 1

            if is_centered:
                if center_hold_start is None:
                    center_hold_start = t_now
                elif (t_now - center_hold_start) >= RECENTER_SETTLE_S:
                    success = True
                    settle_time_s = t_now - t_start
                    reason = "stabilized"
                    break
            else:
                center_hold_start = None

            was_centered = is_centered
            time.sleep(0.001)

        with state.lock:
            runaway_events = state.runaway_events
    finally:
        with state.lock:
            state.stop = True
        ctrl_thread.join(timeout=2.0)

    return TrialResult(
        candidate=candidate.name,
        perturbation="",
        repeat_idx=0,
        success=success,
        settle_time_s=settle_time_s,
        timeout_s=timeout_s,
        center_entries=center_entries,
        runaway_events=runaway_events,
        signal_loss_seen=signal_loss_seen,
        final_dx_px=final_dx,
        final_dy_px=final_dy,
        reason=reason,
    )


def summarize_candidate(results: list[TrialResult]) -> dict:
    total = len(results)
    success_results = [r for r in results if r.success and r.settle_time_s is not None]
    success_count = len(success_results)
    success_rate = (success_count / total) if total else 0.0
    median_time = statistics.median([r.settle_time_s for r in success_results]) if success_results else None
    mean_runaway = statistics.mean([r.runaway_events for r in results]) if results else 0.0
    mean_center_entries = statistics.mean([r.center_entries for r in results]) if results else 0.0
    return {
        "success_count": success_count,
        "total": total,
        "success_rate": success_rate,
        "median_settle_time_s": median_time,
        "mean_runaway_events": mean_runaway,
        "mean_center_entries": mean_center_entries,
    }


def print_candidate_summary(candidate: PIDCandidate, results: list[TrialResult]):
    summary = summarize_candidate(results)
    median_str = (
        f"{summary['median_settle_time_s']:.3f}s"
        if summary["median_settle_time_s"] is not None
        else "n/a"
    )
    print(
        f"Resumo {candidate.name}: "
        f"sucessos={summary['success_count']}/{summary['total']} "
        f"({100.0 * summary['success_rate']:.0f}%), "
        f"mediana={median_str}, "
        f"runaway_medio={summary['mean_runaway_events']:.2f}, "
        f"entradas_no_centro={summary['mean_center_entries']:.2f}"
    )


def save_results(
    path: str,
    candidate_summaries: dict[str, dict],
    trial_results: list[TrialResult],
):
    payload = {
        "timestamp_epoch": time.time(),
        "config": {
            "window_size": WINDOW_SIZE,
            "exposure_seconds": EXPOSURE_SECONDS,
            "tolerancia_px": TOLERANCIA_PX,
            "recenter_settle_s": RECENTER_SETTLE_S,
            "tracker_timeout_s": TRACKER_TIMEOUT_S,
            "repeat_count": REPEAT_COUNT,
            "perturbations": [asdict(p) for p in PERTURBATIONS],
            "pid_candidates": [asdict(c) for c in PID_CANDIDATES],
        },
        "candidate_summaries": candidate_summaries,
        "trial_results": [asdict(r) for r in trial_results],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_podium(candidate_summaries: dict[str, dict]):
    ranking = sorted(
        candidate_summaries.items(),
        key=lambda item: (
            -item[1]["success_rate"],
            item[1]["median_settle_time_s"] if item[1]["median_settle_time_s"] is not None else float("inf"),
            item[1]["mean_runaway_events"],
            item[1]["mean_center_entries"],
        ),
    )

    print("\n=== Podio PID Tracker ===")
    for idx, (name, summary) in enumerate(ranking[:3], start=1):
        median_str = (
            f"{summary['median_settle_time_s']:.3f}s"
            if summary["median_settle_time_s"] is not None
            else "n/a"
        )
        print(
            f"{idx}. {name} | "
            f"sucessos={summary['success_count']}/{summary['total']} "
            f"({100.0 * summary['success_rate']:.0f}%) | "
            f"mediana={median_str} | "
            f"runaway_medio={summary['mean_runaway_events']:.2f}"
        )


def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()
    connect_camera()

    trial_results: list[TrialResult] = []

    try:
        for candidate in matrix_candidates("A_inv_fine.npy", "calibracao_A_inv.npy"):
            if not candidate.exists():
                continue
            A_inv = np.load(candidate)
            source = display_path(candidate)
            break
        else:
            raise FileNotFoundError("Nenhuma matriz de tracking encontrada.")
        if A_inv.shape != (2, 2):
            raise ValueError(f"Matriz {source} precisa ser 2x2")

        usar_mount = bool(int(input("mount 1, simulador 0: ").strip()))

        print("\n=== Calibracao PID do Tracker Continuo ===")
        print("Fluxo: CM centraliza -> perturbacao conhecida -> tracker headless -> ranking final.\n")
        print(f"Usando matriz de tracking: {source}\n")

        current_center_ok, center_info = centralize_with_center_of_mass(A_inv, usar_mount)
        if not current_center_ok:
            raise RuntimeError(f"Falha ao recentralizar pelo CM no inicio: {center_info}")
        set_camera_roi(WINDOW_SIZE, WINDOW_SIZE)

        for candidate in PID_CANDIDATES:
            print("\n" + "=" * 72)
            print(
                f"Testando {candidate.name} | "
                f"KpAz={candidate.kp_az:.3f} KpAlt={candidate.kp_alt:.3f} "
                f"KdAz={candidate.kd_az:.3f} KdAlt={candidate.kd_alt:.3f}"
            )
            print("=" * 72)

            candidate_results: list[TrialResult] = []

            current_center_ok, center_info = centralize_with_center_of_mass(A_inv, usar_mount)
            if not current_center_ok:
                raise RuntimeError(f"Falha ao recentralizar antes do candidato {candidate.name}: {center_info}")
            set_camera_roi(WINDOW_SIZE, WINDOW_SIZE)

            for repeat_idx in range(1, REPEAT_COUNT + 1):
                for perturbation in PERTURBATIONS:
                    print(
                        f"\n[{candidate.name}] repeticao {repeat_idx}/{REPEAT_COUNT} | "
                        f"perturbacao {perturbation.name} "
                        f"(dAz={perturbation.delta_az_deg:+.4f}, dAlt={perturbation.delta_alt_deg:+.4f})"
                    )

                    move_axes_pid_2d(
                        usar_mount,
                        perturbation.delta_az_deg,
                        perturbation.delta_alt_deg,
                    )

                    trial = run_tracker_trial(
                        A_inv=A_inv,
                        usar_mount=usar_mount,
                        candidate=candidate,
                        timeout_s=TRACKER_TIMEOUT_S,
                    )
                    trial.perturbation = perturbation.name
                    trial.repeat_idx = repeat_idx
                    trial_results.append(trial)
                    candidate_results.append(trial)

                    if trial.success:
                        print(
                            f"  -> SUCESSO em {trial.settle_time_s:.3f}s | "
                            f"runaway={trial.runaway_events} | "
                            f"entradas_no_centro={trial.center_entries}"
                        )
                    else:
                        print(
                            f"  -> FALHA ({trial.reason}) | "
                            f"runaway={trial.runaway_events} | "
                            f"dx_final={trial.final_dx_px} dy_final={trial.final_dy_px}"
                        )
                        current_center_ok, center_info = centralize_with_center_of_mass(A_inv, usar_mount)
                        if not current_center_ok:
                            raise RuntimeError(
                                f"Falha ao recentralizar apos o teste {candidate.name}/{perturbation.name}: "
                                f"{center_info}"
                            )
                        set_camera_roi(WINDOW_SIZE, WINDOW_SIZE)

            print_candidate_summary(candidate, candidate_results)

        candidate_summaries = {
            candidate.name: summarize_candidate([r for r in trial_results if r.candidate == candidate.name])
            for candidate in PID_CANDIDATES
        }
        print_podium(candidate_summaries)
        save_results(RESULTS_JSON, candidate_summaries, trial_results)
        print(f"\nResultados salvos em {display_path(RESULTS_JSON)}")

    except KeyboardInterrupt:
        print("\nCalibracao PID interrompida pelo usuario.")
    except Exception as exc:
        print(f"\nErro na calibracao PID do tracker: {exc}")
    finally:
        try:
            reset_camera_roi()
        except Exception:
            pass
        disconnect_camera()


if __name__ == "__main__":
    main()
