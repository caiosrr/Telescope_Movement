import json
import math
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
    MeasurementPDTrim,
    calcular_cm_corrigido,
    capture_frame,
    connect_camera,
    disconnect_camera,
    reset_camera_roi,
    set_camera_roi,
)
from mov_simultaneo import (
    VEL_MAX_LIMITE,
    VEL_MIN_LIMITE,
    move_axis,
    move_axes_pid_2d,
    read_altaz,
)

# ===== Configuracao segura do autotune =====
WINDOW_SIZE = 200
EXPOSURE_SECONDS = 32e-6
TOLERANCIA_PX = 2.0
CONTROL_DEADBAND_PX = 0.35
RECENTER_SETTLE_S = 1.0
TRACKER_TIMEOUT_S = 10.0
CONTROL_HZ = 45.0
SIGNAL_TIMEOUT_S = 0.45
VEL_MAX_TESTE = min(1.6, VEL_MAX_LIMITE)
MEASUREMENT_ALPHA = 0.70
CMD_ACCEL_LIMIT = 2.00
CMD_KEEPALIVE_S = 0.15
MIN_CMD_DELTA_TO_SEND = 2e-4
CMD_ZERO_SNAP = 0.35 * VEL_MIN_LIMITE
DERIVATIVE_ALPHA = 0.70
TRIM_LIMIT = 0.020
TRIM_LEAK = 0.985
TRIM_ERROR_MAX_DEG = 0.0006
TRIM_DERIVATIVE_MAX_DEG_S = 0.006
TRIM_SAME_SIGN_S = 0.80
TRIM_SIGN_EPS_DEG = 0.00010
TRIM_SIGN_FLIP_DAMP = 0.35
TRIM_ENTER_RADIUS_PX = 1.3
TRIM_EXIT_RADIUS_PX = 2.2
ENABLE_RUNAWAY_BRAKE = True
RUNAWAY_MARGIN_PX = 1.0
RUNAWAY_FRAMES = 4
RUNAWAY_HOLD_S = 0.40

SAFE_MAX_DELTA_DEG = 2.0
SAFE_AZ_MIN_DEG = 270.0
SAFE_AZ_MAX_DEG = 30.0
SAFE_ALT_MIN_DEG = -30.0
SAFE_ALT_MAX_DEG = 30.0
SAFE_MAX_RECENTER_DELTA_DEG = 2.0

CM_RECENTER_MAX_ITERS = 8
REPEAT_COUNT = 1
TWIDDLE_MAX_ITERS = 2
TWIDDLE_MIN_DP_SUM = 1.0
DP_SHRINK = 0.60
DP_GROW = 1.20
RESULTS_JSON = json_output_path("autotune_pid_tracker_resultados.json")

INITIAL_KP = 1.44
INITIAL_TRIM_GAIN = 1.2
INITIAL_KD = 0.18
# Busca local curta: prioriza a malha rapida e trata o trim como camada lenta.
INITIAL_DP_KP = 0.08
INITIAL_DP_TRIM_GAIN = 0.8
INITIAL_DP_KD = 0.03
KP_BOUNDS = (0.30, 3.0)
TRIM_GAIN_BOUNDS = (0.0, 8.0)
KD_BOUNDS = (0.00, 0.60)

EARLY_ABORT_AFTER_S = 3.5
EARLY_ABORT_RADIUS_PX = 10.0
EARLY_ABORT_RUNAWAY_EVENTS = 2

FAILURE_PENALTY = 120.0
RUNAWAY_PENALTY = 12.0
REENTRY_PENALTY = 1.5
FIRST_REENTRY_WEIGHT = 0.20
SIGNAL_LOSS_PENALTY = 15.0
SKIPPED_TEST_PENALTY = 80.0


@dataclass(frozen=True)
class TunedParams:
    kp: float
    trim_gain: float
    kd: float


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


@dataclass
class CandidateEvaluation:
    kp: float
    trim_gain: float
    kd: float
    score: float
    success_count: int
    total_tests: int
    median_time_s: float | None
    mean_runaway_events: float
    mean_center_entries: float
    mean_final_radius_px: float
    trials: list[TrialResult]


PERTURBATIONS = [
    # Comeca pelos sentidos que historicamente derrubam candidatos ruins mais cedo.
    Perturbation("+Alt", 0.0, +0.02),
    Perturbation("-Az", -0.02, 0.0),
    Perturbation("+Az", +0.02, 0.0),
    Perturbation("-Alt", 0.0, -0.02),
]


def _clip_params(params: TunedParams) -> TunedParams:
    kp = float(np.clip(params.kp, KP_BOUNDS[0], KP_BOUNDS[1]))
    trim_gain = float(np.clip(params.trim_gain, TRIM_GAIN_BOUNDS[0], TRIM_GAIN_BOUNDS[1]))
    kd = float(np.clip(params.kd, KD_BOUNDS[0], KD_BOUNDS[1]))
    return TunedParams(kp=kp, trim_gain=trim_gain, kd=kd)


def _params_equal(a: TunedParams, b: TunedParams) -> bool:
    return (
        abs(a.kp - b.kp) < 1e-9
        and abs(a.trim_gain - b.trim_gain) < 1e-9
        and abs(a.kd - b.kd) < 1e-9
    )


def _validate_perturbations():
    for perturbation in PERTURBATIONS:
        if abs(perturbation.delta_az_deg) > SAFE_MAX_DELTA_DEG or abs(perturbation.delta_alt_deg) > SAFE_MAX_DELTA_DEG:
            raise ValueError(
                f"Perturbacao {perturbation.name} excede o limite seguro de {SAFE_MAX_DELTA_DEG} deg."
            )


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


def _az_in_safe_window(az_deg: float) -> bool:
    az = az_deg % 360.0
    az_min = SAFE_AZ_MIN_DEG % 360.0
    az_max = SAFE_AZ_MAX_DEG % 360.0
    if az_min <= az_max:
        return az_min <= az <= az_max
    return az >= az_min or az <= az_max


def _reentry_cost(extra_entries: int) -> float:
    if extra_entries <= 0:
        return 0.0
    if extra_entries == 1:
        return FIRST_REENTRY_WEIGHT * REENTRY_PENALTY
    return (FIRST_REENTRY_WEIGHT * REENTRY_PENALTY) + ((extra_entries - 1) * REENTRY_PENALTY)


def _ensure_alt_target_safe(current_alt_deg: float, delta_alt_deg: float):
    target_alt = current_alt_deg + delta_alt_deg
    if target_alt < SAFE_ALT_MIN_DEG or target_alt > SAFE_ALT_MAX_DEG:
        raise RuntimeError(
            f"Movimento de altitude sairia da faixa segura [{SAFE_ALT_MIN_DEG}, {SAFE_ALT_MAX_DEG}] deg: "
            f"{current_alt_deg:.4f} -> {target_alt:.4f}"
        )


def _apply_safe_perturbation(usar_mount: bool, perturbation: Perturbation):
    if abs(perturbation.delta_az_deg) > SAFE_MAX_DELTA_DEG or abs(perturbation.delta_alt_deg) > SAFE_MAX_DELTA_DEG:
        raise RuntimeError(
            f"Perturbacao {perturbation.name} excede o limite seguro de {SAFE_MAX_DELTA_DEG} deg."
        )
    az_now, alt_now = read_altaz()
    target_az = (az_now + perturbation.delta_az_deg) % 360.0
    if not _az_in_safe_window(az_now):
        raise RuntimeError(
            f"Azimute atual ({az_now:.4f} deg) esta fora da janela segura "
            f"[{SAFE_AZ_MIN_DEG}, {SAFE_AZ_MAX_DEG}] deg."
        )
    if not _az_in_safe_window(target_az):
        raise RuntimeError(
            f"Perturbacao {perturbation.name} sairia da janela segura de azimute "
            f"[{SAFE_AZ_MIN_DEG}, {SAFE_AZ_MAX_DEG}] deg."
        )
    _ensure_alt_target_safe(alt_now, perturbation.delta_alt_deg)
    move_axes_pid_2d(usar_mount, perturbation.delta_az_deg, perturbation.delta_alt_deg)


def _control_loop_pd_trim(state: SharedState, A_inv: np.ndarray, usar_mount: bool, params: TunedParams):
    ctrl_az = MeasurementPDTrim(
        kp=params.kp,
        kd=params.kd,
        trim_gain=params.trim_gain,
        output_limits=(-VEL_MAX_TESTE, VEL_MAX_TESTE),
        derivative_alpha=DERIVATIVE_ALPHA,
        trim_limit=TRIM_LIMIT,
        trim_leak=TRIM_LEAK,
        trim_error_max=TRIM_ERROR_MAX_DEG,
        trim_derivative_max=TRIM_DERIVATIVE_MAX_DEG_S,
        trim_same_sign_s=TRIM_SAME_SIGN_S,
        trim_sign_eps=TRIM_SIGN_EPS_DEG,
        trim_sign_flip_damp=TRIM_SIGN_FLIP_DAMP,
    )
    ctrl_alt = MeasurementPDTrim(
        kp=params.kp,
        kd=params.kd,
        trim_gain=params.trim_gain,
        output_limits=(-VEL_MAX_TESTE, VEL_MAX_TESTE),
        derivative_alpha=DERIVATIVE_ALPHA,
        trim_limit=TRIM_LIMIT,
        trim_leak=TRIM_LEAK,
        trim_error_max=TRIM_ERROR_MAX_DEG,
        trim_derivative_max=TRIM_DERIVATIVE_MAX_DEG_S,
        trim_same_sign_s=TRIM_SAME_SIGN_S,
        trim_sign_eps=TRIM_SIGN_EPS_DEG,
        trim_sign_flip_damp=TRIM_SIGN_FLIP_DAMP,
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
    trim_mode_active = False

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
                    ctrl_az.reset()
                    ctrl_alt.reset()
                    prev_radius_px = None
                    runaway_count = 0
                    trim_mode_active = False
                elif seq != last_seq:
                    last_seq = seq
                    radius_px = float(np.hypot(dx_filt, dy_filt))

                    if radius_px <= TRIM_ENTER_RADIUS_PX:
                        trim_mode_active = True
                    elif radius_px >= TRIM_EXIT_RADIUS_PX:
                        if trim_mode_active:
                            ctrl_az.clear_trim()
                            ctrl_alt.clear_trim()
                        trim_mode_active = False

                    if (
                        abs(dx_filt) <= CONTROL_DEADBAND_PX
                        and abs(dy_filt) <= CONTROL_DEADBAND_PX
                    ):
                        err_az = 0.0
                        err_alt = 0.0
                        ctrl_az.clear_trim()
                        ctrl_alt.clear_trim()
                    else:
                        err_az, err_alt = _pixel_error_to_mount_error(dx_filt, dy_filt, A_inv)

                    trim_allowed = trim_mode_active and (loop_t0 >= brake_until)
                    target_cmd_az, _ = ctrl_az.update(err_az, measurement_ts, trim_allowed)
                    target_cmd_alt, _ = ctrl_alt.update(err_alt, measurement_ts, trim_allowed)

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
                            ctrl_az.reset()
                            ctrl_alt.reset()
                            trim_mode_active = False
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


def centralize_with_center_of_mass_safe(A_inv: np.ndarray, usar_mount: bool) -> tuple[bool, dict]:
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
        if abs(d_az_deg) > SAFE_MAX_RECENTER_DELTA_DEG or abs(d_alt_deg) > SAFE_MAX_RECENTER_DELTA_DEG:
            return False, {
                "reason": "unsafe_recenter_delta",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
                "d_az_deg": float(d_az_deg),
                "d_alt_deg": float(d_alt_deg),
            }

        az_now, alt_now = read_altaz()
        target_az = (az_now + float(d_az_deg)) % 360.0
        if not _az_in_safe_window(az_now):
            return False, {
                "reason": "unsafe_recenter_azimuth_current",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
                "az_deg": float(az_now),
            }
        if not _az_in_safe_window(target_az):
            return False, {
                "reason": "unsafe_recenter_azimuth_target",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
                "az_deg": float(az_now),
                "target_az_deg": float(target_az),
            }
        try:
            _ensure_alt_target_safe(alt_now, float(d_alt_deg))
        except RuntimeError as exc:
            return False, {
                "reason": "unsafe_recenter_altitude",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
                "error": str(exc),
            }

        move_axes_pid_2d(usar_mount, float(d_az_deg), float(d_alt_deg))

    return False, {
        "reason": "max_iterations",
        "iterations": CM_RECENTER_MAX_ITERS,
        "dx_px": last_dx,
        "dy_px": last_dy,
    }


def run_tracker_trial(A_inv: np.ndarray, usar_mount: bool, params: TunedParams, timeout_s: float) -> TrialResult:
    state = SharedState()
    ctrl_thread = threading.Thread(
        target=_control_loop_pd_trim,
        args=(state, A_inv, usar_mount, params),
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

            elapsed_s = t_now - t_start
            radius_px = math.hypot(dx, dy)
            if (
                not success
                and center_entries == 0
                and elapsed_s >= EARLY_ABORT_AFTER_S
                and runaway_events >= EARLY_ABORT_RUNAWAY_EVENTS
                and radius_px >= EARLY_ABORT_RADIUS_PX
            ):
                reason = "early_unstable_abort"
                break

            was_centered = is_centered
            time.sleep(0.001)

        with state.lock:
            runaway_events = state.runaway_events
    finally:
        with state.lock:
            state.stop = True
        ctrl_thread.join(timeout=2.0)

    return TrialResult(
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


def _score_trials(trials: list[TrialResult], total_expected: int) -> tuple[float, int, float | None, float, float, float]:
    success_results = [trial for trial in trials if trial.success and trial.settle_time_s is not None]
    success_count = len(success_results)
    median_time = statistics.median([trial.settle_time_s for trial in success_results]) if success_results else None
    mean_runaway = statistics.mean([trial.runaway_events for trial in trials]) if trials else 0.0
    mean_center_entries = statistics.mean([trial.center_entries for trial in trials]) if trials else 0.0

    radii = []
    for trial in trials:
        if trial.final_dx_px is not None and trial.final_dy_px is not None:
            radii.append(math.hypot(trial.final_dx_px, trial.final_dy_px))
    mean_final_radius = statistics.mean(radii) if radii else 0.0

    score = 0.0
    for trial in trials:
        extra_entries = max(trial.center_entries - 1, 0)
        score += RUNAWAY_PENALTY * trial.runaway_events
        score += _reentry_cost(extra_entries)
        if trial.signal_loss_seen:
            score += SIGNAL_LOSS_PENALTY

        if trial.success and trial.settle_time_s is not None:
            score += trial.settle_time_s
        else:
            score += FAILURE_PENALTY
            score += trial.timeout_s

    skipped = max(total_expected - len(trials), 0)
    score += SKIPPED_TEST_PENALTY * skipped
    return score, success_count, median_time, mean_runaway, mean_center_entries, mean_final_radius


def _partial_trial_score(trials: list[TrialResult]) -> float:
    score = 0.0
    for trial in trials:
        extra_entries = max(trial.center_entries - 1, 0)
        score += RUNAWAY_PENALTY * trial.runaway_events
        score += _reentry_cost(extra_entries)
        if trial.signal_loss_seen:
            score += SIGNAL_LOSS_PENALTY
        if trial.success and trial.settle_time_s is not None:
            score += trial.settle_time_s
        else:
            score += FAILURE_PENALTY
            score += trial.timeout_s
    return score


def evaluate_candidate(
    A_inv: np.ndarray,
    usar_mount: bool,
    params: TunedParams,
    incumbent: CandidateEvaluation | None = None,
) -> CandidateEvaluation:
    params = _clip_params(params)
    print(
        f"\nAvaliando candidato: "
        f"Kp={params.kp:.4f}, Trim={params.trim_gain:.4f}, Kd={params.kd:.4f}"
    )

    trials: list[TrialResult] = []
    total_expected = len(PERTURBATIONS) * REPEAT_COUNT
    abort_candidate = False

    for repeat_idx in range(1, REPEAT_COUNT + 1):
        for perturbation in PERTURBATIONS:
            print(
                f"  Teste {perturbation.name} | repeticao {repeat_idx}/{REPEAT_COUNT}"
            )
            center_ok, center_info = centralize_with_center_of_mass_safe(A_inv, usar_mount)
            if not center_ok:
                reason = f"recenter_failed:{center_info.get('reason', 'unknown')}"
                trial = TrialResult(
                    perturbation=perturbation.name,
                    repeat_idx=repeat_idx,
                    success=False,
                    settle_time_s=None,
                    timeout_s=TRACKER_TIMEOUT_S,
                    center_entries=0,
                    runaway_events=0,
                    signal_loss_seen=False,
                    final_dx_px=center_info.get("dx_px"),
                    final_dy_px=center_info.get("dy_px"),
                    reason=reason,
                )
                trials.append(trial)
                print(f"    -> FALHA antes do teste: {reason}")
                break

            set_camera_roi(WINDOW_SIZE, WINDOW_SIZE)

            try:
                _apply_safe_perturbation(usar_mount, perturbation)
                trial = run_tracker_trial(A_inv=A_inv, usar_mount=usar_mount, params=params, timeout_s=TRACKER_TIMEOUT_S)
            except Exception as exc:
                trial = TrialResult(
                    perturbation=perturbation.name,
                    repeat_idx=repeat_idx,
                    success=False,
                    settle_time_s=None,
                    timeout_s=TRACKER_TIMEOUT_S,
                    center_entries=0,
                    runaway_events=0,
                    signal_loss_seen=False,
                    final_dx_px=None,
                    final_dy_px=None,
                    reason=f"exception:{exc}",
                )

            trial.perturbation = perturbation.name
            trial.repeat_idx = repeat_idx
            trials.append(trial)

            final_radius = (
                math.hypot(trial.final_dx_px, trial.final_dy_px)
                if trial.final_dx_px is not None and trial.final_dy_px is not None
                else float("nan")
            )
            status = "SUCESSO" if trial.success else "FALHA"
            settle_str = f"{trial.settle_time_s:.3f}s" if trial.settle_time_s is not None else "n/a"
            radius_str = f"{final_radius:.2f}px" if math.isfinite(final_radius) else "n/a"
            print(
                f"    -> {status} | tempo={settle_str} | "
                f"runaway={trial.runaway_events} | entradas={trial.center_entries} | "
                f"raio_final={radius_str} | motivo={trial.reason}"
            )

            if incumbent is not None:
                success_so_far = sum(1 for item in trials if item.success)
                remaining = total_expected - len(trials)
                max_possible_success = success_so_far + remaining
                partial_score = _partial_trial_score(trials)
                if max_possible_success < incumbent.success_count:
                    print(
                        "    -> Abortando cedo: este candidato nao consegue mais "
                        "igualar o numero de sucessos do melhor atual."
                    )
                    abort_candidate = True
                    break
                if (
                    max_possible_success == incumbent.success_count
                    and partial_score >= incumbent.score
                ):
                    print(
                        "    -> Abortando cedo: mesmo com sucesso nos testes restantes "
                        "o score nao superaria o melhor atual."
                    )
                    abort_candidate = True
                    break

            if (
                len(trials) >= 2
                and sum(1 for item in trials if item.success) == 0
                and all(
                    (not item.success)
                    and item.reason == "early_unstable_abort"
                    and item.center_entries == 0
                    for item in trials[:2]
                )
            ):
                print(
                    "    -> Abortando cedo: duas falhas instaveis seguidas "
                    "logo no inicio indicam candidato ruim para a malha rapida."
                )
                abort_candidate = True
                break

        if abort_candidate or (trials and not trials[-1].success):
            break

    (
        score,
        success_count,
        median_time,
        mean_runaway,
        mean_center_entries,
        mean_final_radius,
    ) = _score_trials(trials, total_expected)
    median_str = f"{median_time:.3f}s" if median_time is not None else "n/a"
    print(
        f"  Score={score:.3f} | sucessos={success_count}/{total_expected} | "
        f"mediana={median_str} | runaway_medio={mean_runaway:.2f} | "
        f"entradas_medias={mean_center_entries:.2f} | raio_final_medio={mean_final_radius:.2f}px"
    )
    return CandidateEvaluation(
        kp=params.kp,
        trim_gain=params.trim_gain,
        kd=params.kd,
        score=score,
        success_count=success_count,
        total_tests=total_expected,
        median_time_s=median_time,
        mean_runaway_events=mean_runaway,
        mean_center_entries=mean_center_entries,
        mean_final_radius_px=mean_final_radius,
        trials=trials,
    )


def twiddle_search(A_inv: np.ndarray, usar_mount: bool):
    current = _clip_params(TunedParams(INITIAL_KP, INITIAL_TRIM_GAIN, INITIAL_KD))
    dp = {"kp": INITIAL_DP_KP, "trim_gain": INITIAL_DP_TRIM_GAIN, "kd": INITIAL_DP_KD}
    history: list[CandidateEvaluation] = []

    best_eval = evaluate_candidate(A_inv, usar_mount, current)
    history.append(best_eval)

    for iteration in range(1, TWIDDLE_MAX_ITERS + 1):
        if (dp["kp"] + dp["trim_gain"] + dp["kd"]) < TWIDDLE_MIN_DP_SUM:
            print("\nParando: soma dos passos do twiddle ficou pequena o suficiente.")
            break

        print("\n" + "=" * 72)
        print(
            f"Iteracao {iteration}/{TWIDDLE_MAX_ITERS} | "
            f"best Kp={current.kp:.4f} Trim={current.trim_gain:.4f} Kd={current.kd:.4f} | "
            f"dp_kp={dp['kp']:.4f} dp_trim={dp['trim_gain']:.4f} dp_kd={dp['kd']:.4f}"
        )
        print("=" * 72)

        for field_name in ("kp", "kd", "trim_gain"):
            improved = False

            trial_up = _clip_params(
                TunedParams(
                    kp=current.kp + (dp[field_name] if field_name == "kp" else 0.0),
                    trim_gain=current.trim_gain + (dp[field_name] if field_name == "trim_gain" else 0.0),
                    kd=current.kd + (dp[field_name] if field_name == "kd" else 0.0),
                )
            )
            if not _params_equal(trial_up, current):
                eval_up = evaluate_candidate(A_inv, usar_mount, trial_up, incumbent=best_eval)
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
                    trim_gain=current.trim_gain - (dp[field_name] if field_name == "trim_gain" else 0.0),
                    kd=current.kd - (dp[field_name] if field_name == "kd" else 0.0),
                )
            )
            if not _params_equal(trial_down, current):
                eval_down = evaluate_candidate(A_inv, usar_mount, trial_down, incumbent=best_eval)
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
        key = (
            round(evaluation.kp, 6),
            round(evaluation.trim_gain, 6),
            round(evaluation.kd, 6),
        )
        existing = unique.get(key)
        if existing is None or evaluation.score < existing.score:
            unique[key] = evaluation

    ranking = sorted(
        unique.values(),
        key=lambda item: (
            -item.success_count,
            item.score,
            item.median_time_s if item.median_time_s is not None else float("inf"),
            item.mean_runaway_events,
            item.mean_center_entries,
        ),
    )

    print("\n=== Podio Autotune PD+Trim Tracker ===")
    for idx, item in enumerate(ranking[:5], start=1):
        median_str = f"{item.median_time_s:.3f}s" if item.median_time_s is not None else "n/a"
        print(
            f"{idx}. Kp={item.kp:.4f} Trim={item.trim_gain:.4f} Kd={item.kd:.4f} | "
            f"sucessos={item.success_count}/{item.total_tests} | "
            f"score={item.score:.3f} | mediana={median_str} | "
            f"runaway_medio={item.mean_runaway_events:.2f} | "
            f"entradas_medias={item.mean_center_entries:.2f}"
        )


def save_results(path: str, best_eval: CandidateEvaluation, history: list[CandidateEvaluation]):
    payload = {
        "timestamp_epoch": time.time(),
        "config": {
            "window_size": WINDOW_SIZE,
            "exposure_seconds": EXPOSURE_SECONDS,
            "tolerancia_px": TOLERANCIA_PX,
            "recenter_settle_s": RECENTER_SETTLE_S,
            "tracker_timeout_s": TRACKER_TIMEOUT_S,
            "safe_max_delta_deg": SAFE_MAX_DELTA_DEG,
            "safe_az_min_deg": SAFE_AZ_MIN_DEG,
            "safe_az_max_deg": SAFE_AZ_MAX_DEG,
            "safe_alt_min_deg": SAFE_ALT_MIN_DEG,
            "safe_alt_max_deg": SAFE_ALT_MAX_DEG,
            "safe_max_recenter_delta_deg": SAFE_MAX_RECENTER_DELTA_DEG,
            "repeat_count": REPEAT_COUNT,
            "twiddle_max_iters": TWIDDLE_MAX_ITERS,
            "runaway_penalty": RUNAWAY_PENALTY,
            "reentry_penalty": REENTRY_PENALTY,
            "first_reentry_weight": FIRST_REENTRY_WEIGHT,
            "signal_loss_penalty": SIGNAL_LOSS_PENALTY,
            "initial_kp": INITIAL_KP,
            "initial_trim_gain": INITIAL_TRIM_GAIN,
            "initial_kd": INITIAL_KD,
            "initial_dp_kp": INITIAL_DP_KP,
            "initial_dp_trim_gain": INITIAL_DP_TRIM_GAIN,
            "initial_dp_kd": INITIAL_DP_KD,
            "trim_limit": TRIM_LIMIT,
            "trim_leak": TRIM_LEAK,
            "trim_error_max_deg": TRIM_ERROR_MAX_DEG,
            "trim_derivative_max_deg_s": TRIM_DERIVATIVE_MAX_DEG_S,
            "trim_same_sign_s": TRIM_SAME_SIGN_S,
            "trim_enter_radius_px": TRIM_ENTER_RADIUS_PX,
            "trim_exit_radius_px": TRIM_EXIT_RADIUS_PX,
            "perturbations": [asdict(p) for p in PERTURBATIONS],
        },
        "best": {
            "kp": best_eval.kp,
            "trim_gain": best_eval.trim_gain,
            "kd": best_eval.kd,
            "score": best_eval.score,
            "success_count": best_eval.success_count,
            "total_tests": best_eval.total_tests,
            "median_time_s": best_eval.median_time_s,
            "mean_runaway_events": best_eval.mean_runaway_events,
            "mean_center_entries": best_eval.mean_center_entries,
            "mean_final_radius_px": best_eval.mean_final_radius_px,
        },
        "history": [
            {
                "kp": evaluation.kp,
                "trim_gain": evaluation.trim_gain,
                "kd": evaluation.kd,
                "score": evaluation.score,
                "success_count": evaluation.success_count,
                "total_tests": evaluation.total_tests,
                "median_time_s": evaluation.median_time_s,
                "mean_runaway_events": evaluation.mean_runaway_events,
                "mean_center_entries": evaluation.mean_center_entries,
                "mean_final_radius_px": evaluation.mean_final_radius_px,
                "trials": [asdict(trial) for trial in evaluation.trials],
            }
            for evaluation in history
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    _validate_perturbations()
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()
    connect_camera()

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

        print("=== Autotune Seguro do Tracker Continuo ===")
        print("Busca local tipo twiddle em Kp/Trim/Kd, com recentralizacao e perturbacoes seguras.\n")
        print(f"Usando matriz de tracking: {source}\n")

        best_eval, history = twiddle_search(A_inv, usar_mount)
        print_podium(history)
        save_results(RESULTS_JSON, best_eval, history)
        print(f"\nResultados salvos em {display_path(RESULTS_JSON)}")

    except KeyboardInterrupt:
        print("\nAutotune do tracker interrompido pelo usuario.")
    finally:
        try:
            if "usar_mount" in locals():
                move_axis(0, 0.0, usar_mount)
                move_axis(1, 0.0, usar_mount)
        except Exception:
            pass
        try:
            reset_camera_roi()
        except Exception:
            pass
        disconnect_camera()


if __name__ == "__main__":
    main()
