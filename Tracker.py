import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

cv2.setUseOptimized(True)

from artifact_paths import display_path, matrix_candidates
from PID_controll import ensure_connected, ensure_not_tracking, ensure_unparked
from mov_simultaneo import VEL_MAX_LIMITE, VEL_MIN_LIMITE, move_axis

# ==== Configuracoes Alpaca da camera ====
BASE_URL = "http://127.0.0.1:11111/api/v1/camera/0"
CLIENT_ID = 1
IMAGE_READY_POLL_S = 0.001
IMAGE_READY_SPIN_POLLS = 3
_transaction_ids = itertools.count(1)
session = requests.Session()

# ===== Parametros do tracker continuo =====
WINDOW_SIZE = 200
TARGET_H = 1080
DISPLAY_HZ = 6.0
TOLERANCIA_PX = 2.0
CONTROL_DEADBAND_PX = 0.35
RECENTER_SETTLE_S = 1.0
EXPOSURE_SECONDS = 32e-6
CONTROL_HZ = 45.0
SIGNAL_TIMEOUT_S = 0.45
VEL_MAX_TESTE = min(1.6, VEL_MAX_LIMITE)
FINE_MATRIX_ENTER_RADIUS_PX = 8.0
FINE_MATRIX_EXIT_RADIUS_PX = 14.0

# Suavizacao das medicoes e envio dos comandos.
MEASUREMENT_ALPHA = 0.70
CMD_ACCEL_LIMIT = 2.00
CMD_KEEPALIVE_S = 0.15
MIN_CMD_DELTA_TO_SEND = 2e-4
CMD_ZERO_SNAP = 0.35 * VEL_MIN_LIMITE

# Ganhos da malha rapida PD.
KP_AZ = 1.4400
KP_ALT = 1.4400
KD_AZ = 0.1800
KD_ALT = 0.1800
DERIVATIVE_ALPHA = 0.70

# "Trim" lento para viés persistente perto do centro. Ele substitui o Ki classico:
# so entra quando o erro permanece com o mesmo sinal por algum tempo e a malha ja
# esta em regime fino, evitando contaminar a resposta rapida.
TRIM_GAIN_AZ = 1.2
TRIM_GAIN_ALT = 1.2
TRIM_LIMIT = 0.020
TRIM_LEAK = 0.985
TRIM_ERROR_MAX_DEG = 0.0006
TRIM_DERIVATIVE_MAX_DEG_S = 0.006
TRIM_SAME_SIGN_S = 0.80
TRIM_SIGN_EPS_DEG = 0.00010
TRIM_SIGN_FLIP_DAMP = 0.35
TRIM_ENTER_RADIUS_PX = 1.3
TRIM_EXIT_RADIUS_PX = 2.2

# Freio simples se o erro cresce em varios frames seguidos.
ENABLE_RUNAWAY_BRAKE = True
RUNAWAY_MARGIN_PX = 1.0
RUNAWAY_FRAMES = 4
RUNAWAY_HOLD_S = 0.40
RUNAWAY_LOG_COOLDOWN_S = 2.0
ENABLE_MANUAL_JUMP_BRAKE = True
MANUAL_JUMP_PX = 18.0
MANUAL_JUMP_HOLD_S = 0.25


def call(method: str, command: str, timeout: float = 5.0, **extra_args):
    params = {
        "ClientID": CLIENT_ID,
        "ClientTransactionID": next(_transaction_ids),
    }
    params.update(extra_args.pop("params", {}))
    resp = session.request(
        method,
        f"{BASE_URL}/{command}",
        params=params,
        timeout=timeout,
        **extra_args,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("ErrorNumber", 0):
        raise RuntimeError(f"{command}: {payload.get('ErrorMessage')}")
    return payload.get("Value")


def set_camera_roi(w: int, h: int) -> None:
    try:
        max_x = int(call("GET", "cameraxsize"))
        max_y = int(call("GET", "cameraysize"))
        start_x = int((max_x / 2) - (w / 2))
        start_y = int((max_y / 2) - (h / 2))
        print(f"Cortando o sensor na fonte (Hardware ROI central): {w}x{h} px...")
        call("PUT", "startx", data={"StartX": start_x})
        call("PUT", "starty", data={"StartY": start_y})
        call("PUT", "numx", data={"NumX": w})
        call("PUT", "numy", data={"NumY": h})
    except Exception as exc:
        print(f"Erro ao setar ROI via hardware: {exc}")


def reset_camera_roi() -> None:
    try:
        max_x = call("GET", "cameraxsize")
        max_y = call("GET", "cameraysize")
        call("PUT", "startx", data={"StartX": 0})
        call("PUT", "starty", data={"StartY": 0})
        call("PUT", "numx", data={"NumX": max_x})
        call("PUT", "numy", data={"NumY": max_y})
    except Exception:
        pass


def connect_camera() -> None:
    print("Conectando à câmera...")
    call("PUT", "connected", data={"Connected": True})


def disconnect_camera() -> None:
    print("Desconectando da câmera...")
    call("PUT", "connected", data={"Connected": False})


def start_exposure(duration_seconds: float, light: bool = True) -> None:
    call("PUT", "startexposure", data={"Duration": duration_seconds, "Light": light})


def wait_until_image_ready(
    poll_interval: float = IMAGE_READY_POLL_S,
    timeout: float = 5.0,
) -> None:
    deadline = time.time() + timeout
    spin_polls = IMAGE_READY_SPIN_POLLS
    while time.time() < deadline:
        ready = bool(call("GET", "imageready"))
        if ready:
            return
        if spin_polls > 0:
            spin_polls -= 1
            continue
        time.sleep(poll_interval)
    raise TimeoutError("Tempo limite esperando ImageReady = True")


def fetch_image_array() -> np.ndarray:
    payload = call("GET", "imagearray")
    return np.asarray(payload)


def capture_frame(exposure_seconds: float) -> np.ndarray:
    start_exposure(exposure_seconds, light=True)
    wait_until_image_ready()
    frame = fetch_image_array().astype(np.float32)

    pedestal = np.median(frame) + (0.5 * np.std(frame))
    max_val = frame.max()
    if max_val <= pedestal:
        return np.zeros_like(frame, dtype=np.uint8)

    norm = np.clip((frame - pedestal) / (max_val - pedestal + 1e-6), 0, 1)
    norm = (norm * 255).astype(np.uint8)
    return np.rot90(norm, 2)


def calcular_cm_corrigido(frame_window: np.ndarray, threshold_percent: float = 0.5):
    if frame_window.ndim == 3:
        frame_gray = frame_window.mean(axis=2)
    else:
        frame_gray = frame_window

    max_val = float(frame_gray.max())
    if max_val < 100:
        return None

    dynamic_threshold = max_val * threshold_percent
    _, weights = cv2.threshold(
        frame_gray.astype(np.float32, copy=False),
        dynamic_threshold,
        0,
        cv2.THRESH_TOZERO,
    )
    moments = cv2.moments(weights, binaryImage=False)
    total_intensidade = moments["m00"]
    if total_intensidade <= 0:
        return None

    x_cm = moments["m10"] / total_intensidade
    y_cm = moments["m01"] / total_intensidade
    return x_cm, y_cm


class MeasurementPDTrim:
    """PD rapido com trim lento para erro persistente perto do centro."""

    def __init__(
        self,
        kp,
        kd,
        trim_gain,
        output_limits,
        derivative_alpha=0.70,
        trim_limit=0.18,
        trim_leak=0.995,
        trim_error_max=0.0025,
        trim_derivative_max=0.03,
        trim_same_sign_s=0.25,
        trim_sign_eps=0.00015,
        trim_sign_flip_damp=0.35,
    ):
        self.kp = kp
        self.kd = kd
        self.trim_gain = trim_gain
        self.min_output, self.max_output = output_limits
        self.derivative_alpha = derivative_alpha
        self.trim_limit = abs(float(trim_limit))
        self.trim_leak = float(trim_leak)
        self.trim_error_max = abs(float(trim_error_max))
        self.trim_derivative_max = abs(float(trim_derivative_max))
        self.trim_same_sign_s = float(trim_same_sign_s)
        self.trim_sign_eps = abs(float(trim_sign_eps))
        self.trim_sign_flip_damp = float(trim_sign_flip_damp)
        self.reset()

    def reset(self):
        self._last_error = None
        self._last_t = None
        self._d_filt = 0.0
        self.clear_trim()

    def clear_trim(self):
        self._trim_bias = 0.0
        self._held_sign = 0
        self._same_sign_elapsed = 0.0

    def _clip(self, value):
        if self.min_output is not None and value < self.min_output:
            return self.min_output
        if self.max_output is not None and value > self.max_output:
            return self.max_output
        return value

    def update(self, error, timestamp, trim_allowed):
        if self._last_t is None or timestamp <= self._last_t:
            self._last_error = error
            self._last_t = timestamp
            self._d_filt = 0.0
            self.clear_trim()
            return self._clip(self.kp * error), 0.0

        dt = max(timestamp - self._last_t, 1e-3)
        deriv = (error - self._last_error) / dt
        self._d_filt = (self.derivative_alpha * self._d_filt) + ((1.0 - self.derivative_alpha) * deriv)

        if abs(error) < self.trim_sign_eps:
            sign = 0
        else:
            sign = 1 if error > 0.0 else -1

        if sign == 0:
            self._held_sign = 0
            self._same_sign_elapsed = 0.0
            self._trim_bias *= self.trim_leak
        else:
            if sign == self._held_sign:
                self._same_sign_elapsed += dt
            else:
                if self._held_sign != 0:
                    self._trim_bias *= self.trim_sign_flip_damp
                self._held_sign = sign
                self._same_sign_elapsed = 0.0

            trim_ready = (
                trim_allowed
                and self._same_sign_elapsed >= self.trim_same_sign_s
                and abs(error) <= self.trim_error_max
                and abs(self._d_filt) <= self.trim_derivative_max
            )
            if trim_ready:
                self._trim_bias += self.trim_gain * error * dt
                self._trim_bias = float(
                    np.clip(
                        self._trim_bias,
                        -self.trim_limit,
                        self.trim_limit,
                    )
                )
            else:
                self._trim_bias *= self.trim_leak

        output_raw = (self.kp * error) + (self.kd * self._d_filt) + self._trim_bias
        output = self._clip(output_raw)
        self._last_error = error
        self._last_t = timestamp
        return output, float(self._trim_bias)


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
    measurement_age_s: float = 0.0
    runaway_events: int = 0
    brake_active: bool = False
    active_matrix_name: str = "coarse"
    trim_mode_active: bool = False
    trim_az_deg_s: float = 0.0
    trim_alt_deg_s: float = 0.0


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


def _normalize_focus_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"2", "dual", "duplo", "dois", "two"}:
        return "dual"
    return "single"


def _load_tracking_calibration_matrices(focus_mode: str):
    focus_mode = _normalize_focus_mode(focus_mode)
    if focus_mode == "dual":
        fine_candidates = matrix_candidates("foco_temp_A_inv_fine.npy")
        coarse_candidates = matrix_candidates("foco_temp_A_inv_coarse.npy")
    else:
        fine_candidates = matrix_candidates(
            "A_inv_fine.npy",
            "calibracao_dual_v3_fine_A_inv.npy",
            "calibracao_A_inv.npy",
        )
        coarse_candidates = matrix_candidates(
            "A_inv_coarse.npy",
            "calibracao_dual_v3_coarse_A_inv.npy",
            "calibracao_A_inv.npy",
        )

    def _load_first_existing(candidates, label):
        for path in candidates:
            try:
                matrix = np.load(path)
            except FileNotFoundError:
                continue
            if matrix.shape != (2, 2):
                raise ValueError(f"Matriz {path} para {label} precisa ser 2x2")
            return matrix, path
        raise FileNotFoundError(
            f"Nao encontrei matriz {label}. Testei: {', '.join(str(path) for path in candidates)}"
        )

    fine_matrix, fine_path = _load_first_existing(fine_candidates, "fine")
    coarse_matrix, coarse_path = _load_first_existing(coarse_candidates, "coarse")
    return {
        "fine": fine_matrix,
        "coarse": coarse_matrix,
        "fine_path": display_path(fine_path),
        "coarse_path": display_path(coarse_path),
        "focus_mode": focus_mode,
    }


def _draw_text_with_outline(
    image,
    text,
    origin,
    font_scale,
    color,
    thickness=2,
    outline_color=(0, 0, 0),
):
    outline_thickness = thickness + 3
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_badge(image, text, top_right, fg_color, bg_color, font_scale=0.9, thickness=2):
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    pad_x = 16
    pad_y = 12
    x2, y1 = top_right
    x1 = x2 - text_w - (2 * pad_x)
    y2 = y1 + text_h + baseline + (2 * pad_y)
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), fg_color, 2)
    _draw_text_with_outline(
        image,
        text,
        (x1 + pad_x, y2 - baseline - pad_y),
        font_scale,
        fg_color,
        thickness=thickness,
        outline_color=(0, 0, 0),
    )


def control_loop_continuo(
    state: SharedState,
    A_inv_fine: np.ndarray,
    A_inv_coarse: np.ndarray,
    usar_mount: bool,
):
    ctrl_az = MeasurementPDTrim(
        kp=KP_AZ,
        kd=KD_AZ,
        trim_gain=TRIM_GAIN_AZ,
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
        kp=KP_ALT,
        kd=KD_ALT,
        trim_gain=TRIM_GAIN_ALT,
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
    prev_dx_filt_px = None
    prev_dy_filt_px = None
    runaway_count = 0
    brake_until = 0.0
    last_runaway_log_t = 0.0
    active_matrix_name = "coarse"
    active_matrix = A_inv_coarse
    trim_mode_active = False
    trim_az = 0.0
    trim_alt = 0.0

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
                    prev_dx_filt_px = None
                    prev_dy_filt_px = None
                    runaway_count = 0
                    trim_mode_active = False
                    trim_az = 0.0
                    trim_alt = 0.0
                elif seq != last_seq:
                    last_seq = seq
                    radius_px = float(np.hypot(dx_filt, dy_filt))
                    manual_jump = False
                    if (
                        ENABLE_MANUAL_JUMP_BRAKE
                        and prev_dx_filt_px is not None
                        and prev_dy_filt_px is not None
                    ):
                        jump_px = float(
                            np.hypot(
                                dx_filt - prev_dx_filt_px,
                                dy_filt - prev_dy_filt_px,
                            )
                        )
                        manual_jump = (
                            jump_px >= MANUAL_JUMP_PX
                            and radius_px > (2.0 * TOLERANCIA_PX)
                        )

                    prev_dx_filt_px = dx_filt
                    prev_dy_filt_px = dy_filt

                    if manual_jump:
                        target_cmd_az = 0.0
                        target_cmd_alt = 0.0
                        cmd_az = 0.0
                        cmd_alt = 0.0
                        err_az = 0.0
                        err_alt = 0.0
                        ctrl_az.reset()
                        ctrl_alt.reset()
                        trim_mode_active = False
                        trim_az = 0.0
                        trim_alt = 0.0
                        prev_radius_px = radius_px
                        runaway_count = 0
                        brake_until = loop_t0 + MANUAL_JUMP_HOLD_S
                        if (loop_t0 - last_runaway_log_t) >= RUNAWAY_LOG_COOLDOWN_S:
                            print(
                                "\nMovimento manual brusco detectado. "
                                "Zerando o controle por um instante antes de recentralizar."
                            )
                            last_runaway_log_t = loop_t0
                        with state.lock:
                            state.runaway_events += 1
                    else:
                        previous_matrix_name = active_matrix_name

                        if active_matrix_name == "coarse" and radius_px <= FINE_MATRIX_ENTER_RADIUS_PX:
                            active_matrix_name = "fine"
                            active_matrix = A_inv_fine
                        elif active_matrix_name == "fine" and radius_px >= FINE_MATRIX_EXIT_RADIUS_PX:
                            active_matrix_name = "coarse"
                            active_matrix = A_inv_coarse

                        if active_matrix_name != previous_matrix_name:
                            ctrl_az.reset()
                            ctrl_alt.reset()
                            trim_mode_active = False
                            trim_az = 0.0
                            trim_alt = 0.0

                        if (
                            active_matrix_name == "fine"
                            and radius_px <= TRIM_ENTER_RADIUS_PX
                        ):
                            trim_mode_active = True
                        elif (
                            active_matrix_name != "fine"
                            or radius_px >= TRIM_EXIT_RADIUS_PX
                        ):
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
                            trim_az = 0.0
                            trim_alt = 0.0
                        else:
                            err_az, err_alt = _pixel_error_to_mount_error(dx_filt, dy_filt, active_matrix)

                        trim_allowed = trim_mode_active and (loop_t0 >= brake_until)
                        target_cmd_az, trim_az = ctrl_az.update(err_az, measurement_ts, trim_allowed)
                        target_cmd_alt, trim_alt = ctrl_alt.update(err_alt, measurement_ts, trim_allowed)

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
                                if (loop_t0 - last_runaway_log_t) >= RUNAWAY_LOG_COOLDOWN_S:
                                    print(
                                        "\n⚠️ Erro aumentou em varios frames seguidos. "
                                        "Freando o mount. Isso pode acontecer se o spot/camera "
                                        "for movido manualmente ou se houver sinal/eixo invertido."
                                    )
                                    last_runaway_log_t = loop_t0
                                target_cmd_az = 0.0
                                target_cmd_alt = 0.0
                                cmd_az = 0.0
                                cmd_alt = 0.0
                                ctrl_az.reset()
                                ctrl_alt.reset()
                                trim_mode_active = False
                                trim_az = 0.0
                                trim_alt = 0.0
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
                    state.measurement_age_s = measurement_age
                    state.brake_active = loop_t0 < brake_until
                    state.active_matrix_name = active_matrix_name
                    state.trim_mode_active = trim_mode_active
                    state.trim_az_deg_s = trim_az
                    state.trim_alt_deg_s = trim_alt

                elapsed = time.perf_counter() - loop_t0
                if elapsed < dt_target:
                    time.sleep(dt_target - elapsed)

        finally:
            move_axis(0, 0.0, usar_mount)
            move_axis(1, 0.0, usar_mount)


def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()
    connect_camera()

    try:
        focus_input = input("Modo do laser (1=foco unico, 2=dupla reflexao) [1]: ").strip() or "1"
        focus_mode = _normalize_focus_mode(focus_input)
        matrices = _load_tracking_calibration_matrices(focus_mode)

        state = SharedState()
        usar_mount = True

        ctrl_thread = threading.Thread(
            target=control_loop_continuo,
            args=(state, matrices["fine"], matrices["coarse"], usar_mount),
            daemon=True,
        )
        ctrl_thread.start()

        win_name = "Tracker 4QD Continuo - V2"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("\nTracker continuo V2 iniciado.")
        print("ROI nativa da camera, PD rapido + trim lento e freio basico de runaway.")
        print(f"Modo do laser: {focus_mode} | usando mount real.")
        print(
            f"Matrizes carregadas | fine: {matrices['fine_path']} | "
            f"coarse: {matrices['coarse_path']}"
        )
        print("Pressione q para encerrar.\n")

        set_camera_roi(WINDOW_SIZE, WINDOW_SIZE)

        t_prev = time.perf_counter()
        tempos_loop = []
        fps_loop = 0.0
        fps_ui = 0.0
        last_capture_ms = 0.0
        last_cm_ms = 0.0
        last_ui_ms = 0.0
        recenter_timer_start = None
        recenter_center_hold_start = None
        recenter_attempt_idx = 0
        last_recenter_elapsed = None
        scale_upscale = TARGET_H / WINDOW_SIZE
        target_w = int(WINDOW_SIZE * scale_upscale)
        cx_L = int((WINDOW_SIZE / 2) * scale_upscale)
        cy_L = int((WINDOW_SIZE / 2) * scale_upscale)
        display_interval_s = 1.0 / DISPLAY_HZ
        last_display_t = 0.0
        display_frames = 0
        display_window_start = time.perf_counter()

        while True:
            t_inicio_loop = time.perf_counter()
            t_capture0 = time.perf_counter()
            frame_window = capture_frame(EXPOSURE_SECONDS)
            t_after_capture = time.perf_counter()
            last_capture_ms = (t_after_capture - t_capture0) * 1000.0

            t_now = t_after_capture
            dt = max(t_now - t_prev, 1e-6)
            fps_cam = 1.0 / dt
            t_prev = t_now

            t_cm0 = time.perf_counter()
            cm = calcular_cm_corrigido(frame_window)
            last_cm_ms = (time.perf_counter() - t_cm0) * 1000.0

            if cm is None:
                dx = 0.0
                dy = 0.0
                x_cm_local = WINDOW_SIZE / 2
                y_cm_local = WINDOW_SIZE / 2
                cor_laser = (0, 0, 255)

                with state.lock:
                    state.has_signal = False
                    state.measurement_seq += 1
                    state.measurement_ts = t_now
                    meas_age = state.measurement_age_s
                    brake_active = state.brake_active
                    active_matrix_name = state.active_matrix_name
                    trim_mode_active = state.trim_mode_active
                    trim_az_deg_s = state.trim_az_deg_s
                    trim_alt_deg_s = state.trim_alt_deg_s
                    err_az_deg = state.err_az_deg
                    err_alt_deg = state.err_alt_deg
                    cmd_az_deg_s = state.cmd_az_deg_s
                    cmd_alt_deg_s = state.cmd_alt_deg_s

                if recenter_timer_start is not None:
                    print("\n⏱️ Recenter cancelado: sinal perdido antes de voltar ao centro.")
                    recenter_timer_start = None
                    recenter_center_hold_start = None
            else:
                x_cm_local, y_cm_local = cm
                dx = float(x_cm_local - (WINDOW_SIZE / 2))
                dy = float(y_cm_local - (WINDOW_SIZE / 2))

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
                    meas_age = state.measurement_age_s
                    brake_active = state.brake_active
                    active_matrix_name = state.active_matrix_name
                    trim_mode_active = state.trim_mode_active
                    trim_az_deg_s = state.trim_az_deg_s
                    trim_alt_deg_s = state.trim_alt_deg_s
                    err_az_deg = state.err_az_deg
                    err_alt_deg = state.err_alt_deg
                    cmd_az_deg_s = state.cmd_az_deg_s
                    cmd_alt_deg_s = state.cmd_alt_deg_s

                cor_laser = (0, 255, 255)

                is_centered_now = (abs(dx) < TOLERANCIA_PX) and (abs(dy) < TOLERANCIA_PX)
                if not is_centered_now:
                    if recenter_timer_start is None:
                        recenter_attempt_idx += 1
                        recenter_timer_start = t_now
                        print(
                            f"\n⏱️ Recenter {recenter_attempt_idx} iniciado: "
                            f"dx={dx:+.1f}px dy={dy:+.1f}px"
                        )
                    recenter_center_hold_start = None
                elif recenter_timer_start is not None:
                    if recenter_center_hold_start is None:
                        recenter_center_hold_start = t_now
                    elif (t_now - recenter_center_hold_start) >= RECENTER_SETTLE_S:
                        recenter_elapsed = t_now - recenter_timer_start
                        last_recenter_elapsed = recenter_elapsed
                        print(
                            f"\n⏱️ Recenter {recenter_attempt_idx} concluido em "
                            f"{recenter_elapsed:.3f}s "
                            f"(estavel por {RECENTER_SETTLE_S:.1f}s, "
                            f"dx={dx:+.1f}px dy={dy:+.1f}px)"
                        )
                        recenter_timer_start = None
                        recenter_center_hold_start = None

            dist_px = float(np.hypot(dx, dy))
            signal_ok = cm is not None
            if not signal_ok:
                status_text = "NO SIGNAL"
                status_fg = (255, 255, 255)
                status_bg = (0, 0, 180)
            elif brake_active:
                status_text = "BRAKE"
                status_fg = (255, 255, 255)
                status_bg = (0, 0, 200)
            elif dist_px <= TOLERANCIA_PX:
                status_text = "LOCKED"
                status_fg = (20, 20, 20)
                status_bg = (0, 255, 140)
            else:
                status_text = "TRACKING"
                status_fg = (20, 20, 20)
                status_bg = (0, 255, 255)

            tempos_loop.append(time.perf_counter() - t_inicio_loop)
            if len(tempos_loop) >= 10:
                media_dt = sum(tempos_loop) / len(tempos_loop)
                fps_loop = 1.0 / media_dt if media_dt > 0 else 0.0
                tempos_loop.clear()

            refresh_display = (
                last_display_t == 0.0
                or (t_now - last_display_t) >= display_interval_s
            )
            if refresh_display:
                t_ui0 = time.perf_counter()
                frame_display = cv2.cvtColor(frame_window, cv2.COLOR_GRAY2BGR)
                frame_display_large = cv2.resize(
                    frame_display,
                    (target_w, TARGET_H),
                    interpolation=cv2.INTER_NEAREST,
                )

                x_cm_L = int(x_cm_local * scale_upscale)
                y_cm_L = int(y_cm_local * scale_upscale)

                cv2.line(frame_display_large, (cx_L, 0), (cx_L, TARGET_H), (255, 0, 0), 2)
                cv2.line(frame_display_large, (0, cy_L), (target_w, cy_L), (0, 0, 255), 2)
                cv2.circle(frame_display_large, (x_cm_L, y_cm_L), 8, cor_laser, -1)

                _draw_text_with_outline(
                    frame_display_large,
                    f"Dist={dist_px:.1f}px  dx={dx:+.1f}  dy={dy:+.1f}",
                    (40, 60),
                    1.1,
                    (0, 255, 0),
                    thickness=2,
                )
                _draw_text_with_outline(
                    frame_display_large,
                    (
                        f"Cam={fps_cam:.1f}Hz  Loop={fps_loop:.1f}Hz  "
                        f"UI={fps_ui:.1f}Hz  Age={meas_age*1000:.0f}ms  "
                        f"Focus={focus_mode}  Map={active_matrix_name}  "
                        f"Trim={'ON' if trim_mode_active else 'OFF'}"
                    ),
                    (40, 110),
                    1.0,
                    (0, 255, 255),
                    thickness=2,
                )
                _draw_text_with_outline(
                    frame_display_large,
                    (
                        f"Err=({err_az_deg:+.4f}, {err_alt_deg:+.4f})deg  "
                        f"Cmd=({cmd_az_deg_s:+.3f}, {cmd_alt_deg_s:+.3f})deg/s  "
                        f"Bias=({trim_az_deg_s:+.3f}, {trim_alt_deg_s:+.3f})deg/s"
                    ),
                    (40, 160),
                    0.95,
                    (255, 255, 255),
                    thickness=2,
                )
                _draw_text_with_outline(
                    frame_display_large,
                    (
                        f"Timing: cap={last_capture_ms:.1f}ms  "
                        f"CM={last_cm_ms:.1f}ms  UI={last_ui_ms:.1f}ms"
                    ),
                    (40, 205),
                    0.9,
                    (200, 255, 200),
                    thickness=2,
                )
                if recenter_timer_start is not None:
                    recenter_text = f"Recentering {t_now - recenter_timer_start:.2f}s"
                elif last_recenter_elapsed is not None:
                    recenter_text = f"Last recenter {last_recenter_elapsed:.2f}s"
                else:
                    recenter_text = f"Tolerance {TOLERANCIA_PX:.1f}px"
                _draw_text_with_outline(
                    frame_display_large,
                    recenter_text,
                    (40, TARGET_H - 35),
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                )

                _draw_badge(
                    frame_display_large,
                    status_text,
                    (target_w - 30, 30),
                    status_fg,
                    status_bg,
                    font_scale=0.95,
                    thickness=2,
                )

                cv2.imshow(win_name, frame_display_large)
                last_ui_ms = (time.perf_counter() - t_ui0) * 1000.0
                last_display_t = t_now
                display_frames += 1
                display_elapsed = t_now - display_window_start
                if display_elapsed >= 1.0:
                    fps_ui = display_frames / display_elapsed
                    display_frames = 0
                    display_window_start = t_now

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
    except Exception as exc:
        print(f"\nErro no tracker continuo V2: {exc}")
    finally:
        if "state" in locals():
            with state.lock:
                state.stop = True
            if "ctrl_thread" in locals() and ctrl_thread.is_alive():
                ctrl_thread.join(timeout=2.0)

        try:
            if "usar_mount" in locals():
                move_axis(0, 0.0, usar_mount)
                move_axis(1, 0.0, usar_mount)
        except Exception:
            pass

        reset_camera_roi()
        disconnect_camera()
        cv2.destroyAllWindows()
        print("Controle encerrado com parada segura.")


if __name__ == "__main__":
    main()
