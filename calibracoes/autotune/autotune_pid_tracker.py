import itertools
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import display_path, json_output_path, matrix_candidates
from controle.Center_of_Mass import centro_camera, centro_massa as centro_massa_full_frame
from controle.Tracker import (
    MeasurementPDTrim,
    calcular_cm_corrigido,
    capture_frame,
    connect_camera,
    disconnect_camera,
    reset_camera_roi,
    set_camera_roi,
)
from controle.mount_control import VEL_MAX_LIMITE, VEL_MIN_LIMITE

try:
    from foco_multiplos import Center_of_Mass_foco_temp as foco_temp
except Exception:
    foco_temp = None


# ===== Alpaca / ASCOM devices =====
DEFAULT_RECEIVER_DEVICE = 0  # "telescopio 1": recebe o laser e corrige.
DEFAULT_SENDER_DEVICE = 1  # "telescopio 2": envia o laser e gera perturbacoes.
DEFAULT_RECEIVER_ALPACA_ROOT = "http://127.0.0.1:11111/api/v1"
DEFAULT_SENDER_ALPACA_ROOT = "http://127.0.0.1:11111/api/v1"
DEFAULT_SENDER_AGENT_URL = "http://10.4.0.145:18080"
DEFAULT_RECEIVER_INVERT_AZ = True
DEFAULT_RECEIVER_INVERT_ALT = False
DEFAULT_SENDER_INVERT_AZ = True
DEFAULT_SENDER_INVERT_ALT = False
CLIENT_ID = 1


# ===== Tracker/autotune configuration =====
WINDOW_SIZE = 200
EXPOSURE_SECONDS = 32e-6
TOLERANCIA_PX = 2.0
CONTROL_DEADBAND_PX = 0.35
RECENTER_SETTLE_S = 1.0
PRE_PERTURB_TRACK_S = 1.2
TRACKER_TIMEOUT_S = 12.0
CONTROL_HZ = 45.0
SIGNAL_TIMEOUT_S = 0.45
VEL_MAX_TESTE = min(1.6, VEL_MAX_LIMITE)
CMD_KEEPALIVE_S = 0.15
MIN_CMD_DELTA_TO_SEND = 2e-4
CMD_ZERO_SNAP = 0.35 * VEL_MIN_LIMITE
DERIVATIVE_ALPHA = 0.70

FINE_MATRIX_ENTER_RADIUS_PX = 8.0
FINE_MATRIX_EXIT_RADIUS_PX = 14.0

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


# ===== Safe movement limits =====
SAFE_MAX_DELTA_DEG = 2.0
SAFE_AZ_MIN_DEG = 270.0
SAFE_AZ_MAX_DEG = 30.0
SAFE_ALT_MIN_DEG = -30.0
SAFE_ALT_MAX_DEG = 30.0
SAFE_MAX_RECENTER_DELTA_DEG = 2.0
SAFE_MAX_SENDER_RETURN_DELTA_DEG = 2.5

MOVE_TOLERANCE_DEG = 0.0005
MOVE_TIMEOUT_S = 45.0
MOVE_MAX_CORRECTIONS = 10
MOVE_KP = 1.3967
MOVE_KI = 0.0001
MOVE_KD = 0.1015

CM_RECENTER_MAX_ITERS = 8
REPEAT_COUNT = 1
TWIDDLE_MAX_ITERS = 1
TWIDDLE_MIN_DP_SUM = 0.05
DP_SHRINK = 0.60
DP_GROW = 1.20
RESULTS_JSON = json_output_path("autotune_pid_tracker_resultados.json")


# ===== Initial search point =====
INITIAL_KP_AZ = 1.44
INITIAL_KP_ALT = 1.44
INITIAL_KD_AZ = 0.18
INITIAL_KD_ALT = 0.18
INITIAL_TRIM_GAIN = 1.2
INITIAL_MEASUREMENT_ALPHA = 0.70
INITIAL_CMD_ACCEL_LIMIT = 2.00

INITIAL_DP = {
    "kp_az": 0.06,
    "kp_alt": 0.06,
    "kd_az": 0.02,
    "kd_alt": 0.02,
    "trim_gain": 0.35,
    "measurement_alpha": 0.05,
    "cmd_accel_limit": 0.30,
}

PARAM_BOUNDS = {
    "kp_az": (0.30, 3.0),
    "kp_alt": (0.30, 3.0),
    "kd_az": (0.00, 0.60),
    "kd_alt": (0.00, 0.60),
    "trim_gain": (0.0, 8.0),
    "measurement_alpha": (0.35, 0.92),
    "cmd_accel_limit": (0.50, 5.00),
}

TWIDDLE_FIELDS = (
    "kp_az",
    "kp_alt",
    "kd_az",
    "kd_alt",
    "cmd_accel_limit",
    "measurement_alpha",
    "trim_gain",
)

EARLY_ABORT_AFTER_S = 4.0
EARLY_ABORT_RADIUS_PX = 12.0
EARLY_ABORT_RUNAWAY_EVENTS = 2

FAILURE_PENALTY = 140.0
RUNAWAY_PENALTY = 14.0
REENTRY_PENALTY = 1.5
FIRST_REENTRY_WEIGHT = 0.20
SIGNAL_LOSS_PENALTY = 20.0
SKIPPED_TEST_PENALTY = 100.0
RMS_RADIUS_WEIGHT = 0.10
MAX_RADIUS_WEIGHT = 0.04
SATURATION_WEIGHT = 8.0


@dataclass(frozen=True)
class TunedParams:
    kp_az: float
    kp_alt: float
    kd_az: float
    kd_alt: float
    trim_gain: float
    measurement_alpha: float
    cmd_accel_limit: float


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
    active_matrix_name: str = "coarse"


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
    rms_radius_px: float | None
    max_radius_px: float | None
    saturation_fraction: float
    mean_cmd_norm_deg_s: float
    samples: int
    sender_return_ok: bool
    reason: str


@dataclass
class CandidateEvaluation:
    kp_az: float
    kp_alt: float
    kd_az: float
    kd_alt: float
    trim_gain: float
    measurement_alpha: float
    cmd_accel_limit: float
    score: float
    success_count: int
    total_tests: int
    median_time_s: float | None
    mean_runaway_events: float
    mean_center_entries: float
    mean_final_radius_px: float
    mean_rms_radius_px: float
    mean_max_radius_px: float
    mean_saturation_fraction: float
    trials: list[TrialResult]


PERTURB_AXIS_DEG = 0.010
PERTURB_DIAG_DEG = 0.007

PERTURBATIONS = [
    Perturbation("+Alt", 0.0, +PERTURB_AXIS_DEG),
    Perturbation("-Az", -PERTURB_AXIS_DEG, 0.0),
    Perturbation("+Az", +PERTURB_AXIS_DEG, 0.0),
    Perturbation("-Alt", 0.0, -PERTURB_AXIS_DEG),
    Perturbation("+Az+Alt", +PERTURB_DIAG_DEG, +PERTURB_DIAG_DEG),
    Perturbation("-Az-Alt", -PERTURB_DIAG_DEG, -PERTURB_DIAG_DEG),
]


class AxisPID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        output_limits: tuple[float, float],
        integral_limit: float | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.min_output, self.max_output = output_limits
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def reset(self):
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def update(self, axis: int, measured_value: float):
        error = calc_error(axis, self.setpoint, measured_value)
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            self._last_error = error
            return 0.0, error

        dt = max(now - self._last_time, 1e-4)
        de = error - self._last_error
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = float(
                np.clip(self._integral, -self.integral_limit, self.integral_limit)
            )

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * (de / dt))
        output = float(np.clip(output, self.min_output, self.max_output))
        self._last_time = now
        self._last_error = error
        return output, error


class AlpacaTelescope:
    def __init__(
        self,
        device_number: int,
        label: str,
        alpaca_root: str,
        invert_az: bool = True,
        invert_alt: bool = False,
    ):
        self.device_number = int(device_number)
        self.label = label
        self.invert_az = invert_az
        self.invert_alt = invert_alt
        root = alpaca_root.rstrip("/")
        if "/telescope/" in root.lower():
            self.base_url = root
        else:
            self.base_url = f"{root}/telescope/{self.device_number}"
        self.session = requests.Session()
        self._transaction_ids = itertools.count(1)

    def call(self, method: str, command: str, timeout: float = 5.0, **extra_args):
        params = {
            "ClientID": CLIENT_ID,
            "ClientTransactionID": next(self._transaction_ids),
        }
        params.update(extra_args.pop("params", {}))
        resp = self.session.request(
            method,
            f"{self.base_url}/{command}",
            params=params,
            timeout=timeout,
            **extra_args,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("ErrorNumber", 0):
            raise RuntimeError(f"{self.label} {command}: {payload.get('ErrorMessage')}")
        return payload.get("Value")

    def ensure_connected(self):
        if not self.call("GET", "connected"):
            self.call("PUT", "connected", data={"Connected": True})

    def ensure_unparked(self):
        try:
            if self.call("GET", "atpark"):
                if self.call("GET", "canunpark"):
                    self.call("PUT", "unpark", timeout=10)
        except Exception:
            pass

    def ensure_not_tracking(self):
        try:
            if str(self.call("GET", "tracking")).lower() in {"true", "1"}:
                self.call("PUT", "tracking", data={"Tracking": False})
        except Exception:
            pass

    def prepare(self):
        print(f"Conectando {self.label} em {self.base_url}...")
        self.ensure_connected()
        self.ensure_unparked()
        self.ensure_not_tracking()

    def read_altaz(self) -> tuple[float, float]:
        az = float(self.call("GET", "azimuth"))
        alt = float(self.call("GET", "altitude"))
        return az, alt

    def move_axis(self, axis: int, rate_deg_per_s: float):
        rate = float(rate_deg_per_s)
        if self.invert_az and axis == 0:
            rate = -rate
        if self.invert_alt and axis == 1:
            rate = -rate
        self.call("PUT", "moveaxis", data={"Axis": axis, "Rate": rate}, timeout=2.0)

    def stop_all(self):
        for axis in (0, 1):
            try:
                self.call(
                    "PUT",
                    "moveaxis",
                    data={"Axis": axis, "Rate": 0.0},
                    timeout=2.0,
                )
            except Exception:
                pass

    def move_relative_pid(
        self,
        delta_az_deg: float,
        delta_alt_deg: float,
        verbose: bool = True,
    ) -> dict:
        az0, alt0 = self.read_altaz()
        target_az = (az0 + float(delta_az_deg)) % 360.0
        target_alt = alt0 + float(delta_alt_deg)
        target_alt = float(np.clip(target_alt, -90.0, 90.0))

        pid_az = AxisPID(
            MOVE_KP,
            MOVE_KI,
            MOVE_KD,
            setpoint=target_az,
            output_limits=(-VEL_MAX_LIMITE, VEL_MAX_LIMITE),
            integral_limit=5,
        )
        pid_alt = AxisPID(
            MOVE_KP,
            MOVE_KI,
            MOVE_KD,
            setpoint=target_alt,
            output_limits=(-VEL_MAX_LIMITE, VEL_MAX_LIMITE),
            integral_limit=5,
        )

        if verbose:
            print(
                f"  {self.label}: alvo Az={target_az:.4f} deg "
                f"(d {delta_az_deg:+.4f}), Alt={target_alt:.4f} deg "
                f"(d {delta_alt_deg:+.4f})"
            )

        t0 = time.perf_counter()
        elapsed = 0.0
        error_last_az = None
        error_last_alt = None
        inversions_az = 0
        inversions_alt = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                while True:
                    elapsed = time.perf_counter() - t0
                    if elapsed > MOVE_TIMEOUT_S:
                        raise TimeoutError(f"{self.label}: tempo limite movendo eixos")

                    az, alt = self.read_altaz()
                    cmd_az, error_az = pid_az.update(0, az)
                    cmd_alt, error_alt = pid_alt.update(1, alt)
                    abs_az = abs(error_az)
                    abs_alt = abs(error_alt)

                    az_ok = abs_az < MOVE_TOLERANCE_DEG
                    alt_ok = abs_alt < MOVE_TOLERANCE_DEG
                    if az_ok and alt_ok:
                        break

                    if error_last_az is not None and error_az * error_last_az < 0:
                        inversions_az += 1
                        pid_az.reset()
                        cmd_az = 0.0
                        if inversions_az > MOVE_MAX_CORRECTIONS:
                            raise RuntimeError(f"{self.label}: excesso de inversoes em az")

                    if error_last_alt is not None and error_alt * error_last_alt < 0:
                        inversions_alt += 1
                        pid_alt.reset()
                        cmd_alt = 0.0
                        if inversions_alt > MOVE_MAX_CORRECTIONS:
                            raise RuntimeError(f"{self.label}: excesso de inversoes em alt")

                    if az_ok:
                        cmd_az = 0.0
                    elif 0 < abs(cmd_az) < VEL_MIN_LIMITE:
                        cmd_az = float(VEL_MIN_LIMITE * np.sign(cmd_az))

                    if alt_ok:
                        cmd_alt = 0.0
                    elif 0 < abs(cmd_alt) < VEL_MIN_LIMITE:
                        cmd_alt = float(VEL_MIN_LIMITE * np.sign(cmd_alt))

                    f_az = executor.submit(self.move_axis, 0, cmd_az)
                    f_alt = executor.submit(self.move_axis, 1, cmd_alt)
                    f_az.result()
                    f_alt.result()

                    error_last_az = error_az
                    error_last_alt = error_alt

                    max_err = max(abs_az, abs_alt)
                    if max_err > 1.0:
                        time.sleep(0.1)
                    elif max_err > 0.1:
                        time.sleep(0.05)
                    else:
                        time.sleep(0.01)
            finally:
                self.stop_all()

        azf, altf = self.read_altaz()
        return {
            "start_az_deg": az0,
            "start_alt_deg": alt0,
            "target_az_deg": target_az,
            "target_alt_deg": target_alt,
            "final_az_deg": azf,
            "final_alt_deg": altf,
            "elapsed_s": elapsed,
        }

    def move_to_altaz(self, target_az_deg: float, target_alt_deg: float, verbose: bool = True) -> dict:
        az, alt = self.read_altaz()
        delta_az = calc_error(0, target_az_deg, az)
        delta_alt = float(target_alt_deg) - alt
        return self.move_relative_pid(delta_az, delta_alt, verbose=verbose)


class MountAgentTelescope:
    def __init__(self, agent_url: str, label: str):
        self.label = label
        self.agent_url = agent_url.rstrip("/")
        self.base_url = self.agent_url
        self.session = requests.Session()
        self.invert_az = None
        self.invert_alt = None

    def _request(self, method: str, endpoint: str, timeout: float = 5.0, **kwargs) -> dict:
        resp = self.session.request(
            method,
            f"{self.agent_url}{endpoint}",
            timeout=timeout,
            **kwargs,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok", False):
            raise RuntimeError(f"{self.label} {endpoint}: {payload.get('error', payload)}")
        return payload

    def prepare(self):
        print(f"Conectando {self.label} via mount_agent em {self.agent_url}...")
        payload = self._request("GET", "/health", timeout=5.0)
        print(f"  Agent OK: {payload.get('label', self.label)}")
        self.read_altaz()

    def read_altaz(self) -> tuple[float, float]:
        payload = self._request("GET", "/position", timeout=5.0)
        return float(payload["az_deg"]), float(payload["alt_deg"])

    def stop_all(self):
        try:
            self._request("POST", "/stop", timeout=5.0, json={})
        except Exception:
            pass

    def move_relative_pid(
        self,
        delta_az_deg: float,
        delta_alt_deg: float,
        verbose: bool = True,
    ) -> dict:
        if verbose:
            print(
                f"  {self.label}: requisitando move_relative "
                f"dAz={delta_az_deg:+.4f} deg dAlt={delta_alt_deg:+.4f} deg"
            )
        payload = self._request(
            "POST",
            "/move_relative",
            timeout=MOVE_TIMEOUT_S + 10.0,
            json={
                "delta_az_deg": float(delta_az_deg),
                "delta_alt_deg": float(delta_alt_deg),
                "tolerance_deg": MOVE_TOLERANCE_DEG,
            },
        )
        if not payload.get("ok", False):
            raise RuntimeError(f"{self.label}: movimento remoto nao convergiu: {payload}")
        return payload

    def move_to_altaz(self, target_az_deg: float, target_alt_deg: float, verbose: bool = True) -> dict:
        az, alt = self.read_altaz()
        delta_az = calc_error(0, target_az_deg, az)
        delta_alt = float(target_alt_deg) - alt
        return self.move_relative_pid(delta_az, delta_alt, verbose=verbose)


def calc_error(axis: int, target: float, pos: float) -> float:
    if axis == 0:
        diff = (target - pos) % 360.0
        if diff > 180.0:
            return diff - 360.0
        return diff
    return target - pos


def _clip_params(params: TunedParams) -> TunedParams:
    values = asdict(params)
    for key, (lower, upper) in PARAM_BOUNDS.items():
        values[key] = float(np.clip(values[key], lower, upper))
    return TunedParams(**values)


def _params_equal(a: TunedParams, b: TunedParams) -> bool:
    return all(abs(getattr(a, key) - getattr(b, key)) < 1e-9 for key in PARAM_BOUNDS)


def _with_param_delta(params: TunedParams, field_name: str, delta: float) -> TunedParams:
    values = asdict(params)
    values[field_name] += float(delta)
    return _clip_params(TunedParams(**values))


def _format_params(params: TunedParams) -> str:
    return (
        f"KpAz={params.kp_az:.3f} KpAlt={params.kp_alt:.3f} "
        f"KdAz={params.kd_az:.3f} KdAlt={params.kd_alt:.3f} "
        f"Trim={params.trim_gain:.3f} Alpha={params.measurement_alpha:.3f} "
        f"Accel={params.cmd_accel_limit:.3f}"
    )


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
            f"Nao encontrei matriz {label}. Testei: "
            f"{', '.join(str(path) for path in candidates)}"
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


def _configure_focus_detector(focus_mode: str) -> None:
    if foco_temp is None:
        if focus_mode == "dual":
            print(
                "Aviso: detector temporario de dois focos nao foi importado; "
                "o autotune vai usar CM local simples dentro do ROI."
            )
        return
    foco_temp.set_focus_mode(focus_mode)


def _measure_frame_cm(frame: np.ndarray, focus_mode: str, full_frame: bool):
    if focus_mode == "dual" and foco_temp is not None:
        cm = foco_temp.centro_massa(frame)
        if cm is None:
            return None
        x_cm, y_cm, _, touches_edge = cm
        return float(x_cm), float(y_cm), bool(touches_edge)

    if full_frame:
        cm = centro_massa_full_frame(frame)
        if cm is None:
            return None
        x_cm, y_cm, _, touches_edge = cm
        return float(x_cm), float(y_cm), bool(touches_edge)

    cm = calcular_cm_corrigido(frame)
    if cm is None:
        return None
    x_cm, y_cm = cm
    return float(x_cm), float(y_cm), False


def _validate_perturbations():
    for perturbation in PERTURBATIONS:
        if (
            abs(perturbation.delta_az_deg) > SAFE_MAX_DELTA_DEG
            or abs(perturbation.delta_alt_deg) > SAFE_MAX_DELTA_DEG
        ):
            raise ValueError(
                f"Perturbacao {perturbation.name} excede o limite seguro "
                f"de {SAFE_MAX_DELTA_DEG} deg."
            )


def _apply_min_velocity(cmd: float) -> float:
    if abs(cmd) < CMD_ZERO_SNAP:
        return 0.0
    if 0.0 < abs(cmd) < VEL_MIN_LIMITE:
        return float(VEL_MIN_LIMITE * np.sign(cmd))
    return float(cmd)


def _slew_limit(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return float(target)
    return float(current + (np.sign(delta) * max_delta))


def _pixel_error_to_mount_error(dx_px: float, dy_px: float, A_inv: np.ndarray):
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


def _ensure_alt_target_safe(current_alt_deg: float, delta_alt_deg: float):
    target_alt = current_alt_deg + delta_alt_deg
    if target_alt < SAFE_ALT_MIN_DEG or target_alt > SAFE_ALT_MAX_DEG:
        raise RuntimeError(
            f"Movimento de altitude sairia da faixa segura "
            f"[{SAFE_ALT_MIN_DEG}, {SAFE_ALT_MAX_DEG}] deg: "
            f"{current_alt_deg:.4f} -> {target_alt:.4f}"
        )


def _ensure_safe_relative_move(
    telescope: AlpacaTelescope,
    delta_az_deg: float,
    delta_alt_deg: float,
    max_delta_deg: float,
) -> tuple[float, float]:
    if abs(delta_az_deg) > max_delta_deg or abs(delta_alt_deg) > max_delta_deg:
        raise RuntimeError(
            f"{telescope.label}: delta ({delta_az_deg:+.4f}, {delta_alt_deg:+.4f}) "
            f"excede limite seguro de {max_delta_deg} deg."
        )

    az_now, alt_now = telescope.read_altaz()
    target_az = (az_now + float(delta_az_deg)) % 360.0
    if not _az_in_safe_window(az_now):
        raise RuntimeError(
            f"{telescope.label}: azimute atual {az_now:.4f} deg fora da janela segura "
            f"[{SAFE_AZ_MIN_DEG}, {SAFE_AZ_MAX_DEG}] deg."
        )
    if not _az_in_safe_window(target_az):
        raise RuntimeError(
            f"{telescope.label}: alvo {target_az:.4f} deg sairia da janela segura "
            f"[{SAFE_AZ_MIN_DEG}, {SAFE_AZ_MAX_DEG}] deg."
        )
    _ensure_alt_target_safe(alt_now, float(delta_alt_deg))
    return az_now, alt_now


def _apply_sender_perturbation(sender: AlpacaTelescope, perturbation: Perturbation):
    _ensure_safe_relative_move(
        sender,
        perturbation.delta_az_deg,
        perturbation.delta_alt_deg,
        SAFE_MAX_DELTA_DEG,
    )
    return sender.move_relative_pid(
        perturbation.delta_az_deg,
        perturbation.delta_alt_deg,
        verbose=True,
    )


def _return_sender_to_home(
    sender: AlpacaTelescope,
    home_az_deg: float,
    home_alt_deg: float,
) -> bool:
    try:
        az_now, alt_now = sender.read_altaz()
        delta_az = calc_error(0, home_az_deg, az_now)
        delta_alt = float(home_alt_deg) - alt_now
        _ensure_safe_relative_move(
            sender,
            delta_az,
            delta_alt,
            SAFE_MAX_SENDER_RETURN_DELTA_DEG,
        )
        sender.move_to_altaz(home_az_deg, home_alt_deg, verbose=False)
        return True
    except Exception as exc:
        print(f"  Aviso: nao consegui retornar {sender.label} para a origem: {exc}")
        sender.stop_all()
        return False


def _reentry_cost(extra_entries: int) -> float:
    if extra_entries <= 0:
        return 0.0
    if extra_entries == 1:
        return FIRST_REENTRY_WEIGHT * REENTRY_PENALTY
    return (FIRST_REENTRY_WEIGHT * REENTRY_PENALTY) + ((extra_entries - 1) * REENTRY_PENALTY)


def _control_loop_pd_trim(
    state: SharedState,
    matrices: dict,
    receiver: AlpacaTelescope,
    params: TunedParams,
):
    ctrl_az = MeasurementPDTrim(
        kp=params.kp_az,
        kd=params.kd_az,
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
        kp=params.kp_alt,
        kd=params.kd_alt,
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
    active_matrix_name = "coarse"
    active_matrix = matrices["coarse"]

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
                    active_matrix_name = "coarse"
                    active_matrix = matrices["coarse"]
                elif seq != last_seq:
                    last_seq = seq
                    radius_px = float(np.hypot(dx_filt, dy_filt))
                    previous_matrix_name = active_matrix_name

                    if active_matrix_name == "coarse" and radius_px <= FINE_MATRIX_ENTER_RADIUS_PX:
                        active_matrix_name = "fine"
                        active_matrix = matrices["fine"]
                    elif active_matrix_name == "fine" and radius_px >= FINE_MATRIX_EXIT_RADIUS_PX:
                        active_matrix_name = "coarse"
                        active_matrix = matrices["coarse"]

                    if active_matrix_name != previous_matrix_name:
                        ctrl_az.reset()
                        ctrl_alt.reset()
                        trim_mode_active = False

                    if active_matrix_name == "fine" and radius_px <= TRIM_ENTER_RADIUS_PX:
                        trim_mode_active = True
                    elif active_matrix_name != "fine" or radius_px >= TRIM_EXIT_RADIUS_PX:
                        if trim_mode_active:
                            ctrl_az.clear_trim()
                            ctrl_alt.clear_trim()
                        trim_mode_active = False

                    if abs(dx_filt) <= CONTROL_DEADBAND_PX and abs(dy_filt) <= CONTROL_DEADBAND_PX:
                        err_az = 0.0
                        err_alt = 0.0
                        ctrl_az.clear_trim()
                        ctrl_alt.clear_trim()
                    else:
                        err_az, err_alt = _pixel_error_to_mount_error(dx_filt, dy_filt, active_matrix)

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

                max_step = params.cmd_accel_limit * dt_loop
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
                    future_az = executor.submit(receiver.move_axis, 0, cmd_az)
                if send_alt:
                    future_alt = executor.submit(receiver.move_axis, 1, cmd_alt)

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
                    state.active_matrix_name = active_matrix_name

                elapsed = time.perf_counter() - loop_t0
                if elapsed < dt_target:
                    time.sleep(dt_target - elapsed)
        finally:
            receiver.stop_all()


def centralize_with_center_of_mass_safe(
    matrices: dict,
    receiver: AlpacaTelescope,
    focus_mode: str,
) -> tuple[bool, dict]:
    reset_camera_roi()

    last_dx = None
    last_dy = None
    for idx in range(1, CM_RECENTER_MAX_ITERS + 1):
        frame = capture_frame(EXPOSURE_SECONDS)
        cm = _measure_frame_cm(frame, focus_mode, full_frame=True)
        if cm is None:
            return False, {"reason": "no_signal", "iterations": idx - 1}

        x_cm, y_cm, touches_edge = cm
        cx, cy = centro_camera(frame)
        dx = float(x_cm - cx)
        dy = float(y_cm - cy)
        radius = float(np.hypot(dx, dy))
        last_dx = dx
        last_dy = dy

        print(
            f"  CM recenter passo {idx}/{CM_RECENTER_MAX_ITERS}: "
            f"dx={dx:+.1f}px dy={dy:+.1f}px"
        )

        if abs(dx) <= TOLERANCIA_PX and abs(dy) <= TOLERANCIA_PX and not touches_edge:
            return True, {
                "reason": "centered",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
            }

        active_matrix = matrices["fine"] if radius <= FINE_MATRIX_ENTER_RADIUS_PX else matrices["coarse"]
        d_az_deg, d_alt_deg = active_matrix @ np.array([-dx, -dy], dtype=float)
        if abs(d_az_deg) > SAFE_MAX_RECENTER_DELTA_DEG or abs(d_alt_deg) > SAFE_MAX_RECENTER_DELTA_DEG:
            return False, {
                "reason": "unsafe_recenter_delta",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
                "d_az_deg": float(d_az_deg),
                "d_alt_deg": float(d_alt_deg),
            }

        try:
            _ensure_safe_relative_move(
                receiver,
                float(d_az_deg),
                float(d_alt_deg),
                SAFE_MAX_RECENTER_DELTA_DEG,
            )
        except RuntimeError as exc:
            return False, {
                "reason": "unsafe_recenter_move",
                "iterations": idx,
                "dx_px": dx,
                "dy_px": dy,
                "error": str(exc),
            }

        receiver.move_relative_pid(float(d_az_deg), float(d_alt_deg), verbose=True)

    return False, {
        "reason": "max_iterations",
        "iterations": CM_RECENTER_MAX_ITERS,
        "dx_px": last_dx,
        "dy_px": last_dy,
    }


def run_tracker_trial(
    matrices: dict,
    receiver: AlpacaTelescope,
    sender: AlpacaTelescope,
    focus_mode: str,
    params: TunedParams,
    perturbation: Perturbation,
    timeout_s: float,
) -> TrialResult:
    state = SharedState()
    ctrl_thread = threading.Thread(
        target=_control_loop_pd_trim,
        args=(state, matrices, receiver, params),
        daemon=True,
    )
    ctrl_thread.start()

    t_start = time.perf_counter()
    perturb_start = None
    center_hold_start = None
    center_entries = 0
    was_centered = False
    signal_loss_seen = False
    final_dx = None
    final_dy = None
    reason = "timeout"
    success = False
    settle_time_s = None

    sample_count = 0
    sum_radius_sq = 0.0
    max_radius = 0.0
    cmd_norm_sum = 0.0
    saturation_count = 0

    perturb_status = {"started": False, "done": False, "error": None}
    perturb_lock = threading.Lock()

    def _sender_worker():
        with perturb_lock:
            perturb_status["started"] = True
        try:
            _apply_sender_perturbation(sender, perturbation)
        except Exception as exc:
            with perturb_lock:
                perturb_status["error"] = str(exc)
        finally:
            with perturb_lock:
                perturb_status["done"] = True

    perturb_thread = None

    try:
        while True:
            t_now = time.perf_counter()
            elapsed_total = t_now - t_start
            if elapsed_total >= (PRE_PERTURB_TRACK_S + timeout_s):
                reason = "timeout"
                break

            if perturb_thread is None and elapsed_total >= PRE_PERTURB_TRACK_S:
                perturb_start = t_now
                center_hold_start = None
                was_centered = False
                print(f"    Perturbando com {sender.label}: {perturbation.name}")
                perturb_thread = threading.Thread(target=_sender_worker, daemon=True)
                perturb_thread.start()

            frame_window = capture_frame(EXPOSURE_SECONDS)
            cm = _measure_frame_cm(frame_window, focus_mode, full_frame=False)

            with perturb_lock:
                perturb_done = bool(perturb_status["done"])
                perturb_error = perturb_status["error"]

            if perturb_error is not None:
                reason = f"sender_perturbation_failed:{perturb_error}"
                break

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

            x_cm_local, y_cm_local, _ = cm
            dx = float(x_cm_local - (WINDOW_SIZE / 2))
            dy = float(y_cm_local - (WINDOW_SIZE / 2))
            radius = float(np.hypot(dx, dy))
            final_dx = dx
            final_dy = dy

            with state.lock:
                if state.measurement_seq == 0 or not state.has_signal:
                    dx_filt = dx
                    dy_filt = dy
                else:
                    alpha = params.measurement_alpha
                    dx_filt = (alpha * dx) + ((1.0 - alpha) * state.dx_filt_px)
                    dy_filt = (alpha * dy) + ((1.0 - alpha) * state.dy_filt_px)

                state.dx_px = dx
                state.dy_px = dy
                state.dx_filt_px = float(dx_filt)
                state.dy_filt_px = float(dy_filt)
                state.has_signal = True
                state.measurement_seq += 1
                state.measurement_ts = t_now
                runaway_events = state.runaway_events
                cmd_az = state.cmd_az_deg_s
                cmd_alt = state.cmd_alt_deg_s

            if perturb_start is not None:
                sample_count += 1
                sum_radius_sq += radius * radius
                max_radius = max(max_radius, radius)
                cmd_norm = float(np.hypot(cmd_az, cmd_alt))
                cmd_norm_sum += cmd_norm
                if abs(cmd_az) >= 0.95 * VEL_MAX_TESTE or abs(cmd_alt) >= 0.95 * VEL_MAX_TESTE:
                    saturation_count += 1

            is_centered = (abs(dx) < TOLERANCIA_PX) and (abs(dy) < TOLERANCIA_PX)
            if perturb_done:
                if is_centered and not was_centered:
                    center_entries += 1

                if is_centered:
                    if center_hold_start is None:
                        center_hold_start = t_now
                    elif (t_now - center_hold_start) >= RECENTER_SETTLE_S:
                        success = True
                        settle_time_s = t_now - (perturb_start or t_start)
                        reason = "stabilized"
                        break
                else:
                    center_hold_start = None

                elapsed_after_perturb = t_now - (perturb_start or t_start)
                if (
                    not success
                    and center_entries == 0
                    and elapsed_after_perturb >= EARLY_ABORT_AFTER_S
                    and runaway_events >= EARLY_ABORT_RUNAWAY_EVENTS
                    and radius >= EARLY_ABORT_RADIUS_PX
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
        if perturb_thread is not None and perturb_thread.is_alive():
            sender.stop_all()
            perturb_thread.join(timeout=2.0)

    rms_radius = math.sqrt(sum_radius_sq / sample_count) if sample_count else None
    mean_cmd = cmd_norm_sum / sample_count if sample_count else 0.0
    saturation_fraction = saturation_count / sample_count if sample_count else 0.0

    return TrialResult(
        perturbation=perturbation.name,
        repeat_idx=0,
        success=success,
        settle_time_s=settle_time_s,
        timeout_s=timeout_s,
        center_entries=center_entries,
        runaway_events=runaway_events,
        signal_loss_seen=signal_loss_seen,
        final_dx_px=final_dx,
        final_dy_px=final_dy,
        rms_radius_px=rms_radius,
        max_radius_px=max_radius if sample_count else None,
        saturation_fraction=saturation_fraction,
        mean_cmd_norm_deg_s=mean_cmd,
        samples=sample_count,
        sender_return_ok=True,
        reason=reason,
    )


def _score_trials(
    trials: list[TrialResult],
    total_expected: int,
) -> tuple[float, int, float | None, float, float, float, float, float, float]:
    success_results = [trial for trial in trials if trial.success and trial.settle_time_s is not None]
    success_count = len(success_results)
    median_time = statistics.median([trial.settle_time_s for trial in success_results]) if success_results else None
    mean_runaway = statistics.mean([trial.runaway_events for trial in trials]) if trials else 0.0
    mean_center_entries = statistics.mean([trial.center_entries for trial in trials]) if trials else 0.0

    final_radii = []
    rms_values = []
    max_values = []
    saturation_values = []
    for trial in trials:
        if trial.final_dx_px is not None and trial.final_dy_px is not None:
            final_radii.append(math.hypot(trial.final_dx_px, trial.final_dy_px))
        if trial.rms_radius_px is not None:
            rms_values.append(trial.rms_radius_px)
        if trial.max_radius_px is not None:
            max_values.append(trial.max_radius_px)
        saturation_values.append(trial.saturation_fraction)

    mean_final_radius = statistics.mean(final_radii) if final_radii else 0.0
    mean_rms_radius = statistics.mean(rms_values) if rms_values else 0.0
    mean_max_radius = statistics.mean(max_values) if max_values else 0.0
    mean_saturation = statistics.mean(saturation_values) if saturation_values else 0.0

    score = 0.0
    for trial in trials:
        extra_entries = max(trial.center_entries - 1, 0)
        score += RUNAWAY_PENALTY * trial.runaway_events
        score += _reentry_cost(extra_entries)
        score += SATURATION_WEIGHT * trial.saturation_fraction
        if trial.rms_radius_px is not None:
            score += RMS_RADIUS_WEIGHT * trial.rms_radius_px
        if trial.max_radius_px is not None:
            score += MAX_RADIUS_WEIGHT * trial.max_radius_px
        if trial.signal_loss_seen:
            score += SIGNAL_LOSS_PENALTY
        if not trial.sender_return_ok:
            score += SIGNAL_LOSS_PENALTY

        if trial.success and trial.settle_time_s is not None:
            score += trial.settle_time_s
        else:
            score += FAILURE_PENALTY
            score += trial.timeout_s

    skipped = max(total_expected - len(trials), 0)
    score += SKIPPED_TEST_PENALTY * skipped
    return (
        score,
        success_count,
        median_time,
        mean_runaway,
        mean_center_entries,
        mean_final_radius,
        mean_rms_radius,
        mean_max_radius,
        mean_saturation,
    )


def _partial_trial_score(trials: list[TrialResult]) -> float:
    score = 0.0
    for trial in trials:
        extra_entries = max(trial.center_entries - 1, 0)
        score += RUNAWAY_PENALTY * trial.runaway_events
        score += _reentry_cost(extra_entries)
        score += SATURATION_WEIGHT * trial.saturation_fraction
        if trial.rms_radius_px is not None:
            score += RMS_RADIUS_WEIGHT * trial.rms_radius_px
        if trial.max_radius_px is not None:
            score += MAX_RADIUS_WEIGHT * trial.max_radius_px
        if trial.signal_loss_seen:
            score += SIGNAL_LOSS_PENALTY
        if trial.success and trial.settle_time_s is not None:
            score += trial.settle_time_s
        else:
            score += FAILURE_PENALTY
            score += trial.timeout_s
    return score


def evaluate_candidate(
    matrices: dict,
    receiver: AlpacaTelescope,
    sender: AlpacaTelescope,
    focus_mode: str,
    params: TunedParams,
    incumbent: CandidateEvaluation | None = None,
) -> CandidateEvaluation:
    params = _clip_params(params)
    print(f"\nAvaliando candidato: {_format_params(params)}")

    trials: list[TrialResult] = []
    total_expected = len(PERTURBATIONS) * REPEAT_COUNT
    abort_candidate = False

    for repeat_idx in range(1, REPEAT_COUNT + 1):
        for perturbation in PERTURBATIONS:
            print(
                f"  Teste {perturbation.name} | repeticao {repeat_idx}/{REPEAT_COUNT}"
            )
            center_ok, center_info = centralize_with_center_of_mass_safe(
                matrices,
                receiver,
                focus_mode,
            )
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
                    rms_radius_px=None,
                    max_radius_px=None,
                    saturation_fraction=0.0,
                    mean_cmd_norm_deg_s=0.0,
                    samples=0,
                    sender_return_ok=True,
                    reason=reason,
                )
                trials.append(trial)
                print(f"    -> FALHA antes do teste: {reason}")
                break

            set_camera_roi(WINDOW_SIZE, WINDOW_SIZE)
            sender_home = None
            sender_return_ok = True
            try:
                sender_home = sender.read_altaz()
                trial = run_tracker_trial(
                    matrices=matrices,
                    receiver=receiver,
                    sender=sender,
                    focus_mode=focus_mode,
                    params=params,
                    perturbation=perturbation,
                    timeout_s=TRACKER_TIMEOUT_S,
                )
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
                    rms_radius_px=None,
                    max_radius_px=None,
                    saturation_fraction=0.0,
                    mean_cmd_norm_deg_s=0.0,
                    samples=0,
                    sender_return_ok=True,
                    reason=f"exception:{exc}",
                )
            finally:
                receiver.stop_all()
                if sender_home is not None:
                    sender_return_ok = _return_sender_to_home(
                        sender,
                        sender_home[0],
                        sender_home[1],
                    )
                else:
                    sender.stop_all()

            trial.perturbation = perturbation.name
            trial.repeat_idx = repeat_idx
            trial.sender_return_ok = sender_return_ok
            trials.append(trial)

            final_radius = (
                math.hypot(trial.final_dx_px, trial.final_dy_px)
                if trial.final_dx_px is not None and trial.final_dy_px is not None
                else float("nan")
            )
            status = "SUCESSO" if trial.success else "FALHA"
            settle_str = f"{trial.settle_time_s:.3f}s" if trial.settle_time_s is not None else "n/a"
            radius_str = f"{final_radius:.2f}px" if math.isfinite(final_radius) else "n/a"
            rms_str = f"{trial.rms_radius_px:.2f}px" if trial.rms_radius_px is not None else "n/a"
            max_str = f"{trial.max_radius_px:.2f}px" if trial.max_radius_px is not None else "n/a"
            print(
                f"    -> {status} | tempo={settle_str} | rms={rms_str} | max={max_str} | "
                f"runaway={trial.runaway_events} | entradas={trial.center_entries} | "
                f"raio_final={radius_str} | sat={trial.saturation_fraction:.2f} | "
                f"retorno_T2={'ok' if sender_return_ok else 'falhou'} | motivo={trial.reason}"
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
                    "logo no inicio indicam candidato ruim."
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
        mean_rms_radius,
        mean_max_radius,
        mean_saturation,
    ) = _score_trials(trials, total_expected)
    median_str = f"{median_time:.3f}s" if median_time is not None else "n/a"
    print(
        f"  Score={score:.3f} | sucessos={success_count}/{total_expected} | "
        f"mediana={median_str} | runaway_medio={mean_runaway:.2f} | "
        f"entradas_medias={mean_center_entries:.2f} | "
        f"raio_final_medio={mean_final_radius:.2f}px | "
        f"rms_medio={mean_rms_radius:.2f}px | max_medio={mean_max_radius:.2f}px"
    )
    return CandidateEvaluation(
        kp_az=params.kp_az,
        kp_alt=params.kp_alt,
        kd_az=params.kd_az,
        kd_alt=params.kd_alt,
        trim_gain=params.trim_gain,
        measurement_alpha=params.measurement_alpha,
        cmd_accel_limit=params.cmd_accel_limit,
        score=score,
        success_count=success_count,
        total_tests=total_expected,
        median_time_s=median_time,
        mean_runaway_events=mean_runaway,
        mean_center_entries=mean_center_entries,
        mean_final_radius_px=mean_final_radius,
        mean_rms_radius_px=mean_rms_radius,
        mean_max_radius_px=mean_max_radius,
        mean_saturation_fraction=mean_saturation,
        trials=trials,
    )


def twiddle_search(
    matrices: dict,
    receiver: AlpacaTelescope,
    sender: AlpacaTelescope,
    focus_mode: str,
):
    current = _clip_params(
        TunedParams(
            kp_az=INITIAL_KP_AZ,
            kp_alt=INITIAL_KP_ALT,
            kd_az=INITIAL_KD_AZ,
            kd_alt=INITIAL_KD_ALT,
            trim_gain=INITIAL_TRIM_GAIN,
            measurement_alpha=INITIAL_MEASUREMENT_ALPHA,
            cmd_accel_limit=INITIAL_CMD_ACCEL_LIMIT,
        )
    )
    dp = dict(INITIAL_DP)
    history: list[CandidateEvaluation] = []

    best_eval = evaluate_candidate(matrices, receiver, sender, focus_mode, current)
    history.append(best_eval)

    for iteration in range(1, TWIDDLE_MAX_ITERS + 1):
        if sum(abs(value) for value in dp.values()) < TWIDDLE_MIN_DP_SUM:
            print("\nParando: soma dos passos do twiddle ficou pequena.")
            break

        print("\n" + "=" * 72)
        print(f"Iteracao {iteration}/{TWIDDLE_MAX_ITERS} | best {_format_params(current)}")
        print(
            "Passos: "
            + ", ".join(f"{key}={value:.4f}" for key, value in dp.items())
        )
        print("=" * 72)

        for field_name in TWIDDLE_FIELDS:
            improved = False

            trial_up = _with_param_delta(current, field_name, dp[field_name])
            if not _params_equal(trial_up, current):
                eval_up = evaluate_candidate(
                    matrices,
                    receiver,
                    sender,
                    focus_mode,
                    trial_up,
                    incumbent=best_eval,
                )
                history.append(eval_up)
                if eval_up.score < best_eval.score:
                    current = trial_up
                    best_eval = eval_up
                    dp[field_name] *= DP_GROW
                    improved = True

            if improved:
                continue

            trial_down = _with_param_delta(current, field_name, -dp[field_name])
            if not _params_equal(trial_down, current):
                eval_down = evaluate_candidate(
                    matrices,
                    receiver,
                    sender,
                    focus_mode,
                    trial_down,
                    incumbent=best_eval,
                )
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
            round(evaluation.kp_az, 6),
            round(evaluation.kp_alt, 6),
            round(evaluation.kd_az, 6),
            round(evaluation.kd_alt, 6),
            round(evaluation.trim_gain, 6),
            round(evaluation.measurement_alpha, 6),
            round(evaluation.cmd_accel_limit, 6),
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
            item.mean_rms_radius_px,
        ),
    )

    print("\n=== Podio Autotune Tracker com dois telescopios ===")
    for idx, item in enumerate(ranking[:5], start=1):
        median_str = f"{item.median_time_s:.3f}s" if item.median_time_s is not None else "n/a"
        params = TunedParams(
            item.kp_az,
            item.kp_alt,
            item.kd_az,
            item.kd_alt,
            item.trim_gain,
            item.measurement_alpha,
            item.cmd_accel_limit,
        )
        print(
            f"{idx}. {_format_params(params)} | sucessos={item.success_count}/{item.total_tests} | "
            f"score={item.score:.3f} | mediana={median_str} | "
            f"rms={item.mean_rms_radius_px:.2f}px | max={item.mean_max_radius_px:.2f}px | "
            f"runaway={item.mean_runaway_events:.2f} | sat={item.mean_saturation_fraction:.2f}"
        )


def save_results(
    path: str | Path,
    best_eval: CandidateEvaluation,
    history: list[CandidateEvaluation],
    metadata: dict,
):
    payload = {
        "timestamp_epoch": time.time(),
        "metadata": metadata,
        "config": {
            "window_size": WINDOW_SIZE,
            "exposure_seconds": EXPOSURE_SECONDS,
            "tolerancia_px": TOLERANCIA_PX,
            "pre_perturb_track_s": PRE_PERTURB_TRACK_S,
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
            "perturbations": [asdict(p) for p in PERTURBATIONS],
            "initial_params": asdict(
                TunedParams(
                    INITIAL_KP_AZ,
                    INITIAL_KP_ALT,
                    INITIAL_KD_AZ,
                    INITIAL_KD_ALT,
                    INITIAL_TRIM_GAIN,
                    INITIAL_MEASUREMENT_ALPHA,
                    INITIAL_CMD_ACCEL_LIMIT,
                )
            ),
            "initial_dp": INITIAL_DP,
            "param_bounds": PARAM_BOUNDS,
            "trim_limit": TRIM_LIMIT,
            "trim_leak": TRIM_LEAK,
            "trim_error_max_deg": TRIM_ERROR_MAX_DEG,
            "trim_derivative_max_deg_s": TRIM_DERIVATIVE_MAX_DEG_S,
            "trim_same_sign_s": TRIM_SAME_SIGN_S,
            "trim_enter_radius_px": TRIM_ENTER_RADIUS_PX,
            "trim_exit_radius_px": TRIM_EXIT_RADIUS_PX,
        },
        "best": {
            "kp_az": best_eval.kp_az,
            "kp_alt": best_eval.kp_alt,
            "kd_az": best_eval.kd_az,
            "kd_alt": best_eval.kd_alt,
            "trim_gain": best_eval.trim_gain,
            "measurement_alpha": best_eval.measurement_alpha,
            "cmd_accel_limit": best_eval.cmd_accel_limit,
            "score": best_eval.score,
            "success_count": best_eval.success_count,
            "total_tests": best_eval.total_tests,
            "median_time_s": best_eval.median_time_s,
            "mean_runaway_events": best_eval.mean_runaway_events,
            "mean_center_entries": best_eval.mean_center_entries,
            "mean_final_radius_px": best_eval.mean_final_radius_px,
            "mean_rms_radius_px": best_eval.mean_rms_radius_px,
            "mean_max_radius_px": best_eval.mean_max_radius_px,
            "mean_saturation_fraction": best_eval.mean_saturation_fraction,
        },
        "history": [
            {
                "kp_az": evaluation.kp_az,
                "kp_alt": evaluation.kp_alt,
                "kd_az": evaluation.kd_az,
                "kd_alt": evaluation.kd_alt,
                "trim_gain": evaluation.trim_gain,
                "measurement_alpha": evaluation.measurement_alpha,
                "cmd_accel_limit": evaluation.cmd_accel_limit,
                "score": evaluation.score,
                "success_count": evaluation.success_count,
                "total_tests": evaluation.total_tests,
                "median_time_s": evaluation.median_time_s,
                "mean_runaway_events": evaluation.mean_runaway_events,
                "mean_center_entries": evaluation.mean_center_entries,
                "mean_final_radius_px": evaluation.mean_final_radius_px,
                "mean_rms_radius_px": evaluation.mean_rms_radius_px,
                "mean_max_radius_px": evaluation.mean_max_radius_px,
                "mean_saturation_fraction": evaluation.mean_saturation_fraction,
                "trials": [asdict(trial) for trial in evaluation.trials],
            }
            for evaluation in history
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _input_int_default(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return int(default)
    return int(raw)


def _input_bool_default(prompt: str, default: bool) -> bool:
    suffix = "s" if default else "n"
    raw = input(f"{prompt} (s/n) [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"s", "sim", "y", "yes", "1", "true"}


def _input_str_default(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def main():
    _validate_perturbations()
    receiver = None
    sender = None
    focus_mode = "single"

    try:
        print("=== Autotune Tracker com dois telescopios ===")
        print("T1 recebe/corrige o laser. T2 envia o laser e gera perturbacoes.")
        receiver_device = _input_int_default(
            "Device Alpaca do telescopio 1 / receiver",
            DEFAULT_RECEIVER_DEVICE,
        )
        receiver_root = _input_str_default(
            "Alpaca root do telescopio 1 / receiver",
            DEFAULT_RECEIVER_ALPACA_ROOT,
        )

        sender_mode = _input_str_default(
            "Controle do telescopio 2 / sender (agent/alpaca)",
            "agent",
        ).strip().lower()
        if sender_mode not in {"agent", "alpaca"}:
            raise ValueError("Controle do sender precisa ser 'agent' ou 'alpaca'.")

        sender_device = None
        sender_root = None
        sender_agent_url = None
        sender_invert_alt = None
        if sender_mode == "agent":
            sender_agent_url = _input_str_default(
                "URL do mount_agent no PC1 / sender",
                DEFAULT_SENDER_AGENT_URL,
            )
        else:
            sender_device = _input_int_default(
                "Device Alpaca do telescopio 2 / sender",
                DEFAULT_SENDER_DEVICE,
            )
            sender_root = _input_str_default(
                "Alpaca root do telescopio 2 / sender",
                DEFAULT_SENDER_ALPACA_ROOT,
            )
            same_root = receiver_root.rstrip("/") == sender_root.rstrip("/")
            if receiver_device == sender_device and same_root:
                raise ValueError("Receiver e sender precisam ser devices Alpaca diferentes.")

            sender_invert_alt = _input_bool_default(
                "Inverter sinal de altitude do telescopio 2 / sender",
                DEFAULT_SENDER_INVERT_ALT,
            )

        focus_input = input("Modo do laser (1=foco unico, 2=dupla reflexao) [1]: ").strip() or "1"
        focus_mode = _normalize_focus_mode(focus_input)
        _configure_focus_detector(focus_mode)
        matrices = _load_tracking_calibration_matrices(focus_mode)

        receiver = AlpacaTelescope(
            receiver_device,
            "T1 receiver",
            alpaca_root=receiver_root,
            invert_az=DEFAULT_RECEIVER_INVERT_AZ,
            invert_alt=DEFAULT_RECEIVER_INVERT_ALT,
        )
        if sender_mode == "agent":
            sender = MountAgentTelescope(
                sender_agent_url,
                "T2 sender agent",
            )
        else:
            sender = AlpacaTelescope(
                sender_device,
                "T2 sender",
                alpaca_root=sender_root,
                invert_az=DEFAULT_SENDER_INVERT_AZ,
                invert_alt=sender_invert_alt,
            )
        receiver.prepare()
        sender.prepare()
        connect_camera()

        print("\nConfiguracao:")
        print(f"  T1 receiver: {receiver.base_url}")
        print(f"  T2 sender:   {sender.base_url} ({sender_mode})")
        if sender_mode == "alpaca":
            print(f"  Inverter Alt T2: {'sim' if sender_invert_alt else 'nao'}")
        print(f"  Modo foco:   {focus_mode}")
        print(f"  Matriz fine:   {matrices['fine_path']}")
        print(f"  Matriz coarse: {matrices['coarse_path']}")
        perturbation_desc = ", ".join(
            f"{p.name}(dAz={p.delta_az_deg:+.4f}, dAlt={p.delta_alt_deg:+.4f})"
            for p in PERTURBATIONS
        )
        print(f"  Perturbacoes: {perturbation_desc}")
        print(
            "\nO tracker fica ativo no T1 antes da perturbacao; "
            "o T2 se move enquanto a camera continua medindo."
        )
        input("Pressione Enter para iniciar ou Ctrl+C para cancelar...")

        metadata = {
            "receiver_device": receiver_device,
            "sender_device": sender_device,
            "sender_control_mode": sender_mode,
            "sender_agent_url": sender_agent_url,
            "receiver_base_url": receiver.base_url,
            "sender_base_url": sender.base_url,
            "receiver_invert_az": receiver.invert_az,
            "receiver_invert_alt": receiver.invert_alt,
            "sender_invert_az": sender.invert_az,
            "sender_invert_alt": sender.invert_alt,
            "focus_mode": focus_mode,
            "fine_matrix_path": matrices["fine_path"],
            "coarse_matrix_path": matrices["coarse_path"],
        }

        best_eval, history = twiddle_search(matrices, receiver, sender, focus_mode)
        print_podium(history)
        save_results(RESULTS_JSON, best_eval, history, metadata)
        print(f"\nResultados salvos em {display_path(RESULTS_JSON)}")

    except KeyboardInterrupt:
        print("\nAutotune do tracker interrompido pelo usuario.")
    finally:
        for telescope in (receiver, sender):
            if telescope is not None:
                telescope.stop_all()
        try:
            reset_camera_roi()
        except Exception:
            pass
        try:
            disconnect_camera()
        except Exception:
            pass


if __name__ == "__main__":
    main()
