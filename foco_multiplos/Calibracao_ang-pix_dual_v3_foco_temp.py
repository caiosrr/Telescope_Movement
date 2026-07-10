import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if not (ROOT_DIR / "artifact_paths.py").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import (
    display_path,
    json_candidates,
    json_output_path,
    matrix_candidates,
    matrix_output_path,
)
from foco_multiplos.Center_of_Mass_foco_temp import (
    capture_frame,
    centro_massa,
    connect_camera,
    disconnect_camera,
    get_focus_debug,
    get_focus_mode,
    set_focus_mode,
)
from controle.mount_control import ensure_connected, ensure_not_tracking, ensure_unparked, move_axes_pid_2d

FOCO_DIR = ROOT_DIR / "foco_multiplos"

EXPOSURE_SECONDS = 32e-6
SETTLE_S = 1.50
CAPTURES_PER_CENTER = 2
CAPTURES_PER_POINT = 2
MAX_SAMPLE_ATTEMPTS = 3
CENTER_DRIFT_WEIGHT = 0.50
DISCONNECT_CAMERA_ON_EXIT = True
DRIFT_LIMITS_PX = {
    "coarse": {"accept": 10.0, "warn": 15.0},
    "fine": {"accept": 6.0, "warn": 10.0},
}
MAX_COND = 1.0e4
MIN_SPREAD_DEG = 0.008
ROBUST_ITERS = 8
HUBER_K = 1.5

COARSE_RADII_DEG = [0.04]
FINE_RADII_DEG = [0.010, 0.015]

QUALITY_LIMITS = {
    "coarse": {"warn_rms_px": 10.0, "max_rms_px": 18.0},
    "fine": {"warn_rms_px": 5.0, "max_rms_px": 10.0},
}

OUTPUT_PREFIX = "calibracao_dual_v3_foco_temp"
COARSE_A_PATH = "foco_temp_A_coarse.npy"
COARSE_A_INV_PATH = "foco_temp_A_inv_coarse.npy"
FINE_A_PATH = "foco_temp_A_fine.npy"
FINE_A_INV_PATH = "foco_temp_A_inv_fine.npy"

AUDIT_DIR: Path | None = None
AUDIT_LOG = []

DIRECTIONS = [
    ("az+", +1.0, 0.0),
    ("az-", -1.0, 0.0),
    ("alt+", 0.0, +1.0),
    ("alt-", 0.0, -1.0),
    ("diag++", +1.0, +1.0),
    ("diag+-", +1.0, -1.0),
    ("diag-+", -1.0, +1.0),
    ("diag--", -1.0, -1.0),
]


@dataclass
class MedicaoCM:
    x_px: float
    y_px: float
    std_x_px: float
    std_y_px: float
    samples: int
    toca_borda: bool


@dataclass
class RegistroDual:
    regime: str
    label: str
    radius_deg: float
    target_az_deg: float
    target_alt_deg: float
    center_before_x_px: float
    center_before_y_px: float
    center_before_std_x_px: float
    center_before_std_y_px: float
    center_after_x_px: float
    center_after_y_px: float
    center_after_std_x_px: float
    center_after_std_y_px: float
    target_x_px: float
    target_y_px: float
    target_std_x_px: float
    target_std_y_px: float
    corrected_x_px: float
    corrected_y_px: float
    center_drift_px: float
    jitter_px: float


def _safe_tag(text: str) -> str:
    safe = []
    for char in text:
        if char.isalnum():
            safe.append(char)
        elif char == "+":
            safe.append("p")
        elif char == "-":
            safe.append("m")
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _save_audit_frame(frame: np.ndarray, path: Path, x_cm: float | None, y_cm: float | None, debug: dict):
    marked = frame.copy()
    if marked.ndim == 2:
        marked = cv2.cvtColor(marked, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]
    cv2.circle(marked, (int(round((w - 1) / 2)), int(round((h - 1) / 2))), 8, (255, 0, 0), -1)

    for candidate in debug.get("candidates", []):
        cx = int(round(candidate["x_cm"]))
        cy = int(round(candidate["y_cm"]))
        cv2.circle(marked, (cx, cy), 10, (0, 255, 255), 2)

    if x_cm is not None and y_cm is not None:
        cv2.circle(marked, (int(round(x_cm)), int(round(y_cm))), 8, (0, 255, 0), -1)

    cv2.imwrite(str(path), marked)


def _audit_capture(tag: str, repeat_idx: int, frame: np.ndarray, cm, debug: dict):
    if AUDIT_DIR is None:
        return

    safe_tag = _safe_tag(tag)
    filename = f"{safe_tag}_rep{repeat_idx + 1:02d}.png"
    path = AUDIT_DIR / filename
    if cm is None:
        x_cm = None
        y_cm = None
    else:
        x_cm = float(cm[0])
        y_cm = float(cm[1])

    _save_audit_frame(frame, path, x_cm, y_cm, debug)
    AUDIT_LOG.append(
        {
            "tag": tag,
            "repeat": repeat_idx + 1,
            "frame": str(path.relative_to(ROOT_DIR)),
            "ok": cm is not None,
            "x_px": x_cm,
            "y_px": y_cm,
            "focus_debug": debug,
        }
    )


def _capture_cm_estavel(exposure: float, repeats: int, audit_tag: str) -> MedicaoCM | None:
    xs = []
    ys = []
    toca_borda = False

    for repeat_idx in range(repeats):
        try:
            frame = capture_frame(exposure, light=True)
        except Exception as exc:
            print(
                f"  -> captura falhou em {audit_tag} "
                f"({repeat_idx + 1}/{repeats}): {exc}"
            )
            return None
        cm = centro_massa(frame)
        debug = get_focus_debug()
        _audit_capture(audit_tag, repeat_idx, frame, cm, debug)
        if cm is None:
            return None
        x_cm, y_cm, _, cm_toca_borda = cm
        xs.append(float(x_cm))
        ys.append(float(y_cm))
        toca_borda = toca_borda or bool(cm_toca_borda)

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    return MedicaoCM(
        x_px=float(np.median(xs_arr)),
        y_px=float(np.median(ys_arr)),
        std_x_px=float(np.std(xs_arr)),
        std_y_px=float(np.std(ys_arr)),
        samples=repeats,
        toca_borda=toca_borda,
    )


def _move_and_settle(mount: bool, delta_az: float, delta_alt: float):
    if abs(delta_az) <= 1e-6 and abs(delta_alt) <= 1e-6:
        return
    move_axes_pid_2d(mount, float(delta_az), float(delta_alt))
    time.sleep(SETTLE_S)


def _collect_bracketed_sample_once(
    regime: str,
    label: str,
    radius_deg: float,
    target_az_deg: float,
    target_alt_deg: float,
    exposure: float,
    mount: bool,
    attempt_idx: int,
) -> RegistroDual | None:
    tag_base = f"{regime}_{label}_try{attempt_idx + 1:02d}"

    center_before = _capture_cm_estavel(
        exposure,
        CAPTURES_PER_CENTER,
        audit_tag=f"{tag_base}_center_before",
    )
    if center_before is None:
        print(f"  -> centro antes falhou em {label}.")
        return None
    if center_before.toca_borda:
        print(f"  -> centro antes tocou borda em {label}; descartando.")
        return None

    target_cm = None
    returned_to_center = False
    try:
        _move_and_settle(mount, target_az_deg, target_alt_deg)
        target_cm = _capture_cm_estavel(
            exposure,
            CAPTURES_PER_POINT,
            audit_tag=f"{tag_base}_target",
        )
    except Exception as exc:
        print(f"  -> erro durante movimento/captura do ponto {label}: {exc}")
    finally:
        try:
            _move_and_settle(mount, -target_az_deg, -target_alt_deg)
            returned_to_center = True
        except Exception as exc:
            print(f"  -> erro voltando ao centro apos {label}: {exc}")

    if not returned_to_center:
        return None

    center_after = _capture_cm_estavel(
        exposure,
        CAPTURES_PER_CENTER,
        audit_tag=f"{tag_base}_center_after",
    )

    if target_cm is None:
        print(f"  -> ponto {label} sem sinal; descartando.")
        return None
    if target_cm.toca_borda:
        print(f"  -> ponto {label} tocou borda; descartando.")
        return None
    if center_after is None:
        print(f"  -> centro depois falhou em {label}.")
        return None
    if center_after.toca_borda:
        print(f"  -> centro depois tocou borda em {label}; descartando.")
        return None

    center_drift_px = float(
        np.hypot(
            center_after.x_px - center_before.x_px,
            center_after.y_px - center_before.y_px,
        )
    )
    x_ref = 0.5 * (center_before.x_px + center_after.x_px)
    y_ref = 0.5 * (center_before.y_px + center_after.y_px)
    corrected_x = target_cm.x_px - x_ref
    corrected_y = target_cm.y_px - y_ref

    jitter_px = float(
        np.hypot(target_cm.std_x_px, target_cm.std_y_px)
        + 0.5 * np.hypot(center_before.std_x_px, center_before.std_y_px)
        + 0.5 * np.hypot(center_after.std_x_px, center_after.std_y_px)
        + CENTER_DRIFT_WEIGHT * center_drift_px
    )

    return RegistroDual(
        regime=regime,
        label=label,
        radius_deg=radius_deg,
        target_az_deg=target_az_deg,
        target_alt_deg=target_alt_deg,
        center_before_x_px=center_before.x_px,
        center_before_y_px=center_before.y_px,
        center_before_std_x_px=center_before.std_x_px,
        center_before_std_y_px=center_before.std_y_px,
        center_after_x_px=center_after.x_px,
        center_after_y_px=center_after.y_px,
        center_after_std_x_px=center_after.std_x_px,
        center_after_std_y_px=center_after.std_y_px,
        target_x_px=target_cm.x_px,
        target_y_px=target_cm.y_px,
        target_std_x_px=target_cm.std_x_px,
        target_std_y_px=target_cm.std_y_px,
        corrected_x_px=float(corrected_x),
        corrected_y_px=float(corrected_y),
        center_drift_px=center_drift_px,
        jitter_px=float(jitter_px),
    )


def _collect_bracketed_sample(
    regime: str,
    label: str,
    radius_deg: float,
    target_az_deg: float,
    target_alt_deg: float,
    exposure: float,
    mount: bool,
) -> RegistroDual | None:
    limits = DRIFT_LIMITS_PX[regime]
    best_record = None

    for attempt_idx in range(MAX_SAMPLE_ATTEMPTS):
        if attempt_idx > 0:
            print(f"  -> repetindo {label}: drift alto na tentativa anterior.")
        registro = _collect_bracketed_sample_once(
            regime=regime,
            label=label,
            radius_deg=radius_deg,
            target_az_deg=target_az_deg,
            target_alt_deg=target_alt_deg,
            exposure=exposure,
            mount=mount,
            attempt_idx=attempt_idx,
        )
        if registro is None:
            continue
        if best_record is None or registro.center_drift_px < best_record.center_drift_px:
            best_record = registro
        if registro.center_drift_px <= limits["accept"]:
            return registro

    if best_record is None:
        return None

    if best_record.center_drift_px >= limits["warn"]:
        print(
            f"  -> aviso: usando {label} com drift centro alto "
            f"({best_record.center_drift_px:.2f}px); peso reduzido no ajuste."
        )
    return best_record


def _build_star_sequence(radii_deg: list[float]):
    sequence = []
    for radius_deg in radii_deg:
        for direction_label, sign_az, sign_alt in DIRECTIONS:
            label = f"{direction_label}@{radius_deg:.4f}"
            sequence.append(
                (
                    label,
                    radius_deg,
                    float(radius_deg * sign_az),
                    float(radius_deg * sign_alt),
                )
            )
    return sequence


def _collect_regime(
    regime: str,
    radii_deg: list[float],
    exposure: float,
    mount: bool,
) -> list[RegistroDual]:
    print(f"\n{'=' * 72}")
    print(f"Coleta {regime.upper()} | raios {', '.join(f'{r:.4f}' for r in radii_deg)} deg")
    print(f"{'=' * 72}")

    registros: list[RegistroDual] = []
    for idx, (label, radius_deg, target_az_deg, target_alt_deg) in enumerate(_build_star_sequence(radii_deg), start=1):
        print(
            f"[{idx:02d}] {label} | "
            f"dAz={target_az_deg:+.4f} deg dAlt={target_alt_deg:+.4f} deg"
        )
        registro = _collect_bracketed_sample(
            regime=regime,
            label=label,
            radius_deg=radius_deg,
            target_az_deg=target_az_deg,
            target_alt_deg=target_alt_deg,
            exposure=exposure,
            mount=mount,
        )
        if registro is None:
            continue
        registros.append(registro)
        print(
            f"  -> corrigido: x={registro.corrected_x_px:+.2f}px "
            f"y={registro.corrected_y_px:+.2f}px | "
            f"drift={registro.center_drift_px:.2f}px | jitter={registro.jitter_px:.2f}px"
        )

    print(f"Registros validos {regime}: {len(registros)}")
    return registros


def _evaluate_matrix(A: np.ndarray, registros: list[RegistroDual]):
    offsets = np.array(
        [[r.target_az_deg, r.target_alt_deg] for r in registros],
        dtype=float,
    )
    x = np.array([r.corrected_x_px for r in registros], dtype=float)
    y = np.array([r.corrected_y_px for r in registros], dtype=float)

    pred_x = offsets @ A[0]
    pred_y = offsets @ A[1]
    residuo = np.sqrt((pred_x - x) ** 2 + (pred_y - y) ** 2)
    return {
        "rms_px": float(np.sqrt(np.mean(residuo ** 2))),
        "max_px": float(np.max(residuo)),
        "mean_px": float(np.mean(residuo)),
    }


def _fit_robusto_sem_intercepto(registros: list[RegistroDual], regime: str):
    if len(registros) < 8:
        raise RuntimeError(f"Poucos pontos em {regime}: {len(registros)}.")

    offsets = np.array(
        [[r.target_az_deg, r.target_alt_deg] for r in registros],
        dtype=float,
    )
    x = np.array([r.corrected_x_px for r in registros], dtype=float)
    y = np.array([r.corrected_y_px for r in registros], dtype=float)
    jitter = np.array([max(r.jitter_px, 0.5) for r in registros], dtype=float)

    spread_az = float(np.ptp(offsets[:, 0]))
    spread_alt = float(np.ptp(offsets[:, 1]))
    if spread_az < MIN_SPREAD_DEG or spread_alt < MIN_SPREAD_DEG:
        raise RuntimeError(
            f"{regime}: pouca excitacao dos eixos (spread_az={spread_az:.4f}, "
            f"spread_alt={spread_alt:.4f})."
        )
    if np.linalg.matrix_rank(offsets) < 2:
        raise RuntimeError(f"{regime}: offsets degenerados.")

    base_weights = 1.0 / np.square(jitter)
    base_weights /= np.max(base_weights)
    robust_weights = np.ones(len(registros), dtype=float)

    coef_x = None
    coef_y = None
    for _ in range(ROBUST_ITERS):
        weights = np.clip(base_weights * robust_weights, 1e-4, 1.0)
        sqrt_w = np.sqrt(weights)
        M_w = offsets * sqrt_w[:, None]
        x_w = x * sqrt_w
        y_w = y * sqrt_w

        coef_x = np.linalg.lstsq(M_w, x_w, rcond=None)[0]
        coef_y = np.linalg.lstsq(M_w, y_w, rcond=None)[0]

        pred_x = offsets @ coef_x
        pred_y = offsets @ coef_y
        residuo = np.sqrt((pred_x - x) ** 2 + (pred_y - y) ** 2)
        mad = float(np.median(np.abs(residuo - np.median(residuo))))
        scale = max(1e-6, 1.4826 * mad)
        cutoff = HUBER_K * scale

        new_weights = np.ones_like(robust_weights)
        mask = residuo > cutoff
        new_weights[mask] = cutoff / np.maximum(residuo[mask], 1e-9)

        if np.allclose(new_weights, robust_weights, atol=1e-3, rtol=1e-2):
            robust_weights = new_weights
            break
        robust_weights = new_weights

    A = np.array(
        [
            [coef_x[0], coef_x[1]],
            [coef_y[0], coef_y[1]],
        ],
        dtype=float,
    )
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > MAX_COND:
        raise RuntimeError(f"{regime}: matriz mal condicionada (cond={cond:.2e}).")

    A_inv = np.linalg.inv(A)
    fit_metrics = _evaluate_matrix(A, registros)
    downweighted = int(np.count_nonzero(robust_weights < 0.99))
    effective_weights = np.clip(base_weights * robust_weights, 1e-4, 1.0)

    limits = QUALITY_LIMITS[regime]
    quality_ok = fit_metrics["rms_px"] <= limits["max_rms_px"]
    quality_warning = None
    if not quality_ok:
        quality_warning = (
            f"{regime}: RMS alto ({fit_metrics['rms_px']:.2f}px) para o alvo local "
            f"de {limits['max_rms_px']:.2f}px."
        )
    elif fit_metrics["rms_px"] > limits["warn_rms_px"]:
        quality_warning = (
            f"{regime}: RMS moderado ({fit_metrics['rms_px']:.2f}px), ainda aceitavel."
        )

    return {
        "A": A,
        "A_inv": A_inv,
        "spread_az_deg": spread_az,
        "spread_alt_deg": spread_alt,
        "condition_number": cond,
        "rms_residual_px": fit_metrics["rms_px"],
        "max_residual_px": fit_metrics["max_px"],
        "mean_residual_px": fit_metrics["mean_px"],
        "num_points": len(registros),
        "downweighted_points": downweighted,
        "quality_ok": quality_ok,
        "quality_warning": quality_warning,
        "weights": effective_weights.tolist(),
    }


def _load_existing_matrix():
    for meta_path in json_candidates("calibracao_meta.json"):
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return {
                "source": display_path(meta_path),
                "A": np.array(meta["A"], dtype=float),
                "meta": meta,
            }
    for matrix_path in matrix_candidates("calibracao_A.npy"):
        if matrix_path.exists():
            return {
                "source": display_path(matrix_path),
                "A": np.load(matrix_path),
                "meta": None,
            }
    return None


def _compare_with_existing(existing, coarse_result, fine_result, coarse_records, fine_records):
    if existing is None:
        return None

    A_old = existing["A"]
    comparison = {
        "source": existing["source"],
        "old_on_coarse": _evaluate_matrix(A_old, coarse_records),
        "old_on_fine": _evaluate_matrix(A_old, fine_records),
        "new_coarse_on_coarse": _evaluate_matrix(coarse_result["A"], coarse_records),
        "new_fine_on_fine": _evaluate_matrix(fine_result["A"], fine_records),
        "new_coarse_on_fine": _evaluate_matrix(coarse_result["A"], fine_records),
        "new_fine_on_coarse": _evaluate_matrix(fine_result["A"], coarse_records),
    }
    if existing["meta"] is not None:
        comparison["old_saved_rms_px"] = existing["meta"].get("rms_residual_px")
        comparison["old_saved_selection_mode"] = existing["meta"].get("selection_mode")
    return comparison


def _save_dual_results(coarse_result, fine_result, coarse_records, fine_records, comparison, output_dir=None):
    coarse_a_path = matrix_output_path(COARSE_A_PATH, output_dir)
    coarse_a_inv_path = matrix_output_path(COARSE_A_INV_PATH, output_dir)
    fine_a_path = matrix_output_path(FINE_A_PATH, output_dir)
    fine_a_inv_path = matrix_output_path(FINE_A_INV_PATH, output_dir)

    np.save(coarse_a_path, coarse_result["A"])
    np.save(coarse_a_inv_path, coarse_result["A_inv"])
    np.save(fine_a_path, fine_result["A"])
    np.save(fine_a_inv_path, fine_result["A_inv"])

    payload = {
        "timestamp_epoch": time.time(),
        "config": {
            "exposure_seconds": EXPOSURE_SECONDS,
            "settle_s": SETTLE_S,
            "captures_per_center": CAPTURES_PER_CENTER,
            "captures_per_point": CAPTURES_PER_POINT,
            "max_sample_attempts": MAX_SAMPLE_ATTEMPTS,
            "center_drift_weight": CENTER_DRIFT_WEIGHT,
            "drift_limits_px": DRIFT_LIMITS_PX,
            "coarse_radii_deg": COARSE_RADII_DEG,
            "fine_radii_deg": FINE_RADII_DEG,
            "directions": DIRECTIONS,
            "robust_iters": ROBUST_ITERS,
            "huber_k": HUBER_K,
            "focus_mode": get_focus_mode(),
            "focus_method": "temporary locked-focus center of mass",
            "focus_audit_dir": None if AUDIT_DIR is None else str(AUDIT_DIR.relative_to(ROOT_DIR)),
        },
        "focus_audit": AUDIT_LOG,
        "coarse": {
            **{k: v for k, v in coarse_result.items() if k not in {"A", "A_inv", "weights"}},
            "A": coarse_result["A"].tolist(),
            "A_inv": coarse_result["A_inv"].tolist(),
            "records": [asdict(r) for r in coarse_records],
        },
        "fine": {
            **{k: v for k, v in fine_result.items() if k not in {"A", "A_inv", "weights"}},
            "A": fine_result["A"].tolist(),
            "A_inv": fine_result["A_inv"].tolist(),
            "records": [asdict(r) for r in fine_records],
        },
        "comparison_with_existing": comparison,
    }

    meta_output_path = json_output_path(f"{OUTPUT_PREFIX}_meta.json", output_dir)
    with meta_output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _print_summary(name: str, result):
    quality = "OK" if result["quality_ok"] else "RESSALVAS"
    print(
        f"{name}: pontos={result['num_points']} | "
        f"RMS={result['rms_residual_px']:.2f}px | "
        f"max={result['max_residual_px']:.2f}px | "
        f"cond={result['condition_number']:.2e} | "
        f"pesos_rebaixados={result['downweighted_points']} | "
        f"{quality}"
    )
    if result["quality_warning"]:
        print(f"  -> {result['quality_warning']}")


def _novo_diretorio_auditoria() -> Path:
    base_dir = FOCO_DIR / "auditoria_foco_temp"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for suffix in range(100):
        run_name = f"run_{timestamp}" if suffix == 0 else f"run_{timestamp}_{suffix:02d}"
        audit_dir = base_dir / run_name
        try:
            audit_dir.mkdir(parents=True, exist_ok=False)
            return audit_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Nao consegui criar diretorio unico de auditoria em {base_dir}")


def main():
    global AUDIT_DIR, AUDIT_LOG

    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    focus_input = input("Modo do laser (1=foco unico, 2=dupla reflexao) [2]: ").strip() or "2"
    focus_mode = set_focus_mode(focus_input)
    mount = True
    AUDIT_LOG = []
    AUDIT_DIR = _novo_diretorio_auditoria()

    print(f"Modo de foco temporario: {focus_mode}")
    print("Usando montagem real. Esta versao temporaria nao pergunta por simulador.")
    print(f"Auditoria visual do foco em: {AUDIT_DIR}")

    try:
        connect_camera()

        coarse_records = _collect_regime(
            regime="coarse",
            radii_deg=COARSE_RADII_DEG,
            exposure=EXPOSURE_SECONDS,
            mount=mount,
        )
        fine_records = _collect_regime(
            regime="fine",
            radii_deg=FINE_RADII_DEG,
            exposure=EXPOSURE_SECONDS,
            mount=mount,
        )

        coarse_result = _fit_robusto_sem_intercepto(coarse_records, regime="coarse")
        fine_result = _fit_robusto_sem_intercepto(fine_records, regime="fine")
        existing = _load_existing_matrix()
        comparison = _compare_with_existing(
            existing,
            coarse_result,
            fine_result,
            coarse_records,
            fine_records,
        )
        _save_dual_results(
            coarse_result,
            fine_result,
            coarse_records,
            fine_records,
            comparison,
        )

        print("\n=== Resumo Nova Calibracao Dual V3 ===")
        _print_summary("COARSE", coarse_result)
        _print_summary("FINE", fine_result)
        if comparison is not None:
            print("\n=== Comparacao com calibracao atual ===")
            print(
                f"Atual no dataset COARSE: RMS={comparison['old_on_coarse']['rms_px']:.2f}px | "
                f"max={comparison['old_on_coarse']['max_px']:.2f}px"
            )
            print(
                f"Atual no dataset FINE: RMS={comparison['old_on_fine']['rms_px']:.2f}px | "
                f"max={comparison['old_on_fine']['max_px']:.2f}px"
            )
            print(
                f"Nova COARSE no dataset COARSE: RMS={comparison['new_coarse_on_coarse']['rms_px']:.2f}px | "
                f"max={comparison['new_coarse_on_coarse']['max_px']:.2f}px"
            )
            print(
                f"Nova FINE no dataset FINE: RMS={comparison['new_fine_on_fine']['rms_px']:.2f}px | "
                f"max={comparison['new_fine_on_fine']['max_px']:.2f}px"
            )

        print(
            f"\nArquivos salvos: "
            f"{display_path(matrix_output_path(COARSE_A_PATH, ROOT_DIR))}, "
            f"{display_path(matrix_output_path(COARSE_A_INV_PATH, ROOT_DIR))}, "
            f"{display_path(matrix_output_path(FINE_A_PATH, ROOT_DIR))}, "
            f"{display_path(matrix_output_path(FINE_A_INV_PATH, ROOT_DIR))}, "
            f"{display_path(json_output_path(f'{OUTPUT_PREFIX}_meta.json', ROOT_DIR))}"
        )

    except KeyboardInterrupt:
        print("\nCalibracao dual interrompida pelo usuario.")
    except Exception as exc:
        print(f"\nErro na calibracao dual V3: {exc}")
    finally:
        if DISCONNECT_CAMERA_ON_EXIT:
            try:
                disconnect_camera()
            except Exception as exc:
                print(f"Aviso: nao consegui desconectar a camera pelo Alpaca: {exc}")
        else:
            print("Camera mantida conectada ao final da calibracao.")


if __name__ == "__main__":
    main()
