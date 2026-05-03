import cv2
import copy
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if not (ROOT_DIR / "Center_of_Mass.py").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Center_of_Mass import (
    centro_camera,
    connect_camera,
    disconnect_camera,
    fetch_image_array,
    start_exposure,
    wait_until_image_ready,
)
from artifact_paths import display_path, matrix_candidates
from PID_controll import ensure_connected, ensure_not_tracking, ensure_unparked
from mov_simultaneo import move_axes_pid_2d


FOCUS_MODE = "single"
RAW_SIGNAL_MIN = 200.0
DUAL_THRESHOLD_PERCENT = 0.25
LOCAL_RADIUS_PX = 90
MIN_LOCAL_PIXELS = 8
MORPH_KERNEL_SIZE = 7
LOCK_FOCUS_IDENTITY = True
LOCK_MIN_SIMILARITY = 0.35
LOCK_STRONG_SIMILARITY = 0.65
lim_px = 2.0
EXPOSURE_SECONDS = 32e-6
LAST_CAPTURE_STATS = {}
LAST_RAW_FRAME = None
LAST_FOCUS_DEBUG = {}
FOCUS_LOCK = {
    "active": False,
    "primary": None,
    "secondary": None,
    "last_x": None,
    "last_y": None,
}


def set_focus_mode(mode: str) -> str:
    global FOCUS_MODE
    normalized = str(mode).strip().lower()
    if normalized in {"2", "dual", "duplo", "dois", "two"}:
        FOCUS_MODE = "dual"
    else:
        FOCUS_MODE = "single"
    reset_focus_lock()
    return FOCUS_MODE


def get_focus_mode() -> str:
    return FOCUS_MODE


def get_focus_debug() -> dict:
    return copy.deepcopy(LAST_FOCUS_DEBUG)


def reset_focus_lock() -> None:
    FOCUS_LOCK["active"] = False
    FOCUS_LOCK["primary"] = None
    FOCUS_LOCK["secondary"] = None
    FOCUS_LOCK["last_x"] = None
    FOCUS_LOCK["last_y"] = None


def capture_frame(exposure_seconds: float, light: bool = True) -> np.ndarray:
    global LAST_CAPTURE_STATS, LAST_RAW_FRAME

    start_exposure(exposure_seconds, light=light)
    wait_until_image_ready()
    frame = fetch_image_array().astype(np.float32)

    min_val = float(frame.min())
    max_val = float(frame.max())
    median_val = float(np.median(frame))
    std_val = float(np.std(frame))

    if max_val < RAW_SIGNAL_MIN or max_val <= min_val:
        norm = np.zeros_like(frame, dtype=np.uint8)
        pedestal = min_val
    else:
        pedestal = max(min_val, median_val + 0.5 * std_val)
        if max_val <= pedestal:
            pedestal = min_val
        norm = np.clip((frame - pedestal) / (max_val - pedestal + 1e-6), 0, 1)
        norm = (norm * 255).astype(np.uint8)

    norm = np.rot90(norm, 2)
    LAST_RAW_FRAME = np.rot90(frame, 2)
    LAST_CAPTURE_STATS = {
        "raw_min": min_val,
        "raw_max": max_val,
        "raw_median": median_val,
        "raw_std": std_val,
        "pedestal": float(pedestal),
        "norm_max": float(norm.max()),
        "norm_nonzero": int(np.count_nonzero(norm)),
    }

    return norm


def _as_gray_float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return frame.mean(axis=2).astype(np.float32)
    return frame.astype(np.float32, copy=True)


def _centro_massa_padrao(frame_gray: np.ndarray, threshold_percent: float):
    max_val = float(frame_gray.max())
    if max_val <= 0:
        return None

    threshold = max_val * threshold_percent
    weights = frame_gray.copy()
    weights[weights < threshold] = 0
    total = float(weights.sum())
    if total <= 0:
        return None

    h, w = weights.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    x_cm = float((xx * weights).sum() / total)
    y_cm = float((yy * weights).sum() / total)
    ix = int(np.clip(round(x_cm), 0, w - 1))
    iy = int(np.clip(round(y_cm), 0, h - 1))
    toca_borda = bool(
        np.any(weights[0, :])
        or np.any(weights[-1, :])
        or np.any(weights[:, 0])
        or np.any(weights[:, -1])
    )
    return x_cm, y_cm, float(weights[iy, ix]), toca_borda


def _raw_signal_frame(frame_gray: np.ndarray) -> np.ndarray:
    if isinstance(LAST_RAW_FRAME, np.ndarray) and LAST_RAW_FRAME.shape == frame_gray.shape:
        pedestal = float(LAST_CAPTURE_STATS.get("pedestal", 0.0))
        return np.clip(LAST_RAW_FRAME.astype(np.float32) - pedestal, 0, None)
    return frame_gray.astype(np.float32, copy=False)


def _similarity(candidate: dict, reference: dict | None) -> float:
    if reference is None:
        return 0.0

    scores = []
    for key in ("raw_peak", "raw_total", "area"):
        cand = max(float(candidate[key]), 1e-6)
        ref = max(float(reference[key]), 1e-6)
        ratio = min(cand / ref, ref / cand)
        scores.append(float(np.clip(ratio, 0.0, 1.0)))

    return float((0.5 * scores[0]) + (0.35 * scores[1]) + (0.15 * scores[2]))


def _lock_reference_from(candidate: dict) -> dict:
    return {
        "raw_peak": float(candidate["raw_peak"]),
        "raw_total": float(candidate["raw_total"]),
        "area": float(candidate["area"]),
    }


def _focus_lock_snapshot() -> dict:
    return {
        "active": bool(FOCUS_LOCK["active"]),
        "primary": copy.deepcopy(FOCUS_LOCK["primary"]),
        "secondary": copy.deepcopy(FOCUS_LOCK["secondary"]),
        "last_x": FOCUS_LOCK["last_x"],
        "last_y": FOCUS_LOCK["last_y"],
    }


def _candidate_debug(candidate: dict, primary: dict | None, secondary: dict | None) -> dict:
    return {
        "x_cm": float(candidate["x_cm"]),
        "y_cm": float(candidate["y_cm"]),
        "area": int(candidate["area"]),
        "raw_peak": float(candidate["raw_peak"]),
        "raw_total": float(candidate["raw_total"]),
        "toca_borda": bool(candidate["toca_borda"]),
        "similarity_primary": float(_similarity(candidate, primary)) if primary is not None else None,
        "similarity_secondary": float(_similarity(candidate, secondary)) if secondary is not None else None,
    }


def _update_focus_lock(candidate: dict, candidates: list[dict]) -> None:
    if not LOCK_FOCUS_IDENTITY:
        return

    if not FOCUS_LOCK["active"]:
        ordered = sorted(candidates, key=lambda item: item["raw_total"], reverse=True)
        FOCUS_LOCK["active"] = True
        FOCUS_LOCK["primary"] = _lock_reference_from(candidate)
        FOCUS_LOCK["secondary"] = (
            _lock_reference_from(ordered[1]) if len(ordered) >= 2 else None
        )
    elif not candidate["toca_borda"]:
        primary = FOCUS_LOCK["primary"]
        if primary is not None:
            # Atualiza devagar para aceitar variacoes reais sem esquecer a identidade inicial.
            for key in ("raw_peak", "raw_total", "area"):
                primary[key] = (0.85 * float(primary[key])) + (0.15 * float(candidate[key]))

    FOCUS_LOCK["last_x"] = float(candidate["x_cm"])
    FOCUS_LOCK["last_y"] = float(candidate["y_cm"])


def _select_focus_candidate(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    if not LOCK_FOCUS_IDENTITY or not FOCUS_LOCK["active"]:
        return max(candidates, key=lambda item: item["raw_total"])

    primary = FOCUS_LOCK["primary"]
    secondary = FOCUS_LOCK["secondary"]

    best = None
    best_score = -1.0
    for candidate in candidates:
        primary_score = _similarity(candidate, primary)
        secondary_score = _similarity(candidate, secondary)

        if secondary is not None and secondary_score > primary_score + 0.12:
            continue
        if primary_score < LOCK_MIN_SIMILARITY:
            continue

        if FOCUS_LOCK["last_x"] is not None and FOCUS_LOCK["last_y"] is not None:
            dist = float(
                np.hypot(
                    candidate["x_cm"] - FOCUS_LOCK["last_x"],
                    candidate["y_cm"] - FOCUS_LOCK["last_y"],
                )
            )
            dist_score = 1.0 / (1.0 + (dist / 300.0))
        else:
            dist_score = 1.0

        score = (0.8 * primary_score) + (0.2 * dist_score)
        if score > best_score:
            best = candidate
            best_score = score

    if best is None:
        return None

    if len(candidates) == 1 and secondary is not None:
        primary_score = _similarity(best, primary)
        secondary_score = _similarity(best, secondary)
        if primary_score < LOCK_STRONG_SIMILARITY and secondary_score >= primary_score:
            return None

    return best


def _find_focus_candidates(frame_gray: np.ndarray, threshold_percent: float) -> list[dict]:
    max_val = float(frame_gray.max())
    if max_val <= 0:
        return []

    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    h, w = frame_gray.shape
    raw_signal = _raw_signal_frame(frame_gray)
    threshold = max_val * threshold_percent
    mask = (blurred >= threshold).astype(np.uint8)
    if MORPH_KERNEL_SIZE > 1:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    if num_labels <= 1:
        return []

    candidates = []
    yy, xx = np.indices(frame_gray.shape, dtype=np.float32)
    for label in range(1, num_labels):
        selected_mask = labels == label
        area = int(np.count_nonzero(selected_mask))
        if area < MIN_LOCAL_PIXELS:
            continue

        ys, xs = np.where(selected_mask)
        peak_index = int(np.argmax(frame_gray[selected_mask]))
        peak_y = int(ys[peak_index])
        peak_x = int(xs[peak_index])

        y0 = max(0, peak_y - LOCAL_RADIUS_PX)
        y1 = min(h, peak_y + LOCAL_RADIUS_PX + 1)
        x0 = max(0, peak_x - LOCAL_RADIUS_PX)
        x1 = min(w, peak_x + LOCAL_RADIUS_PX + 1)
        local_window = np.zeros_like(selected_mask)
        local_window[y0:y1, x0:x1] = True
        selected_mask = selected_mask & local_window
        area = int(np.count_nonzero(selected_mask))
        if area < MIN_LOCAL_PIXELS:
            continue

        raw_weights = np.where(selected_mask, raw_signal, 0)
        weights = raw_weights
        total = float(weights.sum())
        raw_total = float(raw_weights.sum())
        raw_peak = float(raw_weights.max())
        if total <= 0 or raw_total <= 0:
            continue

        x_cm = float((xx * weights).sum() / total)
        y_cm = float((yy * weights).sum() / total)
        ix = int(np.clip(round(x_cm), 0, w - 1))
        iy = int(np.clip(round(y_cm), 0, h - 1))
        toca_borda = bool(
            np.any(selected_mask[0, :])
            or np.any(selected_mask[-1, :])
            or np.any(selected_mask[:, 0])
            or np.any(selected_mask[:, -1])
        )

        candidates.append(
            {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "intensity": float(frame_gray[iy, ix]),
                "toca_borda": toca_borda,
                "area": area,
                "raw_peak": raw_peak,
                "raw_total": raw_total,
            }
        )

    return candidates


def _centro_foco_principal(frame_gray: np.ndarray, threshold_percent: float):
    global LAST_FOCUS_DEBUG

    lock_before = _focus_lock_snapshot()
    candidates = _find_focus_candidates(frame_gray, threshold_percent)
    candidate = _select_focus_candidate(candidates)
    if candidate is None:
        LAST_FOCUS_DEBUG = {
            "mode": FOCUS_MODE,
            "selected": None,
            "candidate_count": len(candidates),
            "lock_before": lock_before,
            "lock_after": _focus_lock_snapshot(),
            "candidates": [
                _candidate_debug(item, lock_before["primary"], lock_before["secondary"])
                for item in candidates
            ],
            "rejected": True,
        }
        return None

    _update_focus_lock(candidate, candidates)
    lock_after = _focus_lock_snapshot()
    LAST_FOCUS_DEBUG = {
        "mode": FOCUS_MODE,
        "selected": _candidate_debug(candidate, lock_before["primary"], lock_before["secondary"]),
        "candidate_count": len(candidates),
        "lock_before": lock_before,
        "lock_after": lock_after,
        "candidates": [
            _candidate_debug(item, lock_before["primary"], lock_before["secondary"])
            for item in candidates
        ],
        "rejected": False,
    }
    return (
        candidate["x_cm"],
        candidate["y_cm"],
        candidate["intensity"],
        candidate["toca_borda"],
    )


def centro_massa(frame: np.ndarray, threshold_percent: float | None = None):
    global LAST_FOCUS_DEBUG

    frame_gray = _as_gray_float(frame)
    if FOCUS_MODE == "dual":
        threshold = DUAL_THRESHOLD_PERCENT if threshold_percent is None else threshold_percent
        return _centro_foco_principal(frame_gray, threshold)

    threshold = 0.5 if threshold_percent is None else threshold_percent
    cm = _centro_massa_padrao(frame_gray, threshold)
    LAST_FOCUS_DEBUG = {
        "mode": FOCUS_MODE,
        "selected": None
        if cm is None
        else {
            "x_cm": float(cm[0]),
            "y_cm": float(cm[1]),
            "intensity": float(cm[2]),
            "toca_borda": bool(cm[3]),
        },
        "candidate_count": None,
        "lock_before": None,
        "lock_after": None,
        "candidates": [],
        "rejected": cm is None,
    }
    return cm


def _load_A_inv() -> np.ndarray | None:
    candidates = matrix_candidates(
        "foco_temp_A_inv_fine.npy",
        "foco_temp_A_inv_coarse.npy",
        "A_inv_fine.npy",
        "A_inv_coarse.npy",
        "calibracao_A_inv.npy",
    )

    try:
        for candidate in candidates:
            if not candidate.exists():
                continue
            A_inv = np.load(candidate)
            if A_inv.shape != (2, 2):
                raise ValueError(f"Matriz {display_path(candidate)} com shape invalido.")
            print(f"Matriz carregada: {display_path(candidate)}")
            return A_inv

        raise FileNotFoundError(f"Testei: {', '.join(str(path) for path in candidates)}")
    except Exception as exc:
        print(f"\nNao foi possivel carregar a matriz de centralizacao: {exc}")
        print("Execute primeiro a calibracao 2D para gerar este arquivo.")
        return None


def _save_marked_frame(
    frame: np.ndarray,
    path: Path,
    cx: float,
    cy: float,
    x_cm: float,
    y_cm: float,
    x_cm0: float | None = None,
    y_cm0: float | None = None,
) -> None:
    marked = frame.copy()
    if marked.ndim == 2:
        marked = cv2.cvtColor(marked, cv2.COLOR_GRAY2BGR)

    c_ix, c_iy = int(round(cx)), int(round(cy))
    cm_ix, cm_iy = int(round(x_cm)), int(round(y_cm))

    cv2.circle(marked, (c_ix, c_iy), 8, (255, 0, 0), -1)
    if x_cm0 is not None and y_cm0 is not None:
        cm0_ix, cm0_iy = int(round(x_cm0)), int(round(y_cm0))
        cv2.circle(marked, (cm0_ix, cm0_iy), 8, (0, 255, 0), -1)
        cv2.line(marked, (cm0_ix, cm0_iy), (cm_ix, cm_iy), (0, 255, 255), 2)
        cv2.circle(marked, (cm_ix, cm_iy), 8, (0, 0, 255), -1)
    else:
        cv2.circle(marked, (cm_ix, cm_iy), 8, (0, 255, 0), -1)

    cv2.imwrite(str(path), marked)


def main() -> None:
    mode = input("Modo do laser (1=foco unico, 2=dupla reflexao) [2]: ").strip() or "2"
    set_focus_mode(mode)
    connect_camera()
    try:
        frame = capture_frame(EXPOSURE_SECONDS, light=True)
        cm = centro_massa(frame)
        if cm is None:
            print("Nao foi possivel medir o centro do foco travado.")
            print(
                "Diagnostico do frame: "
                f"shape={frame.shape}, min={float(frame.min()):.1f}, "
                f"max={float(frame.max()):.1f}, pixels_nao_zero={int(np.count_nonzero(frame))}"
            )
            if LAST_CAPTURE_STATS:
                print(
                    "Diagnostico bruto: "
                    f"raw_min={LAST_CAPTURE_STATS['raw_min']:.1f}, "
                    f"raw_max={LAST_CAPTURE_STATS['raw_max']:.1f}, "
                    f"raw_median={LAST_CAPTURE_STATS['raw_median']:.1f}, "
                    f"raw_std={LAST_CAPTURE_STATS['raw_std']:.1f}, "
                    f"pedestal={LAST_CAPTURE_STATS['pedestal']:.1f}"
                )
            output_path = ROOT_DIR / "temporarios" / "foco_temp_ultimo_frame.png"
            cv2.imwrite(str(output_path), frame)
            print(f"Ultimo frame salvo em: {output_path}")
            return

        x_cm, y_cm, intensidade, toca_borda = cm
        cx, cy = centro_camera(frame)
        print(f"Modo: {FOCUS_MODE}")
        print(f"Centro medido: x={x_cm:.2f}px y={y_cm:.2f}px")
        print(f"Deslocamento: dx={x_cm - cx:+.2f}px dy={y_cm - cy:+.2f}px")
        print(f"Intensidade no centro: {intensidade:.1f} | toca_borda={toca_borda}")
        if LAST_CAPTURE_STATS:
            print(
                "Captura: "
                f"raw_max={LAST_CAPTURE_STATS['raw_max']:.1f}, "
                f"norm_max={LAST_CAPTURE_STATS['norm_max']:.1f}, "
                f"norm_pixels_nao_zero={LAST_CAPTURE_STATS['norm_nonzero']}"
            )

        output_path = ROOT_DIR / "temporarios" / "foco_temp_inicial_cm_centro.png"
        _save_marked_frame(frame, output_path, cx, cy, x_cm, y_cm)
        print(f"Frame inicial salvo em: {output_path}")

        move = input("\nDeseja centralizar o laser? (s/n): ").strip().lower()
        if move != "s":
            print("Centralizacao automatica cancelada pelo usuario.")
            return

        A_inv = _load_A_inv()
        if A_inv is None:
            return

        ensure_connected()
        ensure_unparked()
        ensure_not_tracking()

        x_cm0, y_cm0 = x_cm, y_cm
        usar_mount = True

        while True:
            dx = x_cm - cx
            dy = y_cm - cy
            print(f"\nDeslocamento atual (cm - centro): dx = {dx:+.3f}, dy = {dy:+.3f} px")

            if abs(dx) <= lim_px and abs(dy) <= lim_px:
                print("Dentro da tolerancia em pixels. Encerrando correcoes.")
                break

            vec_px = np.array([-dx, -dy])
            correcao = A_inv @ vec_px
            dAz_deg = float(correcao[0])
            dAlt_deg = float(correcao[1])

            print("--- Correcao de apontamento usando A^{-1} ---")
            print(f"Movimento alvo: dAz = {dAz_deg:+.6f} deg, dAlt = {dAlt_deg:+.6f} deg")

            move_axes_pid_2d(usar_mount, dAz_deg, dAlt_deg)

            print("Capturando novo frame para proxima iteracao...")
            frame = capture_frame(EXPOSURE_SECONDS, light=True)
            cm = centro_massa(frame)
            if cm is None:
                print("Imagem sem sinal ou foco travado nao encontrado apos correcao; interrompendo.")
                output_path = ROOT_DIR / "temporarios" / "foco_temp_ultimo_frame.png"
                cv2.imwrite(str(output_path), frame)
                print(f"Ultimo frame salvo em: {output_path}")
                break

            x_cm, y_cm, intensidade, toca_borda = cm

        print(f"\nCM final: ({x_cm:.2f}, {y_cm:.2f})")
        dx_final = x_cm - cx
        dy_final = y_cm - cy
        print(f"Deslocamento final (cm - centro): dx = {dx_final:+.3f}, dy = {dy_final:+.3f} px")

        output_path = ROOT_DIR / "temporarios" / "foco_temp_final_cm_trajetoria.png"
        _save_marked_frame(frame, output_path, cx, cy, x_cm, y_cm, x_cm0, y_cm0)
        print(f"Frame final salvo em: {output_path}")
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario (Ctrl+C). Encerrando de forma limpa...")
    finally:
        disconnect_camera()


if __name__ == "__main__":
    main()
