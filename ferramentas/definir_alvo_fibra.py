import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import display_path
from controle.alvo_alinhamento import centro_frame, salvar_alvo
from foco_multiplos.Center_of_Mass_foco_temp import (
    EXPOSURE_SECONDS,
    capture_frame,
    centro_massa,
    connect_camera,
    disconnect_camera,
    get_focus_mode,
    set_focus_mode,
)


SAMPLES = 5


def _save_preview(frame: np.ndarray, target_x: float, target_y: float) -> Path:
    marked = frame.copy()
    if marked.ndim == 2:
        marked = cv2.cvtColor(marked, cv2.COLOR_GRAY2BGR)

    cv2.circle(marked, (int(round(target_x)), int(round(target_y))), 10, (0, 255, 0), -1)
    cx, cy = centro_frame(frame)
    cv2.circle(marked, (int(round(cx)), int(round(cy))), 8, (255, 0, 0), 2)

    output_path = ROOT_DIR / "foco_multiplos" / "alvo_fibra_definido.png"
    cv2.imwrite(str(output_path), marked)
    return output_path


def main() -> None:
    print("=== Definir alvo de alinhamento/fibra ===")
    print("Alinhe manualmente o laser no furo/posicao de acoplamento antes de salvar.")
    mode = input("Modo do laser (1=foco unico, 2=dupla reflexao) [2]: ").strip() or "2"
    focus_mode = set_focus_mode(mode)

    print("\nO que salvar?")
    print("  1 = posicao atual do laser como alvo da fibra")
    print("  2 = centro geometrico da camera")
    choice = input("Escolha [1]: ").strip() or "1"

    connect_camera()
    try:
        frames = []
        xs = []
        ys = []

        for idx in range(SAMPLES):
            frame = capture_frame(EXPOSURE_SECONDS, light=True)
            frames.append(frame)

            if choice == "2":
                x_target, y_target = centro_frame(frame)
            else:
                cm = centro_massa(frame)
                if cm is None:
                    raise RuntimeError(f"Nao encontrei o laser na captura {idx + 1}/{SAMPLES}.")
                x_target, y_target = float(cm[0]), float(cm[1])

            xs.append(float(x_target))
            ys.append(float(y_target))
            print(f"  amostra {idx + 1}/{SAMPLES}: x={x_target:.2f} y={y_target:.2f}")

        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        x_final = float(np.median(x_arr))
        y_final = float(np.median(y_arr))
        std_x = float(np.std(x_arr))
        std_y = float(np.std(y_arr))

        source = "camera_center" if choice == "2" else "fiber_hole_manual_alignment"
        target_path = salvar_alvo(
            x_final,
            y_final,
            source=source,
            frame_shape=frames[-1].shape,
            focus_mode=get_focus_mode(),
            samples=SAMPLES,
            std_x_px=std_x,
            std_y_px=std_y,
        )
        preview_path = _save_preview(frames[-1], x_final, y_final)

        print("\nAlvo salvo:")
        print(f"  x={x_final:.2f}px y={y_final:.2f}px | std=({std_x:.2f}, {std_y:.2f})px")
        print(f"  JSON: {display_path(target_path)}")
        print(f"  Preview: {display_path(preview_path)}")
    finally:
        disconnect_camera()


if __name__ == "__main__":
    main()
