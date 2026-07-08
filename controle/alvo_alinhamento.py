import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from artifact_paths import display_path, json_candidates, json_output_path


TARGET_FILENAME = "alvo_alinhamento_camera.json"


@dataclass
class AlvoAlinhamento:
    x_px: float
    y_px: float
    source: str
    path: str | None = None


def centro_frame(frame: np.ndarray) -> tuple[float, float]:
    h, w = frame.shape[:2]
    return (w - 1) / 2, (h - 1) / 2


def carregar_alvo_salvo() -> AlvoAlinhamento | None:
    for path in json_candidates(TARGET_FILENAME):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        return AlvoAlinhamento(
            x_px=float(data["target_x_px"]),
            y_px=float(data["target_y_px"]),
            source=str(data.get("source", "saved_target")),
            path=display_path(path),
        )
    return None


def salvar_alvo(
    x_px: float,
    y_px: float,
    *,
    source: str,
    frame_shape: tuple[int, ...] | None = None,
    focus_mode: str | None = None,
    samples: int | None = None,
    std_x_px: float | None = None,
    std_y_px: float | None = None,
) -> Path:
    payload = {
        "timestamp_epoch": time.time(),
        "target_x_px": float(x_px),
        "target_y_px": float(y_px),
        "source": source,
        "frame_shape": None if frame_shape is None else list(frame_shape),
        "focus_mode": focus_mode,
        "samples": samples,
        "std_x_px": std_x_px,
        "std_y_px": std_y_px,
    }
    path = json_output_path(TARGET_FILENAME)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def escolher_alvo_para_frame(frame: np.ndarray, prompt: str = "Alvo de alinhamento") -> AlvoAlinhamento:
    salvo = carregar_alvo_salvo()
    cx, cy = centro_frame(frame)

    print(f"\n{prompt}:")
    if salvo is not None:
        print(
            f"  1 = usar alvo salvo/posicao da fibra "
            f"({salvo.x_px:.2f}, {salvo.y_px:.2f}) em {salvo.path}"
        )
    else:
        print("  1 = usar alvo salvo/posicao da fibra (nao encontrado)")
    print(f"  2 = usar centro da camera ({cx:.2f}, {cy:.2f})")

    choice = input("Escolha [1]: ").strip() or "1"
    if choice == "1":
        if salvo is not None:
            return salvo
        print("Alvo salvo nao encontrado; usando centro da camera.")

    return AlvoAlinhamento(x_px=float(cx), y_px=float(cy), source="camera_center", path=None)


def escolher_posicao_inicial_ou_centro(
    frame: np.ndarray,
    x_inicial: float,
    y_inicial: float,
    prompt: str = "Referencia de alinhamento",
    salvar_novo_alvo: bool = True,
) -> AlvoAlinhamento:
    salvo = carregar_alvo_salvo()
    cx, cy = centro_frame(frame)

    print(f"\n{prompt}:")
    if salvo is not None:
        print(f"  1 = usar alvo salvo ({salvo.x_px:.2f}, {salvo.y_px:.2f}) em {salvo.path}")
    else:
        print("  1 = usar alvo salvo (nao encontrado)")
    print(f"  2 = definir novo alvo pela posicao inicial ({x_inicial:.2f}, {y_inicial:.2f})")
    print(f"  3 = usar centro da camera ({cx:.2f}, {cy:.2f})")

    choice = input("Escolha [1]: ").strip() or "1"
    if choice == "1" and salvo is not None:
        return salvo
    if choice == "1":
        print("Alvo salvo nao encontrado; definindo novo alvo pela posicao inicial.")
        choice = "2"

    if choice == "3":
        return AlvoAlinhamento(x_px=float(cx), y_px=float(cy), source="camera_center", path=None)

    path = None
    if salvar_novo_alvo:
        path_obj = salvar_alvo(
            x_inicial,
            y_inicial,
            source="initial_laser_position",
            frame_shape=frame.shape,
            samples=1,
            std_x_px=0.0,
            std_y_px=0.0,
        )
        path = display_path(path_obj)
        print(f"Novo alvo salvo em: {path}")

    return AlvoAlinhamento(
        x_px=float(x_inicial),
        y_px=float(y_inicial),
        source="initial_laser_position",
        path=path,
    )


def escolher_alvo_para_sensor(
    sensor_w: int,
    sensor_h: int,
    prompt: str = "Alvo de alinhamento",
) -> AlvoAlinhamento:
    salvo = carregar_alvo_salvo()
    cx = (sensor_w - 1) / 2
    cy = (sensor_h - 1) / 2

    print(f"\n{prompt}:")
    if salvo is not None:
        print(
            f"  1 = usar alvo salvo/posicao da fibra "
            f"({salvo.x_px:.2f}, {salvo.y_px:.2f}) em {salvo.path}"
        )
    else:
        print("  1 = usar alvo salvo/posicao da fibra (nao encontrado)")
    print(f"  2 = usar centro da camera ({cx:.2f}, {cy:.2f})")

    choice = input("Escolha [1]: ").strip() or "1"
    if choice == "1":
        if salvo is not None:
            return salvo
        print("Alvo salvo nao encontrado; usando centro da camera.")

    return AlvoAlinhamento(x_px=float(cx), y_px=float(cy), source="camera_center", path=None)


def roi_incluindo_alvo(
    sensor_w: int,
    sensor_h: int,
    roi_w: int,
    roi_h: int,
    target_x: float,
    target_y: float,
) -> tuple[int, int, float, float]:
    if roi_w > sensor_w or roi_h > sensor_h:
        raise ValueError("ROI maior que o sensor.")

    start_x = int(round(target_x - (roi_w / 2)))
    start_y = int(round(target_y - (roi_h / 2)))
    start_x = int(np.clip(start_x, 0, sensor_w - roi_w))
    start_y = int(np.clip(start_y, 0, sensor_h - roi_h))

    target_x_local = float(target_x - start_x)
    target_y_local = float(target_y - start_y)
    return start_x, start_y, target_x_local, target_y_local
