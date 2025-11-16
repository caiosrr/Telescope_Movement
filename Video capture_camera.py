"""Controle de câmera ASCOM Alpaca: ajuste de exposição/ganho e gravação de vídeo."""

from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
import time

# Substitua pelo endereço IP mostrado no ASCOM Remote Server / Alpaca.
ALPACA_SERVER_ADDRESS = "http://127.0.0.1:11111/"
CAMERA_NUMBER = 0
CLIENT_ID = 1234

# Parâmetros do vídeo
OUTPUT_VIDEO = Path("capturas") / "captura.mp4"
FRAME_COUNT = 150
FPS = 30
EXPOSURE_SECONDS = 0.000272
GAIN = 270                

base_url = f"{ALPACA_SERVER_ADDRESS}api/v1/camera/{CAMERA_NUMBER}"
_tx_counter = count(1)


def _next_tx_id() -> int:
    """Retorna um novo ClientTransactionID sequencial."""
    return next(_tx_counter)


def _as_form_values(data: Dict[str, Any]) -> Dict[str, str]:
    """Converte valores para strings no formato esperado pelo Alpaca."""
    converted: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            converted[key] = str(value).lower()
        else:
            converted[key] = str(value)
    return converted


def alpaca_put(command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Executa um PUT no endpoint especificado e retorna o JSON decodificado."""
    if payload is None:
        payload = {}
    form = _as_form_values({**payload, "ClientID": CLIENT_ID, "ClientTransactionID": _next_tx_id()})
    response = requests.put(
        f"{base_url}/{command}",
        data=form,
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("ErrorNumber", 0) != 0:
        raise RuntimeError(f"Erro Alpaca ({command}): {data.get('ErrorMessage')} [Erro {data.get('ErrorNumber')}]")
    return data


def alpaca_get(command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Executa um GET no endpoint especificado e retorna o JSON decodificado."""
    if params is None:
        params = {}
    merged = _as_form_values({**params, "ClientID": CLIENT_ID, "ClientTransactionID": _next_tx_id()})
    response = requests.get(
        f"{base_url}/{command}",
        params=merged,
        headers={"Accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("ErrorNumber", 0) != 0:
        raise RuntimeError(f"Erro Alpaca ({command}): {data.get('ErrorMessage')} [Erro {data.get('ErrorNumber')}]")
    return data


def get_value(command: str) -> Any:
    """Atalho para retornar apenas a chave Value do JSON."""
    return alpaca_get(command).get("Value")


def connect_camera() -> None:
    print("Conectando à câmera...")
    alpaca_put("connected", {"Connected": True})


def disconnect_camera() -> None:
    print("Desconectando da câmera...")
    try:
        alpaca_put("connected", {"Connected": False})
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Aviso: não foi possível desconectar limpamente ({exc}).")


def set_gain(gain_value: int) -> None:
    print(f"Ajustando ganho para {gain_value}...")
    alpaca_put("gain", {"Gain": gain_value})


def start_exposure(duration_seconds: float, light: bool = True) -> None:
    print(f"Iniciando exposição: {duration_seconds:.3f}s | luz={light}")
    alpaca_put("startexposure", {"Duration": duration_seconds, "Light": light})


def wait_until_image_ready(poll_interval: float = 0.1, timeout: float = 30.0) -> None:
    """Bloqueia até que ImageReady seja verdadeiro ou estoure o timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = bool(get_value("imageready"))
        if ready:
            return
        time.sleep(poll_interval)
    raise TimeoutError("Tempo limite esperando ImageReady = True")


def fetch_image_array() -> np.ndarray:
    payload = alpaca_get("imagearray")
    raw_value = payload.get("Value")
    array = np.asarray(raw_value)
    return array


def capture_frame(exposure_seconds: float, light: bool = True) -> np.ndarray:
    start_exposure(exposure_seconds, light)
    wait_until_image_ready()
    frame = fetch_image_array()
    frame = frame.astype(np.float32)
    min_val = float(frame.min())
    max_val = float(frame.max())
    if np.isclose(max_val, min_val):
        norm = np.zeros_like(frame, dtype=np.uint8)
    else:
        norm = (frame - min_val) / (max_val - min_val)
        norm = (norm * 255).astype(np.uint8)
    return norm


def record_video(
    frame_count: int,
    exposure_seconds: float,
    gain_value: Optional[int],
    fps: int,
    output_path: Path,
    light: bool = True,
) -> Path:
    if gain_value is not None:
        set_gain(gain_value)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = get_value("cameraxsize")
    height = get_value("cameraysize")
    if width is None or height is None:
        raise RuntimeError("Não foi possível obter dimensões da câmera via Alpaca.")

    frame_size = (int(width), int(height))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size, isColor=False)

    if not video_writer.isOpened():
        raise RuntimeError("Não foi possível abrir o VideoWriter do OpenCV. Verifique codecs instalados.")

    try:
        for index in range(frame_count):
            print(f"Capturando frame {index + 1}/{frame_count}...")
            frame = capture_frame(exposure_seconds, light=light)
            frame_resized = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)
            video_writer.write(frame_resized)
    finally:
        video_writer.release()

    return output_path


def main() -> None:
    print(f"Tentando conectar ao servidor Alpaca em: {base_url}")

    try:
        camera_name = get_value("name")
        print(f"Câmera detectada: {camera_name}")
        connect_camera()

        video_path = record_video(
            frame_count=FRAME_COUNT,
            exposure_seconds=EXPOSURE_SECONDS,
            gain_value=GAIN,
            fps=FPS,
            output_path=OUTPUT_VIDEO,
        )

        print("\n--- Captura concluída ---")
        print(f"Arquivo salvo em: {video_path.resolve()}")
        print("Você pode abrir o arquivo diretamente do disco ou processá-lo no seu programa.")

    except requests.exceptions.ConnectionError:
        print("ERRO DE CONEXÃO: não foi possível alcançar o servidor Alpaca.")
        print("Verifique se o endereço IP está correto e se o servidor está rodando.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Erro ao controlar a câmera: {exc}")
    finally:
        disconnect_camera()


if __name__ == "__main__":
    main()