import itertools
import time
import requests
import numpy as np
import cv2

from artifact_paths import display_path, matrix_candidates
# Mantém as importações originais de conexão
from PID_controll import (
    ensure_connected,
    ensure_unparked,
    ensure_not_tracking,
)

# IMPORTAÇÃO NOVA: Puxando a função de movimento 2D com threads que acabamos de criar
from mov_simultaneo import move_axes_pid_2d

# ==== Configurações Alpaca ====
BASE_URL = "http://127.0.0.1:11111/api/v1/camera/0"
CLIENT_ID = 1
_transaction_ids = itertools.count(1)
session = requests.Session()
IMAGE_READY_POLL_S = 0.005

lim_px = 2.0  # tolerância padrão em pixels

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


def connect_camera() -> None:
    print("Conectando à câmera...")
    call("PUT", "connected", data={"Connected": True})

def disconnect_camera() -> None:
    print("Desconectando da câmera...")
    call("PUT", "connected", data={"Connected": False})

def set_gain(gain: int) -> None:
    print(f"Ajustando ganho para {gain}...")
    call("PUT", "gain", data={"Gain": gain})

def start_exposure(duration_seconds: float, light: bool = True) -> None:
    print(f"Iniciando exposição: {duration_seconds:.6f}s | luz={light}")
    call("PUT", "startexposure", data={"Duration": duration_seconds, "Light": light})

def wait_until_image_ready(poll_interval: float = IMAGE_READY_POLL_S, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = bool(call("GET", "imageready"))
        if ready:
            return
        time.sleep(poll_interval)
    raise TimeoutError("Tempo limite esperando ImageReady = True")


def fetch_image_array() -> np.ndarray:
    payload = call("GET", "imagearray")
    array = np.asarray(payload)
    return array


def capture_frame(exposure_seconds: float, light: bool = True) -> np.ndarray:
    start_exposure(exposure_seconds, light=light)
    wait_until_image_ready()
    frame = fetch_image_array()
    frame = frame.astype(np.float32)
    
    min_val = float(frame.min())
    max_val = float(frame.max())

    if max_val < 200:
        norm = np.zeros_like(frame, dtype=np.uint8)
    else:
        pedestal = max(min_val, float(np.median(frame)) + 0.5 * float(np.std(frame)))
        norm = np.clip((frame - pedestal) / (max_val - pedestal + 1e-6), 0, 1)
        norm = (norm * 255).astype(np.uint8)

    norm = np.rot90(norm, 2)
    return norm


def centro_massa(frame: np.ndarray, threshold_percent: float = 0.5):
    if frame.ndim == 3:
        frame_gray = frame.mean(axis=2)
    else:
        frame_gray = frame.copy() 
        
    max_val = frame_gray.max()
    if max_val == 0:
        return None
        
    dynamic_threshold = max_val * threshold_percent
    frame_gray[frame_gray < dynamic_threshold] = 0

    total_intensidade = frame_gray.sum()
    if total_intensidade == 0:
        return None

    h, w = frame_gray.shape
    y = np.arange(h)
    x = np.arange(w)
    X, Y = np.meshgrid(x, y)

    x_cm = (X * frame_gray).sum() / total_intensidade
    y_cm = (Y * frame_gray).sum() / total_intensidade
    
    intensidade_cm = float(frame_gray[int(round(y_cm)), int(round(x_cm))])
    toca_borda = bool(np.any(frame_gray[0, :]) or np.any(frame_gray[-1, :]) or np.any(frame_gray[:, 0]) or np.any(frame_gray[:, -1]))
    
    return x_cm, y_cm, intensidade_cm, toca_borda

def centro_camera(frame: np.ndarray):
    h, w = frame.shape[:2]
    return (w -1)/ 2, (h -1)/ 2

def main() -> None:
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    connect_camera()

    try:
        set_gain(0)
        exposure_seconds = 32e-6

        print("Capturando um frame para cálculo do centro de massa...")
        frame = capture_frame(exposure_seconds, light=True)

        print(f"Frame capturado com shape = {frame.shape} e tipo = {frame.dtype}")

        cm = centro_massa(frame)
        if cm is None:
            print("Imagem sem sinal (toda preta), não há centro de massa. O laser pode estar apagado ou fraco demais (intensidade = 0).")
            return
            
        x_cm, y_cm, aux_i, toca_borda = cm
        
        if aux_i == 0:
            print(f"Atenção: A intensidade no ponto central geométrico é baixa ({aux_i:.2f}/255), mas a imagem conteve sinal claro suficiente. O laser parece estar na borda ou fragmentado.")
            
        if toca_borda:
            print("AVISO: O laser está encostando na beirada da imagem. Portanto, o centro de massa é apenas uma aproximação e está deslocado do real.")

        print("--- LASER DETECTADO ---")
        print(f"Centro de massa (x, y) em pixels: ({x_cm:.2f}, {y_cm:.2f})")
        print(f"Intensidade no CM: {aux_i:.2f}/255")

        frame_marked = frame.copy()
        if frame_marked.ndim == 2:
            frame_marked = cv2.cvtColor(frame_marked, cv2.COLOR_GRAY2BGR)

        cx, cy = centro_camera(frame)
        print(f"Centro da câmera (x, y) em pixels: ({cx:.2f}, {cy:.2f})")

        dx = x_cm - cx
        dy = y_cm - cy
        print(f"Deslocamento (cm - centro) em pixels: (dx = {dx:+.3f}, dy = {dy:+.3f})")

        frame_inicial = frame.copy()
        if frame_inicial.ndim == 2:
            frame_inicial = cv2.cvtColor(frame_inicial, cv2.COLOR_GRAY2BGR)

        cm_ix, cm_iy = int(round(x_cm)), int(round(y_cm))
        c_ix, c_iy = int(round(cx)), int(round(cy))

        cv2.circle(frame_inicial, (cm_ix, cm_iy), 8, (0, 255, 0), -1)
        cv2.circle(frame_inicial, (c_ix, c_iy), 8, (255, 0, 0), -1)

        output_path_inicial = "frame_inicial_cm_centro.png"
        cv2.imwrite(output_path_inicial, frame_inicial)
        print(f"Frame inicial salvo em: {output_path_inicial}")

        candidates = matrix_candidates("A_inv_coarse.npy", "calibracao_A_inv.npy")
        try:
            for candidate in candidates:
                if not candidate.exists():
                    continue
                A_inv = np.load(candidate)
                if A_inv.shape != (2, 2):
                    raise ValueError("Matriz A_inv com shape inválido.")
                print(f"Matriz carregada: {display_path(candidate)}")
                break
            else:
                raise FileNotFoundError(
                    f"Testei: {', '.join(str(path) for path in candidates)}"
                )
        except Exception as e:
            print(f"\nNão foi possível carregar a matriz de centralização: {e}")
            print("Execute primeiro a calibração 2D para gerar este arquivo.")
            return

        usar_mount = True
        move = input("\nDeseja centralizar o laser? (s/n): ").strip().lower()
        if move != 's':
            print("Centralização automática cancelada pelo usuário.")
            return
            
        x_cm0, y_cm0 = x_cm, y_cm

        while True:
            dx = x_cm - cx
            dy = y_cm - cy
            print(f"\nDeslocamento atual (cm - centro): dx = {dx:+.3f}, dy = {dy:+.3f} px")

            if abs(dx) <= lim_px and abs(dy) <= lim_px:
                print("Dentro da tolerância em pixels. Encerrando correções.")
                break

            vec_px = np.array([-dx, -dy])
            correcao = A_inv @ vec_px
            dAz_deg = float(correcao[0])
            dAlt_deg = float(correcao[1])

            print("--- Correção de apontamento usando A^{-1} ---")
            print(f"Movimento alvo: ΔAz ≈ {dAz_deg:+.6f} deg, ΔAlt ≈ {dAlt_deg:+.6f} deg")

            # MUDANÇA PRINCIPAL AQUI: Dispara os dois eixos ao mesmo tempo usando threads
            move_axes_pid_2d(usar_mount, dAz_deg, dAlt_deg)

            print("Capturando novo frame para próxima iteração...")
            frame = capture_frame(exposure_seconds, light=True)
            cm = centro_massa(frame)
            if cm is None:
                print("Imagem sem sinal após correção; interrompendo.")
                break
            x_cm, y_cm, aux_i, toca_borda = cm

        print(f"\nCM final: ({x_cm:.2f}, {y_cm:.2f})")
        dx_final = x_cm - cx
        dy_final = y_cm - cy
        print(f"Deslocamento final (cm - centro): dx = {dx_final:+.3f}, dy = {dy_final:+.3f} px")

        frame_final = frame.copy()
        if frame_final.ndim == 2:
            frame_final = cv2.cvtColor(frame_final, cv2.COLOR_GRAY2BGR)

        cm0_ix, cm0_iy = int(round(x_cm0)), int(round(y_cm0))
        cmf_ix, cmf_iy = int(round(x_cm)), int(round(y_cm))

        cv2.circle(frame_final, (c_ix, c_iy), 8, (255, 0, 0), -1)
        cv2.circle(frame_final, (cm0_ix, cm0_iy), 8, (0, 255, 0), -1)
        cv2.circle(frame_final, (cmf_ix, cmf_iy), 8, (0, 0, 255), -1)
        cv2.line(frame_final, (cm0_ix, cm0_iy), (cmf_ix, cmf_iy), (0, 255, 255), 2)

        output_path_final = "frame_final_cm_trajetoria.png"
        cv2.imwrite(output_path_final, frame_final)
        print(f"Frame final salvo em: {output_path_final}")
        
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário (Ctrl+C). Encerrando de forma limpa...")
    finally:
        disconnect_camera()

if __name__ == "__main__":
    main()
