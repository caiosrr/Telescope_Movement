import itertools
import time
import requests
import numpy as np
import cv2

from Movimento.PID_controll import (
    move_axis_pid,
    ensure_connected,
    ensure_unparked,
    ensure_not_tracking,
)

# ==== Configurações Alpaca ====
BASE_URL = "http://127.0.0.1:11111/api/v1/camera/0"
CLIENT_ID = 1
_transaction_ids = itertools.count(1)

lim_px = 2.0  # tolerância padrão em pixels

def call(method: str, command: str, timeout: float = 5.0, **extra_args):
    params = {
        "ClientID": CLIENT_ID,
        "ClientTransactionID": next(_transaction_ids),
    }
    params.update(extra_args.pop("params", {}))
    resp = requests.request(
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

def wait_until_image_ready(poll_interval: float = 0.05, timeout: float = 5.0) -> None:
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
    
    # Após analisar os logs da sua câmera, sabemos que o ruído escuro puro 
    # de uma imagem sem laser tem um max na casa dos ~170 de ADU bruto.
    # Um laser de verdade estouraria os valores para a casa dos milhares.
    min_val = float(frame.min())
    max_val = float(frame.max())

    # Um corte ABSOLUTO da câmera. Reduzimos o valor para 150 para pegar feixes mais fracos nas bordas.
    if max_val < 200:
        norm = np.zeros_like(frame, dtype=np.uint8)
    else:
        # Pega a mediana do ruído de fundo
        pedestal = max(min_val, float(np.median(frame)) + 0.5 * float(np.std(frame)))
        norm = np.clip((frame - pedestal) / (max_val - pedestal + 1e-6), 0, 1)
        norm = (norm * 255).astype(np.uint8)

    # Corrige orientação: frame original está rotacionado 180° em relação à visão real.
    # Rotacionar aqui garante que todos os cálculos (CM, calibração, etc.)
    # e imagens salvas usem o mesmo referencial da câmera.
    norm = np.rot90(norm, 2)

    return norm


def centro_massa(frame: np.ndarray, threshold_percent: float = 0.5):
    # se for RGB, converte para cinza
    if frame.ndim == 3:
        frame_gray = frame.mean(axis=2)
    else:
        frame_gray = frame.copy() # Cria uma cópia para não alterar a imagem original
        
    max_val = frame_gray.max()
    if max_val == 0:
        return None
        
    # Calcula limiar dinâmico baseado no pico máximo de luz (Mata o rastro)
    dynamic_threshold = max_val * threshold_percent
        
    # Zera todos os pixels que tiverem valor abaixo do limiar
    frame_gray[frame_gray < dynamic_threshold] = 0

    total_intensidade = frame_gray.sum()
    
    # Se depois de remover o ruído a intensidade for 0, não tem laser
    if total_intensidade == 0:
        return None

    h, w = frame_gray.shape
    y = np.arange(h)
    x = np.arange(w)
    X, Y = np.meshgrid(x, y)

    x_cm = (X * frame_gray).sum() / total_intensidade
    y_cm = (Y * frame_gray).sum() / total_intensidade
    
    # Intensidade do pixel correspondente ao centro de massa
    intensidade_cm = float(frame_gray[int(round(y_cm)), int(round(x_cm))])
    
    # Verifica se algum pixel iluminado do laser encostou em uma das 4 extremidades do frame (top, bottom, left, right)
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
        # ajuste aqui exposição e ganho como quiser
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
        
        # A intensidade exata no (x_cm, y_cm) pode ser baixa se o feixe estiver na borda ou for formato de anel.
        # Descartaremos o ruído antes (no capture_frame com limite de brilho 150). 
        # Portanto, se cm foi calculado, aceitamos a detecção.
        if aux_i == 0:
            print(f"Atenção: A intensidade no ponto central geométrico é baixa ({aux_i:.2f}/255), mas a imagem conteve sinal claro suficiente. O laser parece estar na borda ou fragmentado.")
            
        if toca_borda:
            print("AVISO: O laser está encostando na beirada da imagem. Portanto, o centro de massa é apenas uma aproximação e está deslocado do real.")

        print("--- LASER DETECTADO ---")
        print(f"Centro de massa (x, y) em pixels: ({x_cm:.2f}, {y_cm:.2f})")
        print(f"Intensidade no CM: {aux_i:.2f}/255")

    # Desenhar a marcação e salvar a imagem
        frame_marked = frame.copy()
        # converter para BGR se for 2D (preto e branco)
        if frame_marked.ndim == 2:
            frame_marked = cv2.cvtColor(frame_marked, cv2.COLOR_GRAY2BGR)

        cx, cy = centro_camera(frame)
        print(f"Centro da câmera (x, y) em pixels: ({cx:.2f}, {cy:.2f})")

        # diferença (com sinal): cm - centro_da_câmera
        dx = x_cm - cx
        dy = y_cm - cy
        print(f"Deslocamento (cm - centro) em pixels: (dx = {dx:+.3f}, dy = {dy:+.3f})")

        # --- desenhar cm e centro da câmera no frame inicial e salvar ---
        frame_inicial = frame.copy()
        if frame_inicial.ndim == 2:
            frame_inicial = cv2.cvtColor(frame_inicial, cv2.COLOR_GRAY2BGR)

        # coordenadas inteiras para desenho
        cm_ix, cm_iy = int(round(x_cm)), int(round(y_cm))
        c_ix, c_iy = int(round(cx)), int(round(cy))

        # cm inicial em verde (disco pequeno)
        cv2.circle(frame_inicial, (cm_ix, cm_iy), 8, (0, 255, 0), -1)
        # centro da câmera em azul (disco pequeno)
        cv2.circle(frame_inicial, (c_ix, c_iy), 8, (255, 0, 0), -1)

        output_path_inicial = "frame_inicial_cm_centro.png"
        cv2.imwrite(output_path_inicial, frame_inicial)
        print(f"Frame inicial salvo em: {output_path_inicial}")

        # --- Correção automática iterativa usando matriz A^{-1} salva na calibração ---
        try:
            A_inv = np.load("calibracao_A_inv.npy")
            if A_inv.shape != (2, 2):
                raise ValueError("Matriz A_inv com shape inválido.")
        except Exception as e:
            print(f"\nNão foi possível carregar 'calibracao_A_inv.npy': {e}")
            print("Execute primeiro a calibração 2D para gerar este arquivo.")
            return

        usar_mount = True
        move = input("\nDeseja centralizar o laser? (s/n): ").strip().lower()
        if move != 's':
            print("Centralização automática cancelada pelo usuário.")
            return
        # guardar CM inicial para desenhar trajetória depois
        x_cm0, y_cm0 = x_cm, y_cm

        while True:
            # dx, dy atuais (cm - centro)
            dx = x_cm - cx
            dy = y_cm - cy
            print(f"\nDeslocamento atual (cm - centro): dx = {dx:+.3f}, dy = {dy:+.3f} px")

            if abs(dx) <= lim_px and abs(dy) <= lim_px:
                print("Dentro da tolerância em pixels. Encerrando correções.")
                break

            # deslocamento desejado em pixels: levar CM até o centro
            vec_px = np.array([-dx, -dy])
            dAz_deg, dAlt_deg = A_inv @ vec_px
            dAlt_deg = -dAlt_deg  # corrigir sinal para telescópio invertido

            print("--- Correção de apontamento usando A^{-1} ---")
            print(f"Movimento alvo: ΔAz ≈ {dAz_deg:+.6f} deg, ΔAlt ≈ {dAlt_deg:+.6f} deg")

            # Movimento em Az (eixo 0) e Alt (eixo 1) usando PID
            move_axis_pid(usar_mount, 0, dAz_deg)
            move_axis_pid(usar_mount, 1, dAlt_deg)

            # Captura novo frame para próxima iteração
            print("Capturando novo frame para próxima iteração...")
            frame = capture_frame(exposure_seconds, light=True)
            cm = centro_massa(frame)
            if cm is None:
                print("Imagem sem sinal após correção; interrompendo.")
                break
            x_cm, y_cm, aux_i, toca_borda = cm

        # Depois do loop, x_cm, y_cm são o CM final
        print(f"\nCM final: ({x_cm:.2f}, {y_cm:.2f})")
        dx_final = x_cm - cx
        dy_final = y_cm - cy
        print(f"Deslocamento final (cm - centro): dx = {dx_final:+.3f}, dy = {dy_final:+.3f} px")

        # --- desenhar frame final com trajetória do CM ---
        frame_final = frame.copy()
        if frame_final.ndim == 2:
            frame_final = cv2.cvtColor(frame_final, cv2.COLOR_GRAY2BGR)

        cm0_ix, cm0_iy = int(round(x_cm0)), int(round(y_cm0))
        cmf_ix, cmf_iy = int(round(x_cm)), int(round(y_cm))

        # centro da câmera em azul
        cv2.circle(frame_final, (c_ix, c_iy), 8, (255, 0, 0), -1)
        # CM inicial em verde
        cv2.circle(frame_final, (cm0_ix, cm0_iy), 8, (0, 255, 0), -1)
        # CM final em vermelho
        cv2.circle(frame_final, (cmf_ix, cmf_iy), 8, (0, 0, 255), -1)
        # segmento do CM inicial ao final
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