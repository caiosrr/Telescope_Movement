"""MoveAxis contínuo com desaceleração exponencial e correção de overshoot suave."""

import itertools
import time
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
CLIENT_ID = 1
_transaction_ids = itertools.count(1)

# ==== Parâmetros de controle ====
TOLERANCIA_GRAUS = 0.001
VEL_MIN_LIMITE = 0.001042  # menor velocidade efetiva do mount
VEL_MAX_LIMITE = 6.0
MAX_TEMPO_MOV = 450
MAX_CORRECOES = 10  # máximo de inversões de direção
eps = 1e-12         # pequeno valor para evitar log(0)

def call(method: str, command: str, timeout: float = 5.0, **extra_args): 
    params = {"ClientID": CLIENT_ID, "ClientTransactionID": next(_transaction_ids)} # Gera IDs de cliente e transação exigidos pelo protocolo Alpaca.
    params.update(extra_args.pop("params", {})) # Mescla parâmetros obrigatórios com quaisquer parâmetros adicionais.
    resp = requests.request(
        method, f"{BASE_URL}/{command}", params=params, timeout=timeout, **extra_args
    ) # Envia a requisição HTTP usando 'requests'.
    resp.raise_for_status() # Verifica erros HTTP.
    payload = resp.json()
    if payload.get("ErrorNumber", 0):
        raise RuntimeError(f"{command}: {payload.get('ErrorMessage')}") # Verifica erros Alpaca.
    return payload.get("Value")
"""
Executa uma chamada genérica à API Alpaca.

Parâmetros:
    method (str): Método HTTP a usar ("GET", "PUT", ...)
    command (str): Comando do endpoint Alpaca (ex: "moveaxis", "connected")
    timeout (float): Tempo máximo de espera pela resposta (s)
    **request_kwargs: Argumentos opcionais passados ao requests.request()

Retorna:
    O valor do campo "Value" retornado pelo servidor Alpaca.
"""


def ensure_connected(): # Garante que o mount esteja conectado.
    if not call("GET", "connected"):
        call("PUT", "connected", data={"Connected": True})


def ensure_unparked(): # Garante que o mount esteja não estacionado.
    try:
        if call("GET", "atpark"):
            if call("GET", "canunpark"):
                call("PUT", "unpark", timeout=10)
    except Exception:
        pass


def ensure_not_tracking(): # Desativa o tracking se estiver ativo.
    try:
        if str(call("GET", "tracking")).lower() in {"true", "1"}:
            call("PUT", "tracking", data={"Tracking": False})
    except Exception:
        pass


def read_altaz(): # Lê a posição atual em azimute e altitude.
    az = float(call("GET", "azimuth"))
    alt = float(call("GET", "altitude"))
    return np.array([az, alt])


def move_axis(axis: int, rate_deg_per_s: float): # Envia comando MoveAxis com eixo e velocidade. 
    # ZWO: sinal invertido no Az
    # if axis == 0:
        # rate_deg_per_s = -rate_deg_per_s
    call("PUT", "moveaxis", data={"Axis": axis, "Rate": float(rate_deg_per_s)})

def calc_errors(alvo: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Calcula o erro considerando a circularidade no azimute."""
    erro = np.zeros(2)
    # erro com circularidade no azimute
    diff = (alvo[0] - pos[0]) % 360
    if diff > 180:
        erro[0] = diff - 360
    else:
        erro[0] = diff
    erro[1] = alvo[1] - pos[1]
    return erro

# ==== Cálculo de velocidade ====
def calc_vel_exponencial(
    k_exp: np.ndarray,
    error_abs: np.ndarray,
    error_init: np.ndarray,
    last_vel: np.ndarray = None
):  
    if last_vel is None:
        last_vel = np.full(2, VEL_MAX_LIMITE)

    vel_max = np.full(2, VEL_MAX_LIMITE)
    fator = np.ones(2)

    error_init_abs = np.abs(error_init)
    # --- Máscara de erro zero (deve ser tratada primeiro)
    mask0 = error_abs < TOLERANCIA_GRAUS

    # --- Máscaras para cada faixa ---
    mask_far   = error_abs > 12.0
    mask_mid1  = (error_abs > 8.0) & (error_abs <= 12.0)
    mask_mid2  = (error_abs > 1.0) & (error_abs <= 8.0)
    mask_mid3  = (error_abs > 0.1) & (error_abs <= 1.0)
    mask_mid4  = (error_abs > 0.01) & (error_abs <= 0.1)
    mask_close = (error_abs > TOLERANCIA_GRAUS) & (error_abs <= 0.01)

    # --- Atribuições por faixa ---
    vel_max[mask_mid1] = np.minimum(6.0, last_vel[mask_mid1])
    vel_max[mask_mid2] = np.minimum(1.0, last_vel[mask_mid2])
    vel_max[mask_mid3] = np.minimum(0.1, last_vel[mask_mid3])
    vel_max[mask_mid4] = np.minimum(0.01, last_vel[mask_mid4])

    # -----------------
    # CASO FAR (erro > 12)
    # -----------------
    idx_far = np.where(mask_far)[0]

    if len(idx_far) > 0:
        if len(idx_far) == 2:
            vel = np.full(2, VEL_MAX_LIMITE)
            return vel, k_exp

        i = idx_far[0]
        l = 1 - i

        vel_max[i] = VEL_MAX_LIMITE

        if not mask0[l]:
            k_exp[l] += 0.0001
            fator[l] = np.exp(-k_exp[l] * (1 - error_abs[l] / error_init_abs[l]))
            vel_max[l] *= fator[l]

        vel = np.maximum(vel_max, VEL_MIN_LIMITE)
        vel[mask0] = 0.0
        return vel, k_exp

    # -----------------
    # CASO CLOSE (0 < erro <= 0.01)
    # -----------------
    idx_close = np.where(mask_close)[0]

    if len(idx_close) > 0:
        if len(idx_close) == 2:
            vel = np.full(2, VEL_MIN_LIMITE)
            return vel, k_exp

        i = idx_close[0]
        l = 1 - i

        vel_max[i] = VEL_MIN_LIMITE

        if not mask0[l]:
            k_exp[l] += 1e-5
            fator[l] = np.exp(-k_exp[l] * (1 - error_abs[l] / error_init_abs[l]))
            vel_max[l] *= fator[l]

        vel = np.maximum(vel_max, VEL_MIN_LIMITE)
        vel[mask0] = 0.0
        return vel, k_exp

    # -----------------
    # CASO NORMAL
    # -----------------
    if not np.all(mask0):
        if np.any(mask0):
            idx_mask0 = np.where(~mask0)[0] # índices onde error_abs != 0
            k_exp[idx_mask0] += 1e-5
            fator[idx_mask0] = np.exp(-k_exp[idx_mask0] \
                                    * (1 - error_abs[idx_mask0] / error_init_abs[idx_mask0]))
            vel_max[idx_mask0] *= fator[idx_mask0]
        else:
            k_exp += 1e-5
            fator = np.exp(-k_exp * (1 - error_abs / error_init_abs))
            vel_max *= fator

    vel = np.maximum(vel_max, VEL_MIN_LIMITE)
    vel[mask0] = 0.0
    return vel, k_exp


# ==== Movimento inteligente ====
def move_axis_smart(delta: np.ndarray):
    """Movimento contínuo com desaceleração exponencial e correção de overshoot."""
    axis = np.array([0, 1])
    pos0 = read_altaz()  # [az0, alt0]
    alvo = np.array([pos0[0] + delta[0], pos0[1] + delta[1]])
        
    if alvo[1] > 90:
        print(f"⚠️ Movimento ultrapassa o limite em altitude. Ajustado para 90°.")
        alvo[1] = 90
    elif alvo[1] < -90:
        print(f"⚠️ Movimento ultrapassa o limite em altitude. Ajustado para -90°.")
        alvo[1] = -90

    error = calc_errors(alvo, pos0)
    alvo[0] = (pos0[0] + error[0]) % 360  # ajuste final do azimute
    error_init = error.copy()
    error_test = error.copy()
    error_abs = np.abs(error)

    k_exp = np.zeros(2)

    sentido = np.zeros(2)  # [sentido_az, sentido_alt]
    sentido = np.array([1 if e >= 0 else -1 for e in error])

    vel, k_exp = calc_vel_exponencial(k_exp, error_abs, error_init, None)
    cmd = sentido * vel
    


    print(f"\nMovimento inteligente (exponencial):")
    print(f"   Alvo = ({alvo[0]:.2f}, {alvo[1]:.2f})° | VelMáx = [{cmd[0]:.2f}, {cmd[1]:.2f}]°/s")

    t0 = time.time()

    move_axis(axis[0], cmd[0])
    move_axis(axis[1], cmd[1])

    inversoes = np.zeros(2, dtype=int)
    last_vel = vel.copy()
    cooldown_until = np.zeros(2)
    try:
        while True:
            pos = read_altaz()
            move_axis(0, 0.0)
            move_axis(1, 0.0)
            error = calc_errors(alvo, pos)
            error_abs = np.abs(error)

            tempo_decorrido = time.time() - t0

            if tempo_decorrido > MAX_TEMPO_MOV:
                print("⚠️ Tempo limite atingido.")
                break
            if np.all(error_abs < TOLERANCIA_GRAUS):
                print(" " * 80, end="\r")
                print(f"✅ Dentro da tolerância ({TOLERANCIA_GRAUS}°).")
                break

            idx_flip = np.where(error * error_test < 0)[0]
            if len(idx_flip) > 0:
                inversoes[idx_flip] += 1

                if np.any(inversoes > MAX_CORRECOES):
                    print("⚠️ Número máximo de correções atingido, parando.")
                    break

                print(f"\n↩️ Overshoot detectado (#{inversoes}). "
                    "Invertendo sentido e reduzindo velocidade...")

                if len(idx_flip) == 2:
                    sentido *= -1
                    k_exp += 5e-5
                    cooldown_until[:] = time.time() + 0.7
                    error_test = error.copy()

                else:
                    j = idx_flip[0]
                    sentido[j] *= -1
                    k_exp[j] += 5e-5
                    cooldown_until[j] = time.time() + 0.7
                    error_test = error.copy()

            vel, k_exp = calc_vel_exponencial(k_exp, error_abs, error_init, last_vel)
            last_vel = vel.copy()

            mask_drop = np.floor(np.log10(vel + eps)) < np.floor(np.log10(last_vel + eps))
            idx_drop = np.where(mask_drop)[0]
            if len(idx_drop) > 0:
                for j in idx_drop:
                    move_axis(axis[j], 0.0)
                cooldown_until[idx_drop] = time.time() + 0.3

            cmd = sentido * vel
            cmd = sentido * vel

            now = time.time()
            cmd_real = np.where(now < cooldown_until, 0.0, cmd)

            move_axis(0, cmd_real[0])
            move_axis(1, cmd_real[1])

            print(
                f"  Pos = ({pos[0]:.4f}, {pos[1]:.4f})°| Erro = ({error[0]:+.4f}, {error[1]:+.4f})°| "
                f"Vel = ({cmd[0]:.4f}, {cmd[1]:.4f})°/s  ", end="\r"
            )

            time.sleep(0.1)

    finally:
        move_axis(0, 0.0)
        move_axis(1, 0.0)

        posf = read_altaz()
        print(f"\nPos final: ({posf[0]:.2f}, {posf[1]:.2f})°; \
            Erro = ({posf[0] - alvo[0]:+.4f}, {posf[1] - alvo[1]:+.4f})°, \
            Tempo = {tempo_decorrido:.2f}s")

# ==== Programa principal ====
def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    print("Movimento com desaceleração exponencial \n")

    while True:
        try:
            az, alt = read_altaz()
            print(f"Current position: Az = {az:.2f}°, Alt = {alt:.2f}°")
            entrance = input("Desired displacement (Az Alt): ").split()
            if len(entrance) != 2:
                raise ValueError("Digite exatamente dois valores (ex: 10 0.5).")
            delta = np.array([float(entrance[0]), float(entrance[1])])
            move_axis_smart(delta)
            print()
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Erro: {e}\n")


if __name__ == "__main__":
    main()
