"""MoveAxis contínuo com desaceleração exponencial e correção de overshoot suave."""

import itertools
import time
import math
import requests
#### Ver erro +-180° e testar movimento simultaneo em ambos eixos ####

BASE_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
CLIENT_ID = 1
_transaction_ids = itertools.count(1)

# ==== Parâmetros de controle ====
TOLERANCIA_GRAUS = 0.001
VEL_MIN_LIMITE = 0.001042  # menor velocidade efetiva do mount
VEL_MAX_LIMITE = 6.0
MAX_TEMPO_MOV = 450
MAX_CORRECOES = 10  # máximo de inversões de direção

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
    return az, alt


def move_axis(axis: int, rate_deg_per_s: float): # Envia comando MoveAxis com eixo e velocidade. 
    if axis == 0:  # necessário para o mount ZWO
        rate_deg_per_s = -rate_deg_per_s
    call("PUT", "moveaxis", data={"Axis": axis, "Rate": rate_deg_per_s})


# ==== Cálculo de velocidade ====
def calc_vel_exponencial(k_exp: float, erro_abs: float, erro_inicial_abs: float, last_vel: float = VEL_MAX_LIMITE):
    
    if erro_abs > 12.0:
        faixa = "longa"
        return VEL_MAX_LIMITE, faixa, k_exp

    elif erro_abs > 8:
        faixa = "média"
        vel_max = min(6.0, last_vel)

    elif erro_abs > 1.0:
        faixa = "curta"
        vel_max = min(1.0, last_vel)
    
    elif erro_abs > 0.1:
        faixa = "fina"
        vel_max = min(0.1, last_vel)

    elif erro_abs > 0.01:
        faixa = "ultra fina"
        vel_max = min(0.01, last_vel)
    
    else:
        faixa = "mínima"
        return VEL_MIN_LIMITE, faixa, k_exp
    
    k_exp += 0.0001
    fator = math.exp(-k_exp * (1 - erro_abs/erro_inicial_abs))
    vel_max *= fator
    vel = max(VEL_MIN_LIMITE, vel_max)
    return vel, faixa, k_exp


# ==== Movimento inteligente ====
def move_axis_smart(axis: int, delta_deg: float):
    """Movimento contínuo com desaceleração exponencial e correção de overshoot."""
    erro_inicial = abs(delta_deg)
    k_exp = 0.0  # valor inicial do coeficiente exponencial
    vel_max, faixa, k_exp = calc_vel_exponencial(k_exp, erro_inicial, erro_inicial)
    sentido = 1 if delta_deg > 0 else -1

    az0, alt0 = read_altaz()
    if axis != 0 and axis != 1:
        raise ValueError("Eixo inválido. Use 0 para Azimute ou 1 para Altitude.")
    pos0 = alt0 if axis == 1 else az0
    alvo = pos0 + delta_deg

    # --- azimute circular (0–360°) ---
    if axis == 0:
        alvo = (alvo + 360) % 360
        desloc = ((alvo - pos0 + 540) % 360) - 180
        alvo_real = (pos0 + desloc) % 360
        delta_deg = desloc
    else:
        alvo_real = alvo
        if alvo_real > 90 or alvo_real < -90:
            print(f"⚠️ Movimento para {alvo_real:.2f}° ultrapassa ±90° em altitude.")
            continuar = input("Deseja continuar mesmo assim? (s/n): ").strip().lower()
            if continuar != "s":
                print("Movimento cancelado por segurança.")
                return
            alvo_real = max(min(alvo_real, 90), -90)

    if axis == 0:
        print(f"\nMovimento inteligente (exponencial) no eixo Azimute:")
        print(f"  Eixo = Az | Alvo = ({alvo_real:.2f}, {alt0:.2f})° | VelMáx = {vel_max:.2f}°/s")
    else:
        print(f"\nMovimento inteligente (exponencial) no eixo Altitude:")
        print(f"  Eixo = Alt | Alvo = ({az0:.2f}, {alvo_real:.2f})° | VelMáx = {vel_max:.2f}°/s")

    t0 = time.time()
    move_axis(axis, sentido * vel_max)

    inversoes = 0
    erro_prev = delta_deg
    last_vel = vel_max
    k_faixa_atual = "longa"  # <<< controle de faixa K_EXP atual >>>

    try:
        while True:
            az, alt = read_altaz()
            pos = alt if axis == 1 else az

            # erro com circularidade no azimute
            if axis == 0:
                erro = ((alvo_real - pos + 540) % 360) - 180
            else:
                erro = alvo_real - pos

            erro_abs = abs(erro)
            tempo_decorrido = time.time() - t0

            if tempo_decorrido > MAX_TEMPO_MOV:
                print("⚠️ Tempo limite atingido.")
                break
            if erro_abs < TOLERANCIA_GRAUS:
                print(" " * 80, end="\r")
                print(f"✅ Dentro da tolerância ({TOLERANCIA_GRAUS}°).")
                break

            # overshoot
            sign_flip = (erro * erro_prev < 0)
            if sign_flip:
                inversoes += 1
                print(f"\n↩️ Overshoot detectado (#{inversoes}). Invertendo sentido e reduzindo velocidade...")
                if inversoes > MAX_CORRECOES:
                    print("⚠️ Número máximo de correções atingido, parando.")
                    break

                sentido *= -1
                vel_max *= 0.3
                move_axis(axis, 0.0)
                time.sleep(0.7)
                move_axis(axis, sentido * vel_max)
                erro_prev = erro
                continue
            
            new_vel, faixa, k_exp = calc_vel_exponencial(k_exp, erro_abs, erro_inicial, last_vel)
            # --- re-engate de velocidade se mudar de faixa ---
            if faixa != k_faixa_atual:
                print(f"\n🔄 Mudança de faixa → {faixa} — reengatando motor.")
                move_axis(axis, 0.0)
                time.sleep(0.7)
                k_faixa_atual = faixa

            new_vel = min(new_vel, last_vel)

            move_axis(axis, sentido * new_vel)
            print(f"  Pos = {pos:.4f}°| Erro = {erro:+.4f}°| Vel = {new_vel:.4f}°/s  ", end="\r")


            erro_prev = erro
            last_vel = new_vel
            time.sleep(0.2)

    finally:
        move_axis(axis, 0.0)
        azf, altf = read_altaz()
        if axis == 0:
            print(f"\nPos final: ({azf:.2f}, {altf:.2f})°; Erro = ({alvo_real - azf:+.4f}, 0)°, Tempo = {tempo_decorrido:.2f}s")
        else:
            print(f"\nPos final: ({azf:.2f}, {altf:.2f})°; Erro = (0, {alvo_real - altf:+.4f})°, Tempo = {tempo_decorrido:.2f}s")


# ==== Programa principal ====
def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    print("Movimento com desaceleração exponencial \n")

    while True:
        try:
            az, alt = read_altaz()
            print(f"Pos atual: Az = {az:.2f}°, Alt = {alt:.2f}°")
            eixo = int(input("Eixo (0=Az, 1=Alt): ").strip())
            delta = float(input("Delta (graus ±): ").strip())
            move_axis_smart(eixo, delta)
            print()
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Erro: {e}\n")


if __name__ == "__main__":
    main()
