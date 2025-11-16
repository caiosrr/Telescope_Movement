"""Controle básico de MoveAxis para mounts em modo Alt/Az."""

import itertools
import time

import requests

# URL base do driver Alpaca; ajuste se o servidor estiver em outro endereço/porta
BASE_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
CLIENT_ID = 1  # Identificador arbitrário do cliente (qualquer inteiro > 0)
PERIODO_LEITURA = 0.1  # s, intervalo entre leituras durante o movimento

# Gerador sequencial para ClientTransactionID, obrigatório no protocolo Alpaca
_transaction_ids = itertools.count(1)


def call(method: str, command: str, timeout: float = 5.0, **request_kwargs):
    """Envia requisições Alpaca incluindo ClientID/ClientTransactionID."""
    params = {
        "ClientID": CLIENT_ID,
        "ClientTransactionID": next(_transaction_ids),
    }
    params.update(request_kwargs.pop("params", {}))
    response = requests.request(
        method,
        f"{BASE_URL}/{command}",
        params=params,
        timeout=timeout,
        **request_kwargs,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("ErrorNumber", 0):
        raise RuntimeError(f"{command}: {payload.get('ErrorMessage', 'erro desconhecido')}")
    return payload.get("Value")


def ensure_connected() -> None:
    """Garante conexão com o mount antes de qualquer comando."""
    if not call("GET", "connected"):
        call("PUT", "connected", data={"Connected": True})


def ensure_unparked() -> None:
    """Sai do estado parked quando possível."""
    try:
        at_park = call("GET", "atpark")
    except RuntimeError:
        return

    is_parked = bool(at_park) if not isinstance(at_park, str) else at_park.lower() == "true"
    if not is_parked:
        return

    try:
        can_unpark = call("GET", "canunpark")
    except RuntimeError:
        can_unpark = True

    if isinstance(can_unpark, str):
        can_unpark = can_unpark.lower() == "true"

    if not can_unpark:
        raise RuntimeError("Mount estacionado e sem suporte a unpark via driver.")

    call("PUT", "unpark", timeout=10)


def ensure_not_tracking() -> None:
    """Desativa o tracking para evitar drift entre comandos."""
    try:
        tracking_state = call("GET", "tracking")
    except RuntimeError:
        return

    if str(tracking_state).strip().lower() in {"true", "1"}:
        call("PUT", "tracking", data={"Tracking": False}, timeout=5)


def read_axis_rate_limit(axis: int) -> float:
    """Consulta o limite máximo de velocidade para o eixo solicitado."""
    ranges = call("GET", "axisrates", params={"Axis": axis}) or []
    if not ranges:
        raise RuntimeError("Driver não informou limites para MoveAxis.")
    max_pos = max(abs(float(r["Maximum"])) for r in ranges)
    max_neg = max(abs(float(r["Minimum"])) for r in ranges)
    return max(max_pos, max_neg)


def read_altaz() -> tuple[float, float]:
    """Retorna a posição atual em Azimute/Altitude."""
    az = float(call("GET", "azimuth"))
    alt = float(call("GET", "altitude"))
    return az, alt


def prompt_axis() -> int:
    """Pergunta ao usuário qual eixo deseja movimentar."""
    text = input("Escolha o eixo (0=Azimute, 1=Altitude, q para sair): ").strip().lower()
    if text in {"q", "quit", "exit"}:
        raise SystemExit
    if text in {"0", "az", "azimute", "azimuth"}:
        return 0
    if text in {"1", "alt", "altitude"}:
        return 1
    raise ValueError("Eixo inválido. Digite 0 (Azimute) ou 1 (Altitude).")


def prompt_float(message: str) -> float:
    """Lê um número decimal do usuário."""
    try:
        return float(input(message).strip())
    except ValueError as exc:
        raise ValueError("Informe um valor numérico válido.") from exc


def move_axis(axis: int, rate_deg_per_s: float, duration_s: float) -> None:
    """Executa um único MoveAxis e relata a posição em tempo real.

    - Envia a taxa (o sinal indica o sentido)
    - Durante 'duration_s', lê Az/Alt a cada PERIODO_LEITURA e imprime na mesma linha
    - Finaliza com Rate=0.0
    """
    if rate_deg_per_s == 0:
        raise ValueError("A taxa não pode ser zero; use sinal negativo para inverter o sentido.")
    if duration_s <= 0:
        raise ValueError("A duração deve ser positiva.")

    if axis == 0:
        rate_to_send = -rate_deg_per_s
    else:
        rate_to_send = rate_deg_per_s
    call("PUT", "moveaxis", data={"Axis": axis, "Rate": rate_to_send}, timeout=5)

    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_s:
                break
            try:
                az, alt = read_altaz()
                print(
                    f"  t={elapsed:5.2f}s | Az={az:.10f}°, Alt={alt:.10f}°",
                    end="\r",
                    flush=True,
                )
            except Exception:
                # Se a leitura falhar, apenas espera o próximo ciclo
                pass
            time.sleep(PERIODO_LEITURA)
    finally:
        call("PUT", "moveaxis", data={"Axis": axis, "Rate": 0.0}, timeout=5)
        # quebra de linha para não sobrescrever a linha com \r
        try:
            az, alt = read_altaz()
            print(f"  t={duration_s:5.2f}s | Az={az:.10f}°, Alt={alt:.10f}°")
        except Exception:
            print()


def main() -> None:
    """Loop principal: mostra posição, lê parâmetros e envia MoveAxis."""
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    print("Controle direto de MoveAxis (Ctrl+C para sair).")

    while True:
        try:
            az, alt = read_altaz()
            print(f"Posição atual: Az={az:.10f}°, Alt={alt:.10f}°")

            axis = prompt_axis()
            limit = read_axis_rate_limit(axis)
            print(f"Taxa máxima para o eixo {axis}: {limit:.4f} deg/s")

            rate = prompt_float("  Taxa desejada (deg/s, pode ser negativa): ")
            if rate == 0 or abs(rate) > limit:
                raise ValueError("Taxa fora do intervalo permitido (use até ±limite informado).")

            duration = prompt_float("  Duração do movimento (s): ")

            print(f"Movendo eixo {axis} a {rate:.4f} deg/s por {duration:.2f} s...")
            move_axis(axis, rate, duration)
            print("Comando concluído.\n")
        except SystemExit:
            print("Saindo...")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Falha: {exc}\n")


if __name__ == "__main__":
    main()
