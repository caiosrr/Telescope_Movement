import itertools
import time
import requests
import numpy as np

# ==== Configurações Alpaca ====
BASE_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
CLIENT_ID = 1
_transaction_ids = itertools.count(1)

# ==== Parâmetros de controle ====
TOLERANCIA_GRAUS = 0.0005
VEL_MIN_LIMITE = 0.001042
VEL_MAX_LIMITE = 6.0
MAX_TEMPO_MOV = 450
MAX_CORRECOES = 10


class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0,
                 output_limits=(None, None),
                 integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self._integral = 0.0
        self._last_error = None
        self._last_time = None

        self.min_output, self.max_output = output_limits
        self.integral_limit = integral_limit

    def reset(self):
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
    
    def clamp_integral(self):
        if self.integral_limit is not None:
            self._integral = np.clip(
                self._integral,
                -self.integral_limit,
                self.integral_limit,
            )

    def update(self, axis, measured_value):
        error = calc_error(axis, self.setpoint, measured_value)
        now = time.time()

        if self._last_time is None:
            self._last_time = now
            self._last_error = error
            return 0.0, error

        dt = now - self._last_time
        de = error - self._last_error

        P = self.kp * error
        self._integral += error * dt

        self.clamp_integral()

        I = self.ki * self._integral
        D = self.kd * (de / dt) if dt > 0 else 0.0

        output = P + I + D

        if self.min_output is not None and self.max_output is not None:
            output = np.clip(output, self.min_output, self.max_output)

        self._last_error = error
        self._last_time = now
        return output, error


def call(method: str, command: str, timeout: float = 5.0, **extra_args):
    params = {"ClientID": CLIENT_ID,
              "ClientTransactionID": next(_transaction_ids)}
    params.update(extra_args.pop("params", {}))

    resp = requests.request(
        method, f"{BASE_URL}/{command}", params=params,
        timeout=timeout, **extra_args
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("ErrorNumber", 0):
        raise RuntimeError(f"{command}: {payload.get('ErrorMessage')}")
    return payload.get("Value")


def ensure_connected():
    if not call("GET", "connected"):
        call("PUT", "connected", data={"Connected": True})


def ensure_unparked():
    try:
        if call("GET", "atpark"):
            if call("GET", "canunpark"):
                call("PUT", "unpark", timeout=10)
    except Exception:
        pass


def ensure_not_tracking():
    try:
        if str(call("GET", "tracking")).lower() in {"true", "1"}:
            call("PUT", "tracking", data={"Tracking": False})
    except Exception:
        pass


def read_altaz():
    az = float(call("GET", "azimuth"))
    alt = float(call("GET", "altitude"))
    return az, alt


def move_axis(axis: int, rate_deg_per_s: float, mount: bool):
    if mount and axis == 0:
        rate_deg_per_s = -rate_deg_per_s
    call("PUT", "moveaxis", data={"Axis": axis, "Rate": float(rate_deg_per_s)})


def calc_error(axis:int, alvo: float, pos: float) -> float:
    if axis == 0:
        diff = (alvo - pos) % 360
        if diff > 180:
            return diff - 360
        return diff
    else:
        return alvo - pos

def move_axis_pid(mount: bool,axis: int, delta_deg: float):
    """Movimento contínuo em UM eixo usando PID."""

    az0, alt0 = read_altaz()
    pos0 = alt0 if axis == 1 else az0
    alvo = pos0 + delta_deg if axis == 1 else (pos0 + delta_deg) % 360

    if axis == 1:
        if alvo > 90:
            print("⚠️ Movimento ultrapassa o limite em altitude. Ajustado para 90°.")
            alvo = 90
        elif alvo < -90:
            print("⚠️ Movimento ultrapassa o limite em altitude. Ajustado para -90°.")
            alvo = -90

    pid = PID(
        kp = 0.75,
        ki = 0.0001,
        kd = 0.04,
        setpoint=alvo,
        output_limits=(-VEL_MAX_LIMITE, VEL_MAX_LIMITE),
        integral_limit = 5,
        )
    if axis == 0:
        print("\nMovimento PID (Azimute):")
        print(f"  Alvo = ({alvo:.2f}, {alt0:.2f})°")
    else:
        print("\nMovimento PID (Altitude):")
        print(f"  Alvo = ({az0:.2f}, {alvo:.2f})°")

    t0 = time.time()
    error_last = None
    inversoes = 0

    try:
        while True:
            az, alt = read_altaz()
            pos = alt if axis == 1 else az

            tempo_decorrido = time.time() - t0
            cmd, error = pid.update(axis, pos)
            error_abs = abs(error)

            if tempo_decorrido > MAX_TEMPO_MOV:
                print("\n⚠️ Tempo limite atingido.")
                break
            if error_abs < TOLERANCIA_GRAUS:
                print("\n✅ Dentro da tolerância.")
                cmd = 0.0
                move_axis(axis, cmd, mount)
                break

            # overshoot antes de usar cmd
            if error_last is not None and error * error_last < 0:
                inversoes += 1
                if inversoes > MAX_CORRECOES:
                    print("\n⚠️ Excesso de inversões. Abortando.")
                    move_axis(axis, 0.0, mount)
                    break
                print("\n↩️ Overshoot — reset PID e pausa curta.")
                pid.reset()
                move_axis(axis, 0.0, mount)
                time.sleep(0.2)
                error_last = error
                continue

            # zona morta + velocidade mínima
            if error_abs < TOLERANCIA_GRAUS:
                cmd = 0.0
            else:
                mag = abs(cmd)
                if 0 < mag < VEL_MIN_LIMITE:
                    cmd = VEL_MIN_LIMITE * np.sign(cmd)

            move_axis(axis, cmd, mount)

            print(
                f"Pos={pos:.4f}°  Erro={error:+.4f}°  Cmd={cmd:+.4f}°/s",
                end="\r",
            )

            error_last = error

            if error_abs > 1.0:
                dt_sleep = 0.1
            elif error_abs > 0.1:
                dt_sleep = 0.05
            else:
                dt_sleep = 0.01

            time.sleep(dt_sleep)

    finally:
        move_axis(axis, 0.0, mount)
        azf, altf = read_altaz()
        if axis == 0:
            print(
                f"\nPos final: ({azf:.4f}, {altf:.4f})°;"
                f" Erro=({alvo - azf:+.6f}, 0)°; Tempo={tempo_decorrido:.2f}s"
            )
        else:
            print(
                f"\nPos final: ({azf:.4f}, {altf:.4f})°;"
                f" Erro=(0, {alvo - altf:+.6f})°; Tempo={tempo_decorrido:.2f}s"
            )


def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    print("Movimento com controle PID (um eixo por vez)\n")
    mount = bool(int(input("mount 1, simulador 0: ")))

    while True:
        try:
            az, alt = read_altaz()
            print(f"Pos atual: Az={az:.3f}°, Alt={alt:.3f}°")
            eixo = int(input("Eixo (0=Az, 1=Alt): ").strip())
            if eixo not in (0, 1):
                raise ValueError("Eixo inválido. Use 0 para Az ou 1 para Alt.")
            delta = float(input("Delta (graus ±): ").strip())
            move_axis_pid(mount, eixo, delta)
            print()
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Erro: {e}\n")


if __name__ == "__main__":
    main()
