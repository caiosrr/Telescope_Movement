import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests


DEFAULT_BASE_URL = "http://10.6.0.34:11111/api/v1/telescope/0"
CLIENT_ID = 1

TOLERANCIA_GRAUS = 0.0050
CONTROL_PERIOD_S = 0.45
MAX_TEMPO_MOV = 180.0
SETTLE_AFTER_STOP_S = 0.8

VEL_MAX_AZ = 0.70
VEL_MAX_ALT = 0.55
VEL_MIN = 0.010
CMD_ZERO_SNAP = 0.004

KP_AZ = 0.65
KP_ALT = 0.40
KD_AZ = 0.02
KD_ALT = 0.00

# Diagnostics showed positive MoveAxis rate decreases azimuth on these AM5 mounts.
AZ_RATE_SIGN = -1.0
ALT_RATE_SIGN = 1.0


class TelescopeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.transaction_ids = itertools.count(1)

    def call(self, method: str, command: str, timeout: float = 5.0, **extra_args):
        params = {
            "ClientID": CLIENT_ID,
            "ClientTransactionID": next(self.transaction_ids),
        }
        params.update(extra_args.pop("params", {}))
        resp = self.session.request(
            method,
            f"{self.base_url}/{command}",
            params=params,
            timeout=timeout,
            **extra_args,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("ErrorNumber", 0):
            raise RuntimeError(f"{command}: {payload.get('ErrorMessage')}")
        return payload.get("Value")

    def ensure_ready(self) -> None:
        if not bool(self.call("GET", "connected")):
            self.call("PUT", "connected", data={"Connected": True})
        try:
            if bool(self.call("GET", "atpark")) and bool(self.call("GET", "canunpark")):
                self.call("PUT", "unpark", timeout=10)
        except Exception:
            pass
        try:
            if str(self.call("GET", "tracking")).lower() in {"true", "1"}:
                self.call("PUT", "tracking", data={"Tracking": False})
        except Exception:
            pass

    def read_altaz(self) -> tuple[float, float]:
        az = float(self.call("GET", "azimuth"))
        alt = float(self.call("GET", "altitude"))
        return az, alt

    def move_axis(self, axis: int, coordinate_rate_deg_s: float) -> None:
        if axis == 0:
            rate = AZ_RATE_SIGN * coordinate_rate_deg_s
        else:
            rate = ALT_RATE_SIGN * coordinate_rate_deg_s
        self.call("PUT", "moveaxis", data={"Axis": axis, "Rate": float(rate)}, timeout=3.0)

    def stop(self) -> None:
        for axis in (0, 1):
            try:
                self.call("PUT", "moveaxis", data={"Axis": axis, "Rate": 0.0}, timeout=2.0)
            except Exception as exc:
                print(f"Aviso: falha ao parar eixo {axis}: {exc}")


def calc_error_az(target: float, current: float) -> float:
    diff = (target - current) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def apply_min_velocity(cmd: float) -> float:
    if abs(cmd) < CMD_ZERO_SNAP:
        return 0.0
    if 0.0 < abs(cmd) < VEL_MIN:
        return float(VEL_MIN * np.sign(cmd))
    return float(cmd)


def axis_command(error: float, prev_error: float | None, dt: float, kp: float, kd: float, vmax: float) -> float:
    derivative = 0.0 if prev_error is None or dt <= 0 else (error - prev_error) / dt
    cmd = (kp * error) + (kd * derivative)
    cmd = float(np.clip(cmd, -vmax, vmax))
    return apply_min_velocity(cmd)


def move_relative_remote(
    telescope: TelescopeClient,
    delta_az: float,
    delta_alt: float,
    tolerance: float = TOLERANCIA_GRAUS,
) -> bool:
    az0, alt0 = telescope.read_altaz()
    target_az = (az0 + delta_az) % 360.0
    target_alt = float(np.clip(alt0 + delta_alt, -90.0, 90.0))

    print("\nMovimento remoto PID lento:")
    print(f"  Inicio: Az={az0:.6f} deg | Alt={alt0:.6f} deg")
    print(f"  Alvo:   Az={target_az:.6f} deg | Alt={target_alt:.6f} deg")
    print(f"  Delta:  dAz={delta_az:+.6f} deg | dAlt={delta_alt:+.6f} deg")

    prev_err_az = None
    prev_err_alt = None
    last_t = time.perf_counter()
    t0 = last_t
    az_done = False
    alt_done = False

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:
                now = time.perf_counter()
                dt = max(now - last_t, 1e-3)
                last_t = now

                az, alt = telescope.read_altaz()
                err_az = calc_error_az(target_az, az)
                err_alt = target_alt - alt

                az_done = abs(err_az) <= tolerance
                alt_done = abs(err_alt) <= tolerance

                if az_done and alt_done:
                    print("\nDentro da tolerancia. Parando.")
                    return True

                if now - t0 > MAX_TEMPO_MOV:
                    print("\nTempo limite atingido. Parando.")
                    return False

                cmd_az = 0.0 if az_done else axis_command(err_az, prev_err_az, dt, KP_AZ, KD_AZ, VEL_MAX_AZ)
                cmd_alt = 0.0 if alt_done else axis_command(err_alt, prev_err_alt, dt, KP_ALT, KD_ALT, VEL_MAX_ALT)

                fut_az = executor.submit(telescope.move_axis, 0, cmd_az)
                fut_alt = executor.submit(telescope.move_axis, 1, cmd_alt)
                fut_az.result()
                fut_alt.result()

                print(
                    f"\rAz err={err_az:+.5f} cmd={cmd_az:+.4f} | "
                    f"Alt err={err_alt:+.5f} cmd={cmd_alt:+.4f}",
                    end="",
                    flush=True,
                )

                prev_err_az = err_az
                prev_err_alt = err_alt
                time.sleep(CONTROL_PERIOD_S)
        finally:
            telescope.stop()
            time.sleep(SETTLE_AFTER_STOP_S)
            azf, altf = telescope.read_altaz()
            print(f"\nFinal: Az={azf:.6f} deg | Alt={altf:.6f} deg")
            print(
                f"Erro final: dAz={calc_error_az(target_az, azf):+.6f} deg | "
                f"dAlt={target_alt - altf:+.6f} deg"
            )


def main() -> None:
    print("=== Movimento PID lento para mount remoto ===")
    base_url = input(f"URL do mount remoto [{DEFAULT_BASE_URL}]: ").strip() or DEFAULT_BASE_URL
    telescope = TelescopeClient(base_url)
    telescope.ensure_ready()
    az, alt = telescope.read_altaz()
    print(f"Pos atual: Az={az:.6f} deg | Alt={alt:.6f} deg")

    while True:
        try:
            delta_az = float(input("\nDelta Azimute (graus): ").strip())
            delta_alt = float(input("Delta Altitude (graus): ").strip())
            if delta_az == 0.0 and delta_alt == 0.0:
                print("Nenhum movimento ordenado.")
                continue
            move_relative_remote(telescope, delta_az, delta_alt)
        except KeyboardInterrupt:
            print("\nSaindo...")
            telescope.stop()
            break
        except Exception as exc:
            telescope.stop()
            print(f"Erro: {exc}")


if __name__ == "__main__":
    main()
