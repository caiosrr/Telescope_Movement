import itertools
import time
from dataclasses import dataclass

import requests


CLIENT_ID = 1
DEFAULT_RATE_DEG_S = 0.20
DEFAULT_DURATION_S = 2.0
SETTLE_S = 0.8
SAMPLE_INTERVAL_S = 0.5
DEFAULT_MOUNT1_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
DEFAULT_MOUNT2_URL = "http://10.6.0.34:11111/api/v1/telescope/0"


@dataclass
class AxisResult:
    label: str
    axis: int
    commanded_rate_deg_s: float
    duration_s: float
    az_before: float
    alt_before: float
    az_after: float
    alt_after: float
    delta_az: float
    delta_alt: float
    samples: list[tuple[float, float, float]]


class TelescopeClient:
    def __init__(self, label: str, base_url: str):
        self.label = label
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
            raise RuntimeError(f"{self.label}:{command}: {payload.get('ErrorMessage')}")
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

    def name(self) -> str:
        try:
            return str(self.call("GET", "name"))
        except Exception as exc:
            return f"nome indisponivel ({exc})"

    def read_altaz(self) -> tuple[float, float]:
        az = float(self.call("GET", "azimuth"))
        alt = float(self.call("GET", "altitude"))
        return az, alt

    def can_move_axis(self, axis: int) -> str:
        try:
            value = self.call("GET", "canmoveaxis", params={"Axis": axis})
            return str(value)
        except Exception as exc:
            return f"indisponivel ({exc})"

    def move_axis(self, axis: int, rate_deg_s: float) -> None:
        self.call("PUT", "moveaxis", data={"Axis": axis, "Rate": float(rate_deg_s)})

    def stop(self) -> None:
        for axis in (0, 1):
            try:
                self.move_axis(axis, 0.0)
            except Exception as exc:
                print(f"  Aviso: nao consegui parar eixo {axis} em {self.label}: {exc}")


def signed_delta_az(az_before: float, az_after: float) -> float:
    diff = (az_after - az_before) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def run_axis_pulse(
    telescope: TelescopeClient,
    axis: int,
    rate_deg_s: float,
    duration_s: float,
) -> AxisResult:
    axis_name = "Az" if axis == 0 else "Alt"
    print(
        f"\n[{telescope.label}] Pulso {axis_name}: "
        f"rate={rate_deg_s:+.4f} deg/s por {duration_s:.2f}s"
    )
    az_before, alt_before = telescope.read_altaz()
    print(f"  Antes: Az={az_before:.6f} deg | Alt={alt_before:.6f} deg")

    samples = [(0.0, az_before, alt_before)]
    t0 = time.perf_counter()
    try:
        telescope.move_axis(axis, rate_deg_s)
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= duration_s:
                break
            time.sleep(min(SAMPLE_INTERVAL_S, duration_s - elapsed))
            sample_t = time.perf_counter() - t0
            sample_az, sample_alt = telescope.read_altaz()
            samples.append((sample_t, sample_az, sample_alt))
    finally:
        telescope.move_axis(axis, 0.0)

    time.sleep(SETTLE_S)
    az_after, alt_after = telescope.read_altaz()
    delta_az = signed_delta_az(az_before, az_after)
    delta_alt = alt_after - alt_before
    print(f"  Depois: Az={az_after:.6f} deg | Alt={alt_after:.6f} deg")
    print(f"  Delta:  dAz={delta_az:+.6f} deg | dAlt={delta_alt:+.6f} deg")
    if len(samples) > 2:
        print("  Amostras durante pulso:")
        for sample_t, sample_az, sample_alt in samples[1:]:
            print(f"    t={sample_t:5.2f}s | Az={sample_az:.6f} | Alt={sample_alt:.6f}")

    return AxisResult(
        label=telescope.label,
        axis=axis,
        commanded_rate_deg_s=rate_deg_s,
        duration_s=duration_s,
        az_before=az_before,
        alt_before=alt_before,
        az_after=az_after,
        alt_after=alt_after,
        delta_az=delta_az,
        delta_alt=delta_alt,
        samples=samples,
    )


def print_summary(results: list[AxisResult]) -> None:
    print("\n=== Resumo comparativo ===")
    print(
        "Mount        Eixo Rate_cmd  Dur   dAz       dAlt      "
        "ganho_eixo_cmd"
    )
    for result in results:
        axis_name = "Az" if result.axis == 0 else "Alt"
        measured_axis_delta = result.delta_az if result.axis == 0 else result.delta_alt
        expected = result.commanded_rate_deg_s * result.duration_s
        gain = measured_axis_delta / expected if abs(expected) > 1e-12 else float("nan")
        print(
            f"{result.label:<12} {axis_name:<3} "
            f"{result.commanded_rate_deg_s:+.3f}   {result.duration_s:>4.1f} "
            f"{result.delta_az:+.5f}  {result.delta_alt:+.5f}  {gain:+.3f}"
        )

    print("\nComo ler:")
    print("  ganho_eixo_cmd perto de +1: MoveAxis tem escala/sinal esperado.")
    print("  perto de -1: eixo responde invertido em coordenada Alt/Az.")
    print("  muito maior que 1: driver/mount esta se movendo mais que o rate comandado.")
    print("  dAz grande em pulso Alt, ou dAlt grande em pulso Az: acoplamento entre eixos ou leitura instavel.")
    print("  Se pulso curto da ganho baixo mas pulso longo melhora, ha latencia/rampa de aceleracao.")


def prompt_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if not raw else float(raw)


def main() -> None:
    print("=== Diagnostico comparativo de mounts ASCOM/Alpaca ===")
    print("Este teste usa MoveAxis direto, sem PID. Use rates pequenos e fique pronto para interromper.")

    local_url = input(
        "URL mount 1/local "
        f"[{DEFAULT_MOUNT1_URL}]: "
    ).strip() or DEFAULT_MOUNT1_URL
    remote_url = input(
        "URL mount 2/remoto "
        f"[{DEFAULT_MOUNT2_URL}; digite '-' para pular]: "
    ).strip()
    if not remote_url:
        remote_url = DEFAULT_MOUNT2_URL
    elif remote_url == "-":
        remote_url = ""

    rate = prompt_float("Rate de teste em deg/s", DEFAULT_RATE_DEG_S)
    duration = prompt_float("Duracao do pulso em segundos", DEFAULT_DURATION_S)
    bidirectional = (input("Testar tambem sentido negativo? (s/n) [s]: ").strip().lower() or "s") == "s"

    telescopes = [TelescopeClient("mount1", local_url)]
    if remote_url:
        telescopes.append(TelescopeClient("mount2", remote_url))

    results: list[AxisResult] = []
    try:
        for telescope in telescopes:
            print(f"\nPreparando {telescope.label}: {telescope.base_url}")
            telescope.ensure_ready()
            print(f"  Nome: {telescope.name()}")
            print(f"  CanMoveAxis Az: {telescope.can_move_axis(0)}")
            print(f"  CanMoveAxis Alt: {telescope.can_move_axis(1)}")
            az, alt = telescope.read_altaz()
            print(f"  Pos atual: Az={az:.6f} deg | Alt={alt:.6f} deg")

        input("\nPressione Enter para iniciar os pulsos em cada mount...")

        for telescope in telescopes:
            pulse_plan = [(0, rate), (1, rate)]
            if bidirectional:
                pulse_plan.extend([(0, -rate), (1, -rate)])
            for axis, pulse_rate in pulse_plan:
                results.append(run_axis_pulse(telescope, axis=axis, rate_deg_s=pulse_rate, duration_s=duration))
                input("Pressione Enter para continuar para o proximo pulso...")

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
    finally:
        for telescope in telescopes:
            telescope.stop()

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
