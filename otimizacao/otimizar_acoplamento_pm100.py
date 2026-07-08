import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import json_output_path
from controle.mov_mount_remoto import TelescopeClient, move_relative_remote


DEFAULT_RECEIVER_URL = "http://127.0.0.1:11111/api/v1/telescope/0"
DEFAULT_EMITTER_AGENT_URL = "http://10.6.0.145:18080"
DEFAULT_WAVELENGTH_NM = 632.8
DEFAULT_SETTLE_S = 0.8
DEFAULT_SAMPLES = 5

RESULTS_JSON = json_output_path("otimizacao_acoplamento_pm100.json")


@dataclass
class Measurement:
    timestamp_epoch: float
    power_w: float
    power_uw: float
    label: str


@dataclass
class MoveLog:
    target: str
    axis: str
    delta_deg: float
    accepted: bool
    before_uw: float
    after_uw: float
    timestamp_epoch: float


class PM100Reader:
    def __init__(self, wavelength_nm: float, resource_name: str | None = None):
        try:
            import pyvisa
        except ImportError as exc:
            raise RuntimeError(
                "pyvisa nao esta instalado. No PC2 rode: python -m pip install pyvisa"
            ) from exc

        self.pyvisa = pyvisa
        self.rm = pyvisa.ResourceManager()
        resources = list(self.rm.list_resources())
        if not resources:
            raise RuntimeError("Nenhum recurso VISA encontrado. Confira driver Thorlabs/OPM e USB.")

        if resource_name is None:
            resource_name = self._pick_pm100_resource(resources)
        self.resource_name = resource_name
        self.instrument = self.rm.open_resource(resource_name)
        self.instrument.timeout = 3000

        self.idn = self._query_first(["*IDN?"]).strip()
        self.set_wavelength(wavelength_nm)

    @staticmethod
    def _pick_pm100_resource(resources: list[str]) -> str:
        for resource in resources:
            upper = resource.upper()
            if "1313" in upper or "8072" in upper or "PM100" in upper:
                return resource
        for resource in resources:
            if resource.upper().startswith("USB"):
                return resource
        return resources[0]

    def _query_first(self, commands: list[str]) -> str:
        last_exc = None
        for command in commands:
            try:
                return str(self.instrument.query(command))
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Falha consultando PM100 com {commands}: {last_exc}")

    def set_wavelength(self, wavelength_nm: float) -> None:
        commands = [
            f"SENS:CORR:WAV {wavelength_nm}",
            f"SENSE:CORRECTION:WAVELENGTH {wavelength_nm}",
        ]
        for command in commands:
            try:
                self.instrument.write(command)
                return
            except Exception:
                continue
        print("Aviso: nao consegui configurar wavelength via SCPI; confira no OPM.")

    def read_power_w(self) -> float:
        response = self._query_first(["READ?", "MEAS:POW?", "MEASURE:POWER?"])
        return float(response.strip().split(",")[0])

    def read_average_w(self, samples: int, delay_s: float = 0.08) -> float:
        values = []
        for _ in range(samples):
            values.append(self.read_power_w())
            time.sleep(delay_s)
        return sum(values) / len(values)


def call_agent(agent_url: str, endpoint: str, payload: dict | None = None) -> dict:
    base = agent_url.rstrip("/")
    data = None
    headers = {}
    method = "GET" if payload is None else "POST"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(f"{base}/{endpoint.lstrip('/')}", data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


class CouplingOptimizer:
    def __init__(
        self,
        pm: PM100Reader,
        receiver: TelescopeClient,
        emitter_agent_url: str | None,
        settle_s: float,
        samples: int,
    ):
        self.pm = pm
        self.receiver = receiver
        self.emitter_agent_url = emitter_agent_url
        self.settle_s = settle_s
        self.samples = samples
        self.measurements: list[Measurement] = []
        self.moves: list[MoveLog] = []

    def measure(self, label: str) -> float:
        power_w = self.pm.read_average_w(self.samples)
        measurement = Measurement(
            timestamp_epoch=time.time(),
            power_w=power_w,
            power_uw=power_w * 1e6,
            label=label,
        )
        self.measurements.append(measurement)
        print(f"{label}: {measurement.power_uw:.4f} uW")
        return measurement.power_uw

    def move_receiver(self, axis: str, delta_deg: float) -> None:
        delta_az = delta_deg if axis == "az" else 0.0
        delta_alt = delta_deg if axis == "alt" else 0.0
        move_relative_remote(self.receiver, delta_az, delta_alt)

    def move_emitter(self, axis: str, delta_deg: float) -> None:
        if self.emitter_agent_url is None:
            raise RuntimeError("Emitter agent nao configurado.")
        payload = {
            "delta_az_deg": delta_deg if axis == "az" else 0.0,
            "delta_alt_deg": delta_deg if axis == "alt" else 0.0,
            "tolerance_deg": 0.005,
        }
        result = call_agent(self.emitter_agent_url, "/move_relative", payload)
        if not result.get("ok"):
            raise RuntimeError(f"Movimento do emissor falhou: {result}")

    def try_step(self, target: str, axis: str, delta_deg: float, current_uw: float) -> tuple[float, bool]:
        print(f"\nTeste {target} {axis} {delta_deg:+.5f} deg")
        mover = self.move_receiver if target == "receiver" else self.move_emitter
        mover(axis, delta_deg)
        time.sleep(self.settle_s)
        after_uw = self.measure(f"apos {target} {axis} {delta_deg:+.5f}")
        accepted = after_uw > current_uw
        if accepted:
            print(f"  aceito: {current_uw:.4f} -> {after_uw:.4f} uW")
            new_power = after_uw
        else:
            print(f"  rejeitado: {current_uw:.4f} -> {after_uw:.4f} uW; voltando")
            mover(axis, -delta_deg)
            time.sleep(self.settle_s)
            new_power = self.measure("apos voltar")

        self.moves.append(
            MoveLog(
                target=target,
                axis=axis,
                delta_deg=delta_deg,
                accepted=accepted,
                before_uw=current_uw,
                after_uw=after_uw,
                timestamp_epoch=time.time(),
            )
        )
        return new_power, accepted

    def coordinate_search(
        self,
        target: str,
        current_uw: float,
        step_deg: float,
        passes: int = 1,
    ) -> float:
        for _ in range(passes):
            improved = False
            for axis in ("az", "alt"):
                for sign in (+1.0, -1.0):
                    current_uw, accepted = self.try_step(target, axis, sign * step_deg, current_uw)
                    improved = improved or accepted
            if not improved:
                break
        return current_uw

    def save(self) -> None:
        payload = {
            "timestamp_epoch": time.time(),
            "pm100_resource": self.pm.resource_name,
            "pm100_idn": self.pm.idn,
            "measurements": [asdict(item) for item in self.measurements],
            "moves": [asdict(item) for item in self.moves],
        }
        RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nLog salvo em: {RESULTS_JSON}")


def parse_steps(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimiza acoplamento usando PM100USB como metrica.")
    parser.add_argument("--receiver-url", default=DEFAULT_RECEIVER_URL)
    parser.add_argument("--emitter-agent-url", default=None)
    parser.add_argument("--pm-resource", default=None)
    parser.add_argument("--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--receiver-steps", default="0.02,0.01,0.005")
    parser.add_argument("--emitter-steps", default="0.02,0.01")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    pm = PM100Reader(args.wavelength_nm, args.pm_resource)
    print(f"PM100: {pm.idn}")
    print(f"VISA resource: {pm.resource_name}")

    receiver = TelescopeClient(args.receiver_url)
    receiver.ensure_ready()
    print(f"Receiver local: {args.receiver_url}")

    if args.emitter_agent_url:
        print(f"Emitter agent: {args.emitter_agent_url}")
        print(call_agent(args.emitter_agent_url, "/health"))
    else:
        print("Emitter agent desativado; otimizando apenas receptor.")

    optimizer = CouplingOptimizer(
        pm=pm,
        receiver=receiver,
        emitter_agent_url=args.emitter_agent_url,
        settle_s=args.settle_s,
        samples=args.samples,
    )

    current_uw = optimizer.measure("inicial")
    try:
        for cycle in range(1, args.cycles + 1):
            print(f"\n=== Ciclo {cycle}/{args.cycles}: receptor ===")
            for step in parse_steps(args.receiver_steps):
                current_uw = optimizer.coordinate_search("receiver", current_uw, step)

            if args.emitter_agent_url:
                print(f"\n=== Ciclo {cycle}/{args.cycles}: emissor + refinamento receptor ===")
                for step in parse_steps(args.emitter_steps):
                    current_uw = optimizer.coordinate_search("emitter", current_uw, step)
                    refine_step = min(step, parse_steps(args.receiver_steps)[-1])
                    current_uw = optimizer.coordinate_search("receiver", current_uw, refine_step)
    except KeyboardInterrupt:
        print("\nOtimização interrompida.")
    finally:
        receiver.stop()
        if args.emitter_agent_url:
            try:
                call_agent(args.emitter_agent_url, "/stop", {})
            except Exception as exc:
                print(f"Aviso: nao consegui parar emissor via agent: {exc}")
        optimizer.save()


if __name__ == "__main__":
    main()
