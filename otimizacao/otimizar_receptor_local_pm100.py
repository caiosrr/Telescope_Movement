import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_paths import json_output_path
from controle import mount_control
from otimizacao.otimizar_acoplamento_pm100 import PM100Reader


DEFAULT_WAVELENGTH_NM = 632.8
DEFAULT_SETTLE_S = 0.8
DEFAULT_SAMPLES = 5
DEFAULT_STEPS = "0.02,0.01,0.005,0.002"
RESULTS_JSON = json_output_path("otimizacao_receptor_local_pm100.json")


@dataclass
class Measurement:
    timestamp_epoch: float
    power_w: float
    power_uw: float
    label: str


@dataclass
class Trial:
    cycle: int
    step_deg: float
    delta_az_deg: float
    delta_alt_deg: float
    power_uw: float
    accepted: bool
    timestamp_epoch: float


def parse_steps(text: str) -> list[float]:
    if ";" in text:
        return [abs(float(item.strip().replace(",", "."))) for item in text.split(";") if item.strip()]
    return [abs(float(item.strip())) for item in text.split(",") if item.strip()]


def measure_pm(pm: PM100Reader, samples: int, label: str, log: list[Measurement]) -> float:
    power_w = pm.read_average_w(samples)
    power_uw = power_w * 1e6
    log.append(
        Measurement(
            timestamp_epoch=time.time(),
            power_w=power_w,
            power_uw=power_uw,
            label=label,
        )
    )
    print(f"{label}: {power_uw:.5f} uW")
    return power_uw


def move_local(delta_az: float, delta_alt: float, settle_s: float) -> None:
    if delta_az == 0.0 and delta_alt == 0.0:
        return
    mount_control.move_axes_pid_2d(True, delta_az, delta_alt)
    time.sleep(settle_s)


def test_offset(
    pm: PM100Reader,
    samples: int,
    settle_s: float,
    offset_az: float,
    offset_alt: float,
    cycle: int,
    step: float,
    measurements: list[Measurement],
    trials: list[Trial],
) -> float:
    label = f"teste ciclo={cycle} step={step:.5f} dAz={offset_az:+.5f} dAlt={offset_alt:+.5f}"
    move_local(offset_az, offset_alt, settle_s)
    power_uw = measure_pm(pm, samples, label, measurements)
    move_local(-offset_az, -offset_alt, settle_s)
    trials.append(
        Trial(
            cycle=cycle,
            step_deg=step,
            delta_az_deg=offset_az,
            delta_alt_deg=offset_alt,
            power_uw=power_uw,
            accepted=False,
            timestamp_epoch=time.time(),
        )
    )
    return power_uw


def optimize_local_receiver(
    pm: PM100Reader,
    steps: list[float],
    cycles: int,
    samples: int,
    settle_s: float,
) -> tuple[list[Measurement], list[Trial]]:
    measurements: list[Measurement] = []
    trials: list[Trial] = []

    current_uw = measure_pm(pm, samples, "inicial", measurements)

    for cycle in range(1, cycles + 1):
        print(f"\n=== Ciclo {cycle}/{cycles} ===")
        for step in steps:
            print(f"\nBusca local com passo {step:.5f} deg")
            candidates = [
                (0.0, 0.0),
                (+step, 0.0),
                (-step, 0.0),
                (0.0, +step),
                (0.0, -step),
                (+step, +step),
                (+step, -step),
                (-step, +step),
                (-step, -step),
            ]

            best_az = 0.0
            best_alt = 0.0
            best_uw = current_uw

            for offset_az, offset_alt in candidates[1:]:
                power_uw = test_offset(
                    pm,
                    samples,
                    settle_s,
                    offset_az,
                    offset_alt,
                    cycle,
                    step,
                    measurements,
                    trials,
                )
                if power_uw > best_uw:
                    best_uw = power_uw
                    best_az = offset_az
                    best_alt = offset_alt

            if best_az == 0.0 and best_alt == 0.0:
                print(f"Sem melhora para passo {step:.5f}; mantendo posicao.")
                continue

            print(
                f"Melhor vizinho: dAz={best_az:+.5f} dAlt={best_alt:+.5f} "
                f"| {current_uw:.5f} -> {best_uw:.5f} uW"
            )
            move_local(best_az, best_alt, settle_s)
            current_uw = measure_pm(pm, samples, "apos aceitar melhor vizinho", measurements)

            for trial in reversed(trials):
                if (
                    trial.cycle == cycle
                    and trial.step_deg == step
                    and trial.delta_az_deg == best_az
                    and trial.delta_alt_deg == best_alt
                ):
                    trial.accepted = True
                    break

    return measurements, trials


def save_results(pm: PM100Reader, measurements: list[Measurement], trials: list[Trial], args) -> None:
    payload = {
        "timestamp_epoch": time.time(),
        "pm100_resource": pm.resource_name,
        "pm100_idn": pm.idn,
        "wavelength_nm": args.wavelength_nm,
        "steps_deg": parse_steps(args.steps),
        "cycles": args.cycles,
        "samples": args.samples,
        "settle_s": args.settle_s,
        "measurements": [asdict(item) for item in measurements],
        "trials": [asdict(item) for item in trials],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nLog salvo em: {RESULTS_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Otimiza acoplamento movendo apenas o mount receptor local e lendo o PM100."
    )
    parser.add_argument("--base-url", default=mount_control.BASE_URL)
    parser.add_argument("--pm-resource", default=None)
    parser.add_argument("--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--steps", default=DEFAULT_STEPS, help="Passos em graus. Ex: 0.02,0.01,0.005")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    mount_control.BASE_URL = args.base_url
    mount_control.ensure_connected()
    mount_control.ensure_unparked()
    mount_control.ensure_not_tracking()

    pm = PM100Reader(args.wavelength_nm, args.pm_resource)
    print(f"PM100: {pm.idn}")
    print(f"VISA resource: {pm.resource_name}")
    print(f"Mount receptor local: {args.base_url}")
    print(f"Passos: {parse_steps(args.steps)} deg | ciclos={args.cycles}")
    print("Ctrl+C interrompe e salva o log.\n")

    measurements: list[Measurement] = []
    trials: list[Trial] = []
    try:
        measurements, trials = optimize_local_receiver(
            pm=pm,
            steps=parse_steps(args.steps),
            cycles=args.cycles,
            samples=args.samples,
            settle_s=args.settle_s,
        )
    except KeyboardInterrupt:
        print("\nOtimizacao interrompida pelo usuario.")
    finally:
        try:
            mount_control.move_axis(0, 0.0, True)
            mount_control.move_axis(1, 0.0, True)
        except Exception:
            pass
        save_results(pm, measurements, trials, args)


if __name__ == "__main__":
    main()
