import argparse
import cv2
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
from controle.alvo_alinhamento import salvar_alvo
from foco_multiplos import Center_of_Mass_foco_temp as foco_temp
from otimizacao.otimizar_acoplamento_pm100 import PM100Reader


DEFAULT_WAVELENGTH_NM = 632.8
DEFAULT_SETTLE_S = 0.8
DEFAULT_SAMPLES = 7
DEFAULT_WARMUP_SAMPLES = 5
DEFAULT_STEPS = "0.02,0.01,0.005,0.002"
DEFAULT_CAMERA_EXPOSURE_S = 32e-6
MIN_ACCEPT_IMPROVEMENT_UW = 0.002
RESULTS_JSON = json_output_path("otimizacao_receptor_local_pm100.json")
CAMERA_TARGET_DEBUG = ROOT_DIR / "resultados" / "debug" / "otimizacao_receptor_alvo_camera.png"


@dataclass
class Measurement:
    timestamp_epoch: float
    power_w: float
    power_uw: float
    min_uw: float
    max_uw: float
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


@dataclass
class AcceptedMove:
    cycle: int
    step_deg: float
    delta_az_deg: float
    delta_alt_deg: float
    baseline_before_uw: float
    baseline_after_scan_uw: float
    candidate_uw: float
    confirmed_uw: float
    timestamp_epoch: float


def parse_steps(text: str) -> list[float]:
    if ";" in text:
        return [abs(float(item.strip().replace(",", "."))) for item in text.split(";") if item.strip()]
    return [abs(float(item.strip())) for item in text.split(",") if item.strip()]


def measure_pm(
    pm: PM100Reader,
    samples: int,
    warmup_samples: int,
    label: str,
    log: list[Measurement],
) -> float:
    for _ in range(max(0, warmup_samples)):
        pm.read_power_w()
        time.sleep(0.03)

    values_w = []
    for _ in range(samples):
        values_w.append(pm.read_power_w())
        time.sleep(0.08)

    values_uw = [value * 1e6 for value in values_w]
    power_w = float(sorted(values_w)[len(values_w) // 2])
    power_uw = power_w * 1e6
    log.append(
        Measurement(
            timestamp_epoch=time.time(),
            power_w=power_w,
            power_uw=power_uw,
            min_uw=float(min(values_uw)),
            max_uw=float(max(values_uw)),
            label=label,
        )
    )
    print(f"{label}: {power_uw:.5f} uW (mediana; min={min(values_uw):.5f}, max={max(values_uw):.5f})")
    return power_uw


def move_local(delta_az: float, delta_alt: float, settle_s: float) -> None:
    if delta_az == 0.0 and delta_alt == 0.0:
        return
    mount_control.move_axes_pid_2d(True, delta_az, delta_alt)
    time.sleep(settle_s)


def test_offset(
    pm: PM100Reader,
    samples: int,
    warmup_samples: int,
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
    power_uw = measure_pm(pm, samples, warmup_samples, label, measurements)
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
    warmup_samples: int,
    settle_s: float,
) -> tuple[list[Measurement], list[Trial], list[AcceptedMove]]:
    measurements: list[Measurement] = []
    trials: list[Trial] = []
    accepted_moves: list[AcceptedMove] = []

    measure_pm(pm, samples, warmup_samples, "inicial", measurements)

    for cycle in range(1, cycles + 1):
        print(f"\n=== Ciclo {cycle}/{cycles} ===")
        for step in steps:
            print(f"\nBusca local com passo {step:.5f} deg")
            baseline_before_uw = measure_pm(
                pm,
                samples,
                warmup_samples,
                f"baseline ciclo={cycle} step={step:.5f}",
                measurements,
            )
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
            best_uw = baseline_before_uw

            for offset_az, offset_alt in candidates[1:]:
                power_uw = test_offset(
                    pm,
                    samples,
                    warmup_samples,
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

            baseline_after_scan_uw = measure_pm(
                pm,
                samples,
                warmup_samples,
                f"baseline apos varredura ciclo={cycle} step={step:.5f}",
                measurements,
            )
            acceptance_reference_uw = baseline_after_scan_uw
            improvement_uw = best_uw - acceptance_reference_uw

            if best_az == 0.0 and best_alt == 0.0:
                print(
                    f"Sem candidato melhor que o baseline para passo {step:.5f}; "
                    f"baseline={acceptance_reference_uw:.5f} uW."
                )
                continue

            if improvement_uw < MIN_ACCEPT_IMPROVEMENT_UW:
                print(
                    f"Melhor candidato nao passou margem minima: "
                    f"{best_uw:.5f} contra baseline {acceptance_reference_uw:.5f} uW "
                    f"(ganho {improvement_uw:+.5f} uW). Mantendo posicao."
                )
                continue

            print(
                f"Melhor vizinho: dAz={best_az:+.5f} dAlt={best_alt:+.5f} "
                f"| baseline {acceptance_reference_uw:.5f} -> candidato {best_uw:.5f} uW"
            )
            move_local(best_az, best_alt, settle_s)
            confirmed_uw = measure_pm(
                pm,
                samples,
                warmup_samples,
                "apos aceitar melhor vizinho",
                measurements,
            )
            accepted_moves.append(
                AcceptedMove(
                    cycle=cycle,
                    step_deg=step,
                    delta_az_deg=best_az,
                    delta_alt_deg=best_alt,
                    baseline_before_uw=baseline_before_uw,
                    baseline_after_scan_uw=baseline_after_scan_uw,
                    candidate_uw=best_uw,
                    confirmed_uw=confirmed_uw,
                    timestamp_epoch=time.time(),
                )
            )

            for trial in reversed(trials):
                if (
                    trial.cycle == cycle
                    and trial.step_deg == step
                    and trial.delta_az_deg == best_az
                    and trial.delta_alt_deg == best_alt
                ):
                    trial.accepted = True
                    break

    return measurements, trials, accepted_moves


def save_camera_target_at_current_position(focus_mode: str, exposure_s: float) -> None:
    print("\nSalvando alvo da camera na posicao atual do melhor acoplamento...")
    foco_temp.set_focus_mode(focus_mode)
    foco_temp.connect_camera()
    try:
        frame = foco_temp.capture_frame(exposure_s, light=True)
        cm = foco_temp.centro_massa(frame)
        if cm is None:
            CAMERA_TARGET_DEBUG.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(CAMERA_TARGET_DEBUG), frame)
            print("Nao consegui detectar o laser na camera para salvar o alvo.")
            print(f"Frame salvo para diagnostico: {CAMERA_TARGET_DEBUG}")
            return

        x_cm, y_cm, intensidade, toca_borda = cm
        target_path = salvar_alvo(
            x_cm,
            y_cm,
            source="pm100_peak_receiver_local",
            frame_shape=frame.shape,
            focus_mode=focus_mode,
            samples=1,
            std_x_px=0.0,
            std_y_px=0.0,
        )

        marked = frame.copy()
        if marked.ndim == 2:
            marked = cv2.cvtColor(marked, cv2.COLOR_GRAY2BGR)
        cv2.circle(marked, (int(round(x_cm)), int(round(y_cm))), 10, (0, 255, 0), -1)
        CAMERA_TARGET_DEBUG.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(CAMERA_TARGET_DEBUG), marked)

        print(f"Alvo salvo em: {target_path}")
        print(
            f"Alvo camera: x={x_cm:.2f}px y={y_cm:.2f}px "
            f"| intensidade={intensidade:.1f} | toca_borda={toca_borda}"
        )
        print(f"Frame marcado salvo em: {CAMERA_TARGET_DEBUG}")
    finally:
        foco_temp.disconnect_camera()


def save_results(
    pm: PM100Reader,
    measurements: list[Measurement],
    trials: list[Trial],
    accepted_moves: list[AcceptedMove],
    args,
) -> None:
    payload = {
        "timestamp_epoch": time.time(),
        "pm100_resource": pm.resource_name,
        "pm100_idn": pm.idn,
        "wavelength_nm": args.wavelength_nm,
        "steps_deg": parse_steps(args.steps),
        "cycles": args.cycles,
        "samples": args.samples,
        "warmup_samples": args.warmup_samples,
        "settle_s": args.settle_s,
        "min_accept_improvement_uw": MIN_ACCEPT_IMPROVEMENT_UW,
        "measurements": [asdict(item) for item in measurements],
        "trials": [asdict(item) for item in trials],
        "accepted_moves": [asdict(item) for item in accepted_moves],
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
    parser.add_argument("--warmup-samples", type=int, default=DEFAULT_WARMUP_SAMPLES)
    parser.add_argument("--steps", default=DEFAULT_STEPS, help="Passos em graus. Ex: 0.02,0.01,0.005")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--focus-mode", default="dual", help="Modo para salvar alvo da camera: single ou dual.")
    parser.add_argument("--camera-exposure-s", type=float, default=DEFAULT_CAMERA_EXPOSURE_S)
    parser.add_argument(
        "--no-save-camera-target",
        action="store_true",
        help="Nao salva alvo da camera ao final/interrupcao.",
    )
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
    accepted_moves: list[AcceptedMove] = []
    try:
        measurements, trials, accepted_moves = optimize_local_receiver(
            pm=pm,
            steps=parse_steps(args.steps),
            cycles=args.cycles,
            samples=args.samples,
            warmup_samples=args.warmup_samples,
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
        if not args.no_save_camera_target:
            try:
                if accepted_moves:
                    save_camera_target_at_current_position(args.focus_mode, args.camera_exposure_s)
                else:
                    print("Nenhum movimento foi aceito; alvo da camera nao foi sobrescrito.")
            except Exception as exc:
                print(f"Aviso: nao consegui salvar alvo da camera: {exc}")
        save_results(pm, measurements, trials, accepted_moves, args)


if __name__ == "__main__":
    main()
