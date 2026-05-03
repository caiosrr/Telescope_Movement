import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from artifact_paths import display_path, json_output_path, matrix_output_path
from Center_of_Mass import capture_frame, centro_massa, connect_camera, disconnect_camera
from PID_controll import ensure_connected, ensure_not_tracking, ensure_unparked
from mov_simultaneo import move_axes_pid_2d

PASSOS_TENTATIVA = [0.06, 0.04, 0.02, 0.01]
PADRAO_OFFSETS = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (2.0, 0.0),
    (0.0, 2.0),
    (-2.0, 0.0),
    (0.0, -2.0),
    (3.0, 0.0),
    (0.0, 3.0),
    (-3.0, 0.0),
    (0.0, -3.0),
    (3.0, 3.0),
    (0.0, 0.0),
]

EXPOSURE_SECONDS = 32e-6
SETTLE_S = 0.50
MIN_PONTOS = 12
MIN_SPREAD_DEG = 0.02
MAX_COND = 1.0e4
MAX_RMS_PX = 15.0
WARN_RMS_PX = 8.0


@dataclass
class RegistroCalibracao:
    step_deg: float
    offset_az_deg: float
    offset_alt_deg: float
    x_px: float
    y_px: float


def _padrao_em_graus(step_deg: float):
    return [(step_deg * x, step_deg * y) for x, y in PADRAO_OFFSETS]


def _capturar_cm(exposure: float):
    frame = capture_frame(exposure, light=True)
    cm = centro_massa(frame)
    if cm is None:
        return None
    x_cm, y_cm, _, toca_borda = cm
    return float(x_cm), float(y_cm), bool(toca_borda)


def _voltar_para_centro(mount: bool, az_offset: float, alt_offset: float):
    if abs(az_offset) > 1e-4 or abs(alt_offset) > 1e-4:
        print(
            f"Retornando ao centro (dAz={-az_offset:+.4f} deg, dAlt={-alt_offset:+.4f} deg)..."
        )
        move_axes_pid_2d(mount, -az_offset, -alt_offset)
        time.sleep(SETTLE_S)


def _avaliar_jacobiana(registros):
    if len(registros) < MIN_PONTOS:
        raise RuntimeError(
            f"Poucos pontos para ajuste: {len(registros)}. Minimo recomendado: {MIN_PONTOS}."
        )

    az = np.array([r.offset_az_deg for r in registros], dtype=float)
    alt = np.array([r.offset_alt_deg for r in registros], dtype=float)
    x = np.array([r.x_px for r in registros], dtype=float)
    y = np.array([r.y_px for r in registros], dtype=float)

    spread_az = float(np.ptp(az))
    spread_alt = float(np.ptp(alt))
    if spread_az < MIN_SPREAD_DEG or spread_alt < MIN_SPREAD_DEG:
        raise RuntimeError(
            "Offsets com pouca variacao. Refaça a calibracao com um padrao mais aberto."
        )

    M = np.column_stack((az, alt, np.ones(len(registros))))
    if np.linalg.matrix_rank(M) < 3:
        raise RuntimeError("Ajuste degenerado: os pontos nao excitam bem os dois eixos.")

    coef_x, _, _, _ = np.linalg.lstsq(M, x, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(M, y, rcond=None)

    A = np.array(
        [
            [coef_x[0], coef_x[1]],
            [coef_y[0], coef_y[1]],
        ],
        dtype=float,
    )

    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > MAX_COND:
        raise RuntimeError(
            f"Matriz mal condicionada (cond={cond:.2e}). Melhor repetir a calibracao."
        )

    A_inv = np.linalg.inv(A)

    pred_x = M @ coef_x
    pred_y = M @ coef_y
    residuo = np.sqrt((pred_x - x) ** 2 + (pred_y - y) ** 2)
    rms_px = float(np.sqrt(np.mean(residuo ** 2)))
    max_px = float(np.max(residuo))

    return {
        "A": A,
        "A_inv": A_inv,
        "coef_x": coef_x,
        "coef_y": coef_y,
        "spread_az": spread_az,
        "spread_alt": spread_alt,
        "cond": cond,
        "rms_px": rms_px,
        "max_px": max_px,
        "num_points": len(registros),
    }


def _emitir_aviso_rms(resultado):
    rms_px = resultado["rms_px"]
    if rms_px > WARN_RMS_PX:
        print(
            f"Aviso: residual RMS relativamente alto ({rms_px:.2f} px), "
            "mas ainda dentro do limite aceito para a bancada."
        )


def _offset_max_abs(registro: RegistroCalibracao) -> float:
    return max(abs(registro.offset_az_deg), abs(registro.offset_alt_deg))


def _marcar_resultado_base(
    resultado,
    selection_mode,
    selected_max_offset_deg,
    selected_max_step_deg=None,
):
    resultado = dict(resultado)
    resultado["selection_mode"] = selection_mode
    resultado["selected_max_offset_deg"] = selected_max_offset_deg
    resultado["selected_max_step_deg"] = selected_max_step_deg
    return resultado


def _chave_resultado(resultado):
    offset = resultado["selected_max_offset_deg"]
    offset = float("inf") if offset is None else float(offset)
    max_step = resultado.get("selected_max_step_deg")
    max_step = float("inf") if max_step is None else float(max_step)
    penalidade_subset = 1 if resultado["selection_mode"] == "central_subset" else 0
    return (
        float(resultado["rms_px"]),
        -int(resultado["num_points"]),
        penalidade_subset,
        offset,
        max_step,
    )


def _resultado_aprovado(resultado):
    return resultado["rms_px"] <= MAX_RMS_PX


def _mensagem_qualidade_ruim(resultado, total_pontos):
    detalhe = (
        f"RMS={resultado['rms_px']:.2f}px, "
        f"cond={resultado['cond']:.2e}, "
        f"pontos usados={resultado['num_points']}/{total_pontos}"
    )
    if resultado["selection_mode"] == "central_subset":
        detalhe += (
            f", |offset| <= {resultado['selected_max_offset_deg']:.4f} deg"
        )
    if resultado["selection_mode"] == "fine_steps_subset":
        detalhe += (
            f", step <= {resultado['selected_max_step_deg']:.4f} deg"
        )
    return (
        "Calibracao salva em modo de melhor esforco. "
        "Use com cautela: a matriz pode funcionar para recentrar o spot, "
        "mas ha risco de erro maior fora da regiao calibrada. "
        + detalhe
    )


def _ajustar_jacobiana(registros, permitir_melhor_esforco=False):
    candidatos_validos = []
    erro_total = None

    try:
        resultado_total = _marcar_resultado_base(
            _avaliar_jacobiana(registros),
            selection_mode="all_points",
            selected_max_offset_deg=None,
            selected_max_step_deg=None,
        )
        candidatos_validos.append(resultado_total)
    except RuntimeError as exc:
        resultado_total = None
        erro_total = str(exc)

    if resultado_total is not None and _resultado_aprovado(resultado_total):
        resultado_total["quality_ok"] = True
        resultado_total["quality_status"] = "accepted"
        resultado_total["quality_warning"] = None
        _emitir_aviso_rms(resultado_total)
        return resultado_total

    raios_candidatos = sorted(
        {
            round(_offset_max_abs(registro), 10)
            for registro in registros
            if _offset_max_abs(registro) > 0.0
        }
    )

    for raio in raios_candidatos:
        subconjunto = [
            registro
            for registro in registros
            if _offset_max_abs(registro) <= raio + 1e-9
        ]
        if len(subconjunto) < MIN_PONTOS or len(subconjunto) >= len(registros):
            continue

        try:
            resultado = _marcar_resultado_base(
                _avaliar_jacobiana(subconjunto),
                selection_mode="central_subset",
                selected_max_offset_deg=raio,
                selected_max_step_deg=None,
            )
        except RuntimeError:
            continue

        candidatos_validos.append(resultado)

    candidatos_aprovados = [
        resultado for resultado in candidatos_validos if _resultado_aprovado(resultado)
    ]
    melhor_subconjunto = None
    subconjuntos_aprovados = [
        resultado
        for resultado in candidatos_aprovados
        if resultado["selection_mode"] == "central_subset"
    ]
    if subconjuntos_aprovados:
        melhor_subconjunto = min(subconjuntos_aprovados, key=_chave_resultado)

    if melhor_subconjunto is not None:
        melhor_subconjunto["quality_ok"] = True
        melhor_subconjunto["quality_status"] = "accepted"
        melhor_subconjunto["quality_warning"] = None
        mensagem_rms_anterior = (
            f"{resultado_total['rms_px']:.2f} px"
            if resultado_total is not None
            else "um valor nao aceito no ajuste completo"
        )
        print(
            "Ajuste aceito com subconjunto central: "
            f"{melhor_subconjunto['num_points']} de {len(registros)} pontos "
            f"(|offset| <= {melhor_subconjunto['selected_max_offset_deg']:.4f} deg). "
            f"RMS caiu de {mensagem_rms_anterior} para "
            f"{melhor_subconjunto['rms_px']:.2f} px."
        )
        _emitir_aviso_rms(melhor_subconjunto)
        return melhor_subconjunto

    passos_candidatos = sorted({registro.step_deg for registro in registros}, reverse=True)
    for step_max in passos_candidatos[1:]:
        subconjunto = [
            registro for registro in registros if registro.step_deg <= step_max + 1e-12
        ]
        if len(subconjunto) < MIN_PONTOS:
            continue

        try:
            resultado = _marcar_resultado_base(
                _avaliar_jacobiana(subconjunto),
                selection_mode="fine_steps_subset",
                selected_max_offset_deg=None,
                selected_max_step_deg=step_max,
            )
        except RuntimeError:
            continue

        candidatos_validos.append(resultado)

        if _resultado_aprovado(resultado):
            resultado["quality_ok"] = True
            resultado["quality_status"] = "accepted"
            resultado["quality_warning"] = None
            mensagem_rms_anterior = (
                f"{resultado_total['rms_px']:.2f} px"
                if resultado_total is not None
                else "um valor nao aceito no ajuste completo"
            )
            print(
                "Ajuste aceito usando apenas os passos mais finos: "
                f"{resultado['num_points']} de {len(registros)} pontos "
                f"(step <= {resultado['selected_max_step_deg']:.4f} deg). "
                f"RMS caiu de {mensagem_rms_anterior} para "
                f"{resultado['rms_px']:.2f} px."
            )
            _emitir_aviso_rms(resultado)
            return resultado

    melhor_disponivel = min(candidatos_validos, key=_chave_resultado) if candidatos_validos else None
    detalhe_subconjunto = ""
    if melhor_disponivel is not None and melhor_disponivel["selection_mode"] == "central_subset":
        detalhe_subconjunto = (
            " Melhor subconjunto central: "
            f"{melhor_disponivel['num_points']} pontos ate "
            f"|offset| <= {melhor_disponivel['selected_max_offset_deg']:.4f} deg "
            f"com RMS {melhor_disponivel['rms_px']:.2f} px."
        )
    elif melhor_disponivel is not None and melhor_disponivel["selection_mode"] == "fine_steps_subset":
        detalhe_subconjunto = (
            " Melhor ajuste valido encontrado usando apenas os passos mais finos: "
            f"{melhor_disponivel['num_points']} pontos com "
            f"step <= {melhor_disponivel['selected_max_step_deg']:.4f} deg "
            f"e RMS {melhor_disponivel['rms_px']:.2f} px."
        )
    elif melhor_disponivel is not None:
        detalhe_subconjunto = (
            " Melhor ajuste valido encontrado com todos os pontos, "
            f"mas RMS ainda alto ({melhor_disponivel['rms_px']:.2f} px)."
        )

    if permitir_melhor_esforco and melhor_disponivel is not None:
        melhor_disponivel["quality_ok"] = False
        melhor_disponivel["quality_status"] = "best_effort"
        melhor_disponivel["quality_warning"] = _mensagem_qualidade_ruim(
            melhor_disponivel,
            len(registros),
        )
        print(f"Aviso: {melhor_disponivel['quality_warning']}")
        return melhor_disponivel

    mensagem_base = (
        f"Residuo RMS alto ({resultado_total['rms_px']:.2f} px). "
        if resultado_total is not None
        else (
            f"Nao foi possivel aceitar o ajuste completo: {erro_total}. "
            if erro_total is not None
            else "Nao foi possivel aceitar o ajuste completo. "
        )
    )

    raise RuntimeError(
        mensagem_base
        + "Pode haver drift, borda, nao linearidade fora do centro ou erro de medicao."
        + detalhe_subconjunto
    )


def _salvar_resultados(resultado, registros, output_dir="."):
    matrix_a_path = matrix_output_path("calibracao_A.npy", output_dir)
    matrix_a_inv_path = matrix_output_path("calibracao_A_inv.npy", output_dir)

    np.save(matrix_a_path, resultado["A"])
    np.save(matrix_a_inv_path, resultado["A_inv"])

    meta = {
        "timestamp_epoch": time.time(),
        "num_points": resultado["num_points"],
        "total_points_captured": resultado.get("total_points_captured", len(registros)),
        "spread_az_deg": resultado["spread_az"],
        "spread_alt_deg": resultado["spread_alt"],
        "condition_number": resultado["cond"],
        "rms_residual_px": resultado["rms_px"],
        "max_residual_px": resultado["max_px"],
        "selection_mode": resultado.get("selection_mode", "all_points"),
        "selected_max_offset_deg": resultado.get("selected_max_offset_deg"),
        "selected_max_step_deg": resultado.get("selected_max_step_deg"),
        "quality_ok": resultado.get("quality_ok", True),
        "quality_status": resultado.get("quality_status", "accepted"),
        "quality_warning": resultado.get("quality_warning"),
        "A": resultado["A"].tolist(),
        "A_inv": resultado["A_inv"].tolist(),
        "records": [asdict(r) for r in registros],
    }

    meta_output_path = json_output_path("calibracao_meta.json", output_dir)
    with meta_output_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def calibracao_2d_simultanea_v2(exposure: float, mount: bool):
    print("\n>>> Iniciando calibracao 2D simultanea V2 <<<")
    print("A V2 valida condicionamento, residual e salva metadata junto da matriz.")

    cm_init = _capturar_cm(exposure)
    if cm_init is None:
        raise RuntimeError("Laser nao detectado no inicio da calibracao.")

    registros = []
    total_points_captured = 0

    for step_idx, step in enumerate(PASSOS_TENTATIVA):
        print(f"\n{'=' * 50}")
        print(f"Varredura com passo de {step:.4f} deg")
        print(f"{'=' * 50}")

        az_offset = 0.0
        alt_offset = 0.0
        registros_step = []
        limite_atingido = False

        for idx, (alvo_az, alvo_alt) in enumerate(_padrao_em_graus(step)):
            delta_az = alvo_az - az_offset
            delta_alt = alvo_alt - alt_offset

            if abs(delta_az) > 1e-4 or abs(delta_alt) > 1e-4:
                print(
                    f"[{idx:02d}] Offset alvo ({alvo_az:+.4f}, {alvo_alt:+.4f}) deg "
                    f"| movendo dAz={delta_az:+.4f}, dAlt={delta_alt:+.4f}"
                )
                move_axes_pid_2d(mount, delta_az, delta_alt)
                time.sleep(SETTLE_S)
                az_offset = alvo_az
                alt_offset = alvo_alt
            else:
                print(f"[{idx:02d}] Offset alvo ({alvo_az:+.4f}, {alvo_alt:+.4f}) deg | medindo")

            cm = _capturar_cm(exposure)
            if cm is None:
                print("  -> sinal sumiu; encerrando este passo.")
                limite_atingido = True
                break

            x_cm, y_cm, toca_borda = cm
            if toca_borda:
                print("  -> laser tocou a borda; encerrando este passo.")
                limite_atingido = True
                break

            registros_step.append(
                RegistroCalibracao(
                    step_deg=step,
                    offset_az_deg=az_offset,
                    offset_alt_deg=alt_offset,
                    x_px=x_cm,
                    y_px=y_cm,
                )
            )
            print(f"  -> CM valido: x={x_cm:.2f} px, y={y_cm:.2f} px")

        total_points_captured += len(registros_step)
        registros.extend(registros_step)
        _voltar_para_centro(mount, az_offset, alt_offset)

        print(f"Pontos acumulados: {len(registros)}")

        if limite_atingido:
            print(
                "Step encerrado antes do fim; pontos validos foram preservados e "
                "a calibracao seguira para um passo menor."
            )
            continue

        if len(registros) >= MIN_PONTOS:
            try:
                resultado = _ajustar_jacobiana(registros)
                resultado["total_points_captured"] = total_points_captured
                break
            except Exception as exc:
                print(f"Ajuste ainda nao aceito: {exc}")
                has_next_step = step_idx < (len(PASSOS_TENTATIVA) - 1)
                if has_next_step and "RMS" in str(exc):
                    print(
                        "RMS alto neste lote; descartando os pontos acumulados "
                        "antes de tentar o proximo passo menor."
                    )
                    registros = []
                continue
    else:
        resultado = _ajustar_jacobiana(registros, permitir_melhor_esforco=True)
        resultado["total_points_captured"] = total_points_captured

    return resultado, registros


def main():
    ensure_connected()
    ensure_unparked()
    ensure_not_tracking()

    mount_input = input("Usar montagem real? (1 para sim, 0 para simulador): ").strip()
    mount = bool(int(mount_input)) if mount_input in {"0", "1"} else False

    try:
        connect_camera()

        resultado, registros = calibracao_2d_simultanea_v2(
            exposure=EXPOSURE_SECONDS,
            mount=mount,
        )

        _salvar_resultados(resultado, registros)

        if resultado.get("quality_ok", True):
            print("\nCalibracao concluida com sucesso.")
        else:
            print("\nCalibracao concluida com ressalvas.")
        print("Jacobiana A (px/deg):")
        print(resultado["A"])
        print("\nMatriz inversa A_inv (deg/px):")
        print(resultado["A_inv"])
        print(
            f"\nMetricas: pontos={resultado['num_points']}, "
            f"capturados={resultado.get('total_points_captured', len(registros))}, "
            f"cond={resultado['cond']:.2e}, "
            f"RMS={resultado['rms_px']:.2f}px, "
            f"max={resultado['max_px']:.2f}px"
        )
        if resultado.get("selection_mode") == "central_subset":
            print(
                "Ajuste final baseado em subconjunto central ate "
                f"|offset| <= {resultado['selected_max_offset_deg']:.4f} deg."
            )
        elif resultado.get("selection_mode") == "fine_steps_subset":
            print(
                "Ajuste final baseado apenas nos passos mais finos ate "
                f"step <= {resultado['selected_max_step_deg']:.4f} deg."
            )
        if resultado.get("quality_warning"):
            print(f"Aviso final: {resultado['quality_warning']}")
        print(
            "Arquivos salvos: "
            f"{display_path(matrix_output_path('calibracao_A.npy'))}, "
            f"{display_path(matrix_output_path('calibracao_A_inv.npy'))}, "
            f"{display_path(json_output_path('calibracao_meta.json'))}"
        )

    except KeyboardInterrupt:
        print("\nCalibracao interrompida pelo usuario.")
    except Exception as exc:
        print(f"\nErro na calibracao V2: {exc}")
    finally:
        disconnect_camera()


if __name__ == "__main__":
    main()
