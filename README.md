# Free-Space-QKD

Controle, calibracao e tracking para testes de apontamento com telescopios, camera Alpaca/ASCOM e power meter.

## Estrutura

- `controle/`: codigo principal de conexao, movimento, tracking, agente remoto e utilitarios compartilhados.
- `controle/mount_control.py`: movimento principal do mount local; `mov_simultaneo.py` e apenas alias de compatibilidade.
- `calibracoes/Calibracao_ang-pix_dual_v3.py`: calibracao angular-pixel principal para um foco.
- `foco_multiplos/`: fluxo para imagens com dois ou mais focos/reflexoes; tambem funciona para foco unico e pode virar o padrao.
- `calibracoes/autotune/`: autotunes, validacoes e buscas de parametros.
- `calibracoes/legado/`: versoes antigas ou experimentais mantidas como referencia.
- `otimizacao/`: scripts que usam power meter/camera como metrica para maximizar acoplamento.
- `ferramentas/`: scripts de bancada/diagnostico, como definir alvo da fibra e diagnosticar mounts.
- `resultados/matrizes/`: matrizes usadas pelos scripts.
- `resultados/json/`: resultados de execucao e logs em JSON.
- `artifact_paths.py`: helper central para salvar/carregar artefatos em `resultados/` sem espalhar caminhos fixos pelos scripts.
- `Anotaçoes/` e `notas.md`: notas de continuidade do experimento.

## Comandos Uteis

```powershell
python .\foco_multiplos\Center_of_Mass_foco_temp.py
python .\foco_multiplos\Calibracao_ang-pix_dual_v3_foco_temp.py
python .\controle\Tracker.py
python .\controle\mount_control.py
python .\controle\mov_mount_remoto.py
python .\ferramentas\definir_alvo_fibra.py
python .\ferramentas\diagnostico_mounts.py
python .\calibracoes\autotune\autotune_mov_mount_remoto.py
python .\otimizacao\otimizar_acoplamento_pm100.py
```

## Observacoes

Arquivos de auditoria, imagens, videos e JSONs de resultado ficam ignorados pelo git para evitar commits muito grandes. As matrizes `.npy` pequenas ficam versionadas porque sao uteis para repetir testes com o tracker.
