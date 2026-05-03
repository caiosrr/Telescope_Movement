# Telescope Movement

Programas de controle, calibracao e tracking para os testes de IC com telescopio, camera Alpaca e apontamento 2D.

## Arquivos principais

- `Center_of_Mass.py`: captura da camera e centralizacao por centro de massa.
- `Calibracao_ang-pix_dual_v3.py`: calibracao angular-pixel em regimes coarse/fine.
- `Tracker.py`: tracker continuo para manter o laser centralizado.
- `mov_simultaneo.py`: movimentos simultaneos dos eixos.
- `PID_controll.py`: conexao e comandos basicos do mount.
- `temporarios/`: versoes temporarias para o caso de dupla reflexao/dois focos.
- `resultados/matrizes/`: matrizes atuais usadas pelos scripts.
- `Anotaçoes/notas.md`: notas de continuidade do experimento.

## Observacoes

Arquivos de auditoria, imagens, videos e JSONs de resultado ficam ignorados pelo git para evitar commits muito grandes. As matrizes `.npy` pequenas ficam versionadas porque sao uteis para repetir testes com o tracker.
