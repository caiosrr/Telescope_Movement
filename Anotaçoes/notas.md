<h1 align="center">Notas IC</h1>

## Ideia principal: autotune do tracker com dois telescopios

Data da anotacao: 2026-04-30.

O objetivo futuro e criar um autotune mais realista para o tracker. O laser que chega no telescopio principal vem de um segundo telescopio. A ideia e conectar os dois telescopios ao computador:

* Telescopio 1: sistema controlado pelo tracker. Ele usa a camera para manter o laser centralizado no sensor e, pelo prototipo mecanico atual, isso tambem deve manter o foco no encaixe da fibra.
* Telescopio 2: gerador de perturbacoes. O autotune deve mover esse telescopio para deslocar o feixe de entrada enquanto o telescopio 1 tenta acompanhar.

Essa abordagem deve testar rejeicao de perturbacao do sistema real, em vez de testar apenas uma perturbacao artificial aplicada no mesmo telescopio que esta corrigindo.

## Estrutura sugerida

Criar um novo arquivo, por exemplo:

`autotune_tracker_duplo_telescopio.py`

Esse arquivo deve:

* Rodar o tracker controlando apenas o telescopio 1.
* Mover apenas o telescopio 2 para criar perturbacoes padronizadas.
* Testar varios conjuntos de parametros do tracker.
* Medir o erro na camera durante cada ensaio.
* Gerar um ranking dos parametros.

## Parametros mais importantes para tunar

O tracker atual e mais um controle `PD + trim lento` do que um PID classico. Para o autotune, testar primeiro:

* `KP_AZ`
* `KP_ALT`
* `KD_AZ`
* `KD_ALT`
* `CMD_ACCEL_LIMIT`
* `MEASUREMENT_ALPHA`

O trim deve ser ajustado depois. Ele serve mais para erro persistente pequeno perto do centro, nao para perseguir perturbacao rapida.

## Ensaio padrao sugerido

Para cada conjunto de parametros:

1. Centralizar o laser com o tracker.
2. Esperar estabilizar dentro da tolerancia.
3. Aplicar perturbacoes pequenas no telescopio 2:
   * `az+`
   * `az-`
   * `alt+`
   * `alt-`
   * diagonais pequenas
4. Voltar o telescopio 2 para a posicao inicial apos cada perturbacao.
5. Repetir com rampas lentas, simulando o feixe andando continuamente.

Comecar com perturbacoes pequenas, idealmente gerando algo como `10-40 px` de deslocamento na camera. Depois aumentar se a malha estiver estavel.

## Metricas para ranquear

Nao escolher simplesmente o ganho mais rapido. Para acoplamento em fibra, estabilidade perto do centro e mais importante.

Metricas sugeridas:

* RMS do erro em pixels.
* Erro maximo em pixels.
* Tempo para voltar para dentro de `2 px`.
* Numero de brakes/runaway events.
* Tempo em saturacao de comando.
* Oscilacao perto do centro.
* Perda de sinal.
* Se o laser saiu do ROI da camera.

O melhor conjunto deve ser o que centraliza rapido sem ficar nervoso, sem movimento circular e sem depender de muitos brakes.

## Estado atual relevante

O `Tracker.py` ja foi ajustado para perguntar:

`Modo do laser (1=foco unico, 2=dupla reflexao)`

No modo `1`, ele usa as matrizes normais:

* `A_inv_fine.npy`
* `A_inv_coarse.npy`

No modo `2`, ele usa as matrizes temporarias da calibracao com dois focos:

* `foco_temp_A_inv_fine.npy`
* `foco_temp_A_inv_coarse.npy`

O tracker sempre usa o mount real; a pergunta de simulador foi removida.

Tambem foi adicionado um freio para movimento manual brusco: se o spot salta muito entre frames, o controle zera por um instante antes de tentar recentralizar.

## Observacoes sobre desempenho

Na ultima medicao do tracker:

* Camera ficou por volta de `10-13 Hz`.
* `cap` ficou perto de `70-80 ms`.
* `CM` ficou perto de `0.2 ms`.
* `UI` ficou perto de `13 ms`.

Conclusao: o gargalo principal e captura/transferencia da camera via Alpaca, nao o calculo do centro de massa.

Foi testado reduzir `WINDOW_SIZE` de `200` para `160`, o que melhorou a taxa, mas a preferencia atual e manter `200 px` por dar mais margem quando o laser se move. Se necessario em testes futuros, reduzir o ROI pode ser uma opcao.

## Ideia importante para o futuro

O autotune de dois telescopios deve ser tratado como um teste de rejeicao de perturbacao do experimento completo:

`telescopio 2 move o feixe -> telescopio 1 corrige com o tracker -> camera mede erro residual`

Isso deve produzir parametros mais uteis para acoplamento na fibra do que o autotune antigo.

## Controle futuro usando potencia da fibra

Data da anotacao: 2026-07-08.

A potencia medida no powermeter pode ser usada para otimizar e manter o acoplamento na fibra, mas nao funciona como um PID direto simples.

No tracker da camera, o erro tem direcao:

`erro_px = posicao_alvo - posicao_laser`

Esse erro diz para qual lado mover o mount. Se o laser esta a direita do alvo, o controle sabe que precisa mover no sentido oposto.

No powermeter, a potencia e uma medida escalar:

`erro_potencia = potencia_alvo - potencia_medida`

Esse valor diz que o acoplamento esta ruim ou bom, mas nao diz se o melhor movimento e `+Az`, `-Az`, `+Alt`, `-Alt` ou uma diagonal. Portanto, um PID direto usando apenas `potencia_alvo - potencia_medida` ficaria cego para a direcao.

### Estrategia mais correta

Usar a potencia para estimar a inclinacao local da superficie de acoplamento:

1. Medir a potencia atual `P0`.
2. Testar um pequeno movimento `+Az` e medir `P(+Az)`.
3. Testar `-Az` e medir `P(-Az)`.
4. Estimar `dP/dAz`.
5. Repetir para `+Alt` e `-Alt`, estimando `dP/dAlt`.
6. Mover na direcao em que a potencia aumenta.

Isso e mais parecido com:

* hill climbing;
* gradient ascent;
* extremum seeking control;
* lock-in com dither.

### Versao discreta atual

O script `otimizacao/otimizar_receptor_local_pm100.py` faz uma busca local discreta:

* mede a potencia atual;
* testa vizinhos em Az/Alt;
* volta para a posicao inicial de cada teste;
* escolhe o vizinho de maior potencia;
* aceita esse movimento;
* repete com passos menores.

Essa versao e lenta, mas segura e reversivel para bancada.

### Possivel versao futura continua

Criar uma malha de dither:

* aplicar uma pequena oscilacao em Az e/ou Alt;
* medir se a potencia oscila em fase ou contra-fase com o dither;
* usar isso para descobrir o sinal do gradiente;
* mover lentamente o mount no sentido que aumenta a potencia;
* reduzir o passo perto do pico.

Essa abordagem poderia manter o acoplamento no pico mesmo se o feixe derivar lentamente.

### Estrategia com dois telescopios

Quando os dois mounts estiverem disponiveis:

* receptor: ajuste fino e rapido, usando camera/tracker e powermeter;
* emissor: ajuste grosso/lento, procurando colocar o feixe dentro da regiao de captura do receptor;
* depois que o receptor achar o pico de potencia, salvar a posicao do spot na camera como novo alvo de alinhamento;
* o tracker deve manter o spot nesse alvo salvo, nao necessariamente no centro geometrico da camera.

Uma rotina promissora:

1. Usar camera/tracker para manter o spot no alvo salvo.
2. Fazer busca por potencia no receptor.
3. Salvar o pico encontrado como `alvo_alinhamento_camera.json`.
4. Usar o emissor para melhorar a potencia global.
5. Refazer ajuste fino no receptor.
6. Repetir ate a melhora ficar pequena.
