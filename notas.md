# Notas de impressao - encaixe/rosca para telescopio

## Ajustes no Fusion

- Modelo baseado no adaptador Thorlabs SM1FC.
- Correcao importante de interpretacao:
  - A peca tem uma rosca SM1 externa grande no diametro externo do disco: 1.035"-40, diametro nominal aproximado de 26,29 mm e passo de 0,635 mm.
  - A rosca do cilindro central, onde entra o conector FC, nao e SM1; pelo desenho ela e M8 x 0,75 external thread.
  - O furo interno desse cilindro central e indicado no desenho como 0,24 in / 6,1 mm bore, para aceitar conectores FC padrao.
  - A medida feita no Fusion de raio 3,035 mm / diametro 6,071 mm bate com esse bore de 6,1 mm.
- A rosca central importada veio com faces em V, sem crista cilindrica/plana clara.
- Offset via Pressionar/Puxar nas faces da rosca:
  - Faces principais dos dois sentidos: -0,05 mm.
  - Face inferior mais perto da base, que dava erro: -0,02 mm.
- Chanfro na entrada da rosca nao foi aplicado porque as arestas estavam muito segmentadas/finas.
- Ja existe uma parte circular/lisa na base depois da ultima volta da rosca, entao nao foi feito alivio adicional na base.

## Configuracoes no Bambu Studio

- Orientacao: rosca para cima, base plana na mesa.
- Altura da camada: 0,08 mm.
- Altura inicial da camada: 0,20 mm.
- Largura de linha:
  - Padrao: 0,42 mm.
  - Camada inicial: 0,50 mm.
  - Parede externa: 0,42 mm.
  - Parede interna: 0,45 mm.
- Densidade de preenchimento esparsa: 20%.
- Preenchimento/sobreposicao de parede: 15%.
- Paredes / wall loops: 4.
- Camadas inferiores: 5.
- Camadas de casca superior: 5.
- Suporte: interno, arvore slim.
- Velocidade ajustada para rosca:
  - Parede externa: cerca de 40-50 mm/s recomendado.
  - Parede interna: cerca de 80-100 mm/s recomendado.

## O que observar depois da impressao

- Se o suporte interno saiu sem marcar ou arrancar a parede interna.
- Se nao ficou rebarba dentro do furo.
- Se a rosca externa ficou limpa.
- Se o encaixe comeca a rosquear sem forcar.
- Se entrar muito apertado: aumentar o offset da rosca no Fusion.
- Se entrar folgado: reduzir o offset da rosca no Fusion.
- Se nao comecar a rosquear: criar uma entrada/chanfro manual ou ajustar a primeira volta.
- Se travar so no final: revisar a regiao da base/alivio da rosca.

## Resultado da primeira impressao

- A rosca imprimiu e chegou a enroscar, mas ficou firme/apertada demais.
- Antes da rosca terminar de entrar, o furo onde entra a parte do nucleo/conector da fibra estava muito fino.
- O furo nao entrou inicialmente; foi necessario forcar com uma chave para abrir.
- Alem de fino, o encaixe interno pareceu raso: a parte da fibra bateu antes de entrar tudo.
- Como o furo interno travou/limitou a entrada, nao deu para avaliar a rosca ate o final.
- A parte da rosca que chegou a enroscar ja pareceu muito justa.

## Proxima versao sugerida

- Corrigir primeiro o furo interno, porque ele pode estar impedindo a peca de rosquear ate o fim.
- Aumentar o furo interno:
  - Sugestao inicial: +0,20 mm radial.
  - Isso equivale a aproximadamente +0,40 mm no diametro.
  - O diametro de 2,5 mm visto no Fusion provavelmente corresponde a ferrule do conector FC/PC, nao ao nucleo da fibra.
  - Para a proxima versao, usar furo da ferrule com diametro de 2,9 mm.
- Aumentar a profundidade do furo interno:
  - Sugestao: +1 a +2 mm.
  - Se possivel, deixar passante ou com folga suficiente no fundo.
- Evitar suporte dentro do furo critico, se a geometria permitir.
- Se precisar limpar o furo depois da impressao, preferir broca do diametro correto girada a mao em vez de chave.
- Aumentar a folga da rosca depois de corrigir o furo:
  - Faces principais dos dois sentidos: tentar -0,07 mm ou -0,08 mm.
  - Face inferior perto da base: tentar -0,03 mm, se o Fusion aceitar.
