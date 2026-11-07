# MimePlant — código da dissertação

Código e protocolo experimental da dissertação *MimePlant: Uma Abordagem de Aprendizado por Reforço Automatizado com
Otimização de Hiperparâmetros para o FarmGym* (PPCC/UFOP).

O **MimePlant** é um agente de manejo agrícola simulado construído em três camadas:

1. **Imitação** — clona um professor heurístico de regras agronômicas por clonagem comportamental (BC) + DAgger.
2. **Reforço** — refina o clone com PPO recorrente (LSTM), otimizando a recompensa do ambiente.
3. **Otimização de hiperparâmetros (HPO)** — escolhe automaticamente os hiperparâmetros da fase de reforço.

O estudo compara **cinco otimizadores** sob orçamento idêntico, com **três execuções independentes cada**: TPE + Hyperband,
busca aleatória com a mesma poda, DEHB, otimização bayesiana com processo gaussiano (GP-BO) e amostragem de Sobol (QMC).
O reforço treinado do zero também recebe HPO (TPE + Hyperband, busca aleatória e GP-BO, uma busca de cada), para fechar o
desenho com e sem imitação, com e sem HPO.

Os agentes (BC, DAgger e PPO recorrente) e o DEHB são implementação própria (PyTorch e NumPy); TPE, busca aleatória, GP-BO,
Sobol e a poda Hyperband vêm do Optuna. O ambiente (`environment/`, FarmGym v4) é uma reimplementação sobre Gymnasium, em
NumPy: grade 4 × 4, 365 passos por episódio, observação de 176 variáveis, 14 ações com máscara.

Este arquivo explica **como executar os experimentos do início ao fim**. Sumário:

1. [Resultados em uma tela](#1-resultados-em-uma-tela)
2. [Estrutura do repositório](#2-estrutura-do-repositório)
3. [Requisitos e instalação](#3-requisitos-e-instalação)
4. [Onde as saídas são gravadas](#4-onde-as-saídas-são-gravadas)
5. [Passo a passo](#5-passo-a-passo)
6. [Executar à mão, sem os encadeadores](#6-executar-à-mão-sem-os-encadeadores)
7. [Acompanhar, interromper e retomar](#7-acompanhar-interromper-e-retomar)
8. [Testes e fumaça](#8-testes-e-fumaça)
9. [Protocolo experimental](#9-protocolo-experimental)
10. [Problemas conhecidos](#10-problemas-conhecidos)

---

## 1. Resultados em uma tela

Recompensa bruta (soma das recompensas do episódio) no conjunto de teste (sementes 999–1048, 50 episódios por regime),
média dos três regimes climáticos. Agentes treinados e clone: média interquartil (IQM) dos 250 episódios das 5 sementes de
treino; professor e agente aleatório: IQM dos 50 episódios.

| Agente | Recompensa |
|---|---|
| **MimePlant** (imitação + reforço + HPO) | **4.606** |
| Professor heurístico | 4.602 |
| Reforço do zero + HPO (GP-BO) | 4.503 |
| Imitação + reforço, configuração manual | 4.445 |
| Reforço do zero, configuração manual | 4.398 |
| Só imitação (clone) | 4.259 |
| Agente aleatório | 709 |

O **MimePlant** é o agente da busca aleatória, execução *a* (`rs_r3_a_trial54`): a configuração de maior nota de validação
(escala normalizada, 0 = aleatório e 1 = professor; `hpo/reavaliar_topk.py`, função `escolher_melhor_candidato`) entre os
seis incumbentes de TPE e de busca aleatória, que formam o conjunto de candidatos definido antes das demais buscas. A
mesma regra o escolheria entre os 15 incumbentes. A melhor execução de uma busca em teste é outra (GP-BO, 4.633, na tabela
abaixo); ela não é adotada porque nenhuma escolha usa o teste.

| Otimizador no IL + RL (3 execuções) | Média | Melhor | Pior | Desvio |
|---|---|---|---|---|
| Busca aleatória | 4.557 | 4.606 | 4.515 | 46 |
| Sobol | 4.543 | 4.561 | 4.511 | 27 |
| TPE | 4.532 | 4.590 | 4.445 | 77 |
| GP-BO | 4.527 | 4.633 | 4.373 | 137 |
| DEHB | 4.521 | 4.550 | 4.469 | 45 |

TPE, busca aleatória, GP-BO e Sobol usam a mesma poda Hyperband (Optuna); o DEHB aplica a sua própria promoção entre as
mesmas fidelidades. Os cinco otimizadores não se distinguem estatisticamente (ANOVA: F = 0,105; p = 0,978). As médias
diferem em até 36 pontos, e as 15 execuções se espalham por 260. O desvio entre as três execuções de cada otimizador varia de
27 (Sobol) a 137 (GP-BO), mas, com três execuções, essa diferença de dispersão também não é estatisticamente estabelecida.

| Otimizador no reforço do zero (1 busca) | Recompensa | Ganho sobre o reforço do zero manual |
|---|---|---|
| GP-BO | 4.503 | +105 |
| TPE | 4.465 | +67 |
| Busca aleatória | 4.308 | −90 |

No reforço do zero, o GP-BO é o único com ganho consistente: vence o reforço manual nas 5 sementes de treino (p = 0,044).
Foi o escolhido como "reforço do zero com HPO" por ter a maior nota de validação entre os três. O MimePlant (4.606) fica
103 pontos acima dele. O GP-BO entrou neste controle depois de conhecido o resultado dos otimizadores no IL + RL, em que
teve a melhor execução em teste.

**Por que o professor piora no regime seco** (`analise/diag_regime_seco.py`, validação 500–549; `Resultados/diag_regime_seco/`).
A recompensa do professor cai de 5.442 (normal) para 2.943 (seco), e 98% dessa queda vem do componente de colheita (−2.458 de
−2.499). No seco, o professor colhe 10,5 vezes por episódio contra 23,2 no normal, apesar de regar 109 vezes contra 33: a
fração dos dias com planta e umidade do solo abaixo do limiar de rega (0,58) sobe de 0,18 para 0,50, a fração de dias com chuva cai de 0,35
para 0,05, a temperatura média sobe de 16,6 para 32,0 °C e a umidade do ar fica fora de [0,3; 0,9] em 99% dos dias
com planta (multiplicador médio do clima sobre o crescimento de 0,87 para 0,72). O calor não é a causa direta (a temperatura
passa de 35 °C em apenas um dia com planta por episódio), e a irrigação não compensa o clima. O laço do diagnóstico reproduz a
recompensa oficial do professor em cada regime (5.442, 2.943 e 5.313 contra 5.442, 2.943 e 5.311).

Os estudos complementares sobre parada antecipada na imitação e sobre as seis condições da etapa de imitação estão no
repositório [MimePlant-Earlystop](https://github.com/homerodepaula/MimePlant-Earlystop).

---

## 2. Estrutura do repositório

```
protocolo.py                 constantes do protocolo: sementes por papel, professor, regimes, pastas, orçamento (fonte única)
avaliacao.py                 a única função de avaliação (política amostrada, gerador próprio por episódio)
environment/                 FarmGym v4: env.py, subsistemas e mascara_acoes.py
agents/imitation-learning/   professor (bot_perfeito.py), rede LSTM, BC e DAgger
agents/rl/                   PPO recorrente (ppo.py), ator-crítico, ambientes em lote, espaços de busca (hp.py)
gerar_demos.py               demonstrações do professor
referencias.py               professor, aleatório e variantes nos 3 conjuntos de sementes × 3 regimes
treinar_warmstart_il.py      clone do professor (BC + DAgger)
bakeoff.py                   reforço com configuração manual (com e sem imitação)
hpo/                         hpo_rl.py (Optuna), hpo_dehb.py + dehb_nucleo.py (DEHB nativo), pos_hpo.py,
                             reavaliar_topk.py, ablacoes.py, analise_hpo.py, controle_zero.py
execucao/                    encadeadores da cadeia experimental e geração das tabelas e figuras
Resultados/                  resultados publicados da rodada 3 (JSON, logs e bancos dos estudos), tabelas e figuras da
                             dissertação e diagnóstico do regime seco
tests/                       pytest
analise/                     figuras auxiliares e diagnóstico do regime seco (diag_regime_seco.py)
results/                     saídas das suas execuções; os modelos (*.pt) e as demonstrações ficam fora do git
                             (os bancos *.db dos estudos de HPO ficam em ../experimentos/hpo/)
reavaliar_decodificacao.py   histórico: reavaliação de checkpoints da rodada 2 com política amostrada (τ = 1) no lugar do argmax
requirements.txt             da etapa de qualificação; não serve para reproduzir a rodada 3 (use a Seção 3)
requirements-lock.txt        registro dos pacotes de outro interpretador da máquina original; não use para instalar
```

---

## 3. Requisitos e instalação

| Item | Usado na dissertação |
|---|---|
| Sistema | Windows 11 |
| GPU | NVIDIA GeForce RTX 4060, 8 GB (CUDA 12.4). Sem GPU o código roda em CPU, muito mais devagar |
| Python | **3.10.11** |
| Disco | ~8 GB livres (ambiente virtual com PyTorch CUDA ≈ 5 GB; saídas ≈ 3 GB, dos quais os modelos somam ≈ 2 GB) |

Crie um ambiente virtual e instale **exatamente** estas versões:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.2.6 gymnasium==1.2.3 optuna==4.8.0 scipy==1.15.3 matplotlib==3.10.9 scikit-learn==1.7.2 pytest==9.1.1
```

Confira:

```
python -c "import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), numpy.__version__)"
```

Tem de sair `3.10.11 2.6.0+cu124 True 2.2.6` (`False` no lugar de `True` se for rodar em CPU).

Observações:

- **Não use o `requirements-lock.txt` para instalar.** Ele lista 153 pacotes de outro interpretador da máquina original (não
  inclui, por exemplo, o scikit-learn usado nas análises), quase todos alheios ao projeto. As duas linhas acima são tudo o
  que o código importa (o scikit-learn entra pelo fANOVA do Optuna).
- **Não instale o pacote `dehb`.** Ele exige `numpy < 2` e rebaixaria o NumPy do projeto. O DEHB usado aqui é implementação
  própria (`hpo/dehb_nucleo.py`).
- Se o ambiente virtual ficar dentro de uma pasta sincronizada (OneDrive, Dropbox), a sincronização de milhares de arquivos do
  PyTorch atrapalha. Prefira criá-lo fora, por exemplo em `C:\Users\<você>\.venvs\mimeplant`.
- Use sempre `python -X utf8`: há acentos nos caminhos e nas saídas.

---

## 4. Onde as saídas são gravadas

> Os resultados da rodada 3 usados na dissertação já estão publicados em [`Resultados/`](Resultados/)
> (JSON, logs e bancos dos estudos de HPO; sem os modelos `.pt`). Não é preciso reexecutar nada para conferi-los.

Os resultados ficam em dois lugares, e **o segundo fica ao lado do repositório, não dentro dele**:

```
<pasta qualquer>/
├── MimePlant-Dissertacao/          este repositório
│   └── results/rodada3/            referências, clone, reforço, reavaliações e ablações (JSON + modelos .pt)
└── experimentos/                   criada automaticamente na primeira execução
    ├── hpo/                        um banco SQLite e uma pasta por estudo de HPO
    ├── _jobs/                      log e estado de cada job dos encadeadores
    ├── analise_hpo_rodada3*/       análises (JSON, figura, tabela LaTeX)
    ├── analise_rl_padrao_rodada3/  curvas do reforço sem HPO (`execucao/curvas_rl_padrao.py`)
    └── analise_final/              `numeros_final.json` (Passo 7)
```

Clone o repositório dentro de uma pasta sua para que `experimentos/` seja criada ali, e não em um lugar inesperado. No
Windows, use uma pasta de caminho curto ou ative `git config --global core.longpaths true` antes de clonar: o caminho
mais longo do repositório tem 111 caracteres.

Cada execução de referências, clone, reforço, reavaliação e ablação grava, na sua pasta de saída, `config.json` (versões,
dispositivo, protocolo e configuração), `metricas.json`, `_STATUS.txt` e, ao terminar, `_DONE.flag` (em caso de falha,
`_FAILED.flag`). Nos estudos de HPO, cada tentativa grava só um `config.json` reduzido e o `metricas.json` em
`trial_<n>/`; o estudo grava `PROGRESSO.json`, `_STATUS.txt` e `_DONE.flag`, e as versões ficam nos atributos do banco.

---

## 5. Passo a passo

Todos os comandos são dados **da raiz do repositório**, com o ambiente virtual ativado.

A cadeia completa consome cerca de **9,5 dias de GPU** na máquina da dissertação. Os tempos abaixo foram medidos nela.

| Passo | Comando | Tempo medido |
|---|---|---|
| 0 | Testes rápidos | 3–4 min |
| 1 | Demonstrações do professor | 5 min (CPU) |
| 2 | Referências, clone e reforço sem HPO | 5 h |
| 3 | Estudo de AutoRL: TPE, busca aleatória e DEHB + pós-HPO | 123 h |
| 4 | Controle: reforço do zero com HPO | 29 h |
| 5 | Otimizadores adicionais: GP-BO e Sobol | 56 h |
| 6 | Controle adicional: reforço do zero com GP-BO | 15 h |
| 7 | Tabelas e figuras | 1 min |
| 8 | Diagnóstico do regime seco | 5 min (CPU ou GPU) |

Os encadeadores dos passos 2 a 6 **só rodam em Windows** (usam prioridade de processo e janela oculta do Windows). Em outro
sistema, use os comandos da [Seção 6](#6-executar-à-mão-sem-os-encadeadores), que são os mesmos que eles disparam.

Todo encadeador aceita `--so-plano`, que imprime cada comando que seria lançado e termina **sem lançar nenhum job** (só cria
`../experimentos/_jobs/` e registra o plano no log). Rode-o sempre antes. Os demais scripts não têm essa opção: executam ao
serem chamados.

### Passo 0 — conferir a instalação

```
python -X utf8 -m pytest -q -m "not lento"
```

Resultado esperado: `454 passed, 2 skipped, 44 deselected, 1 xfailed`. O `xfailed` é intencional. Um teste do DEHB pode
falhar de forma intermitente sob a carga da suíte completa; rode de novo (ver a Seção 10).

### Passo 1 — demonstrações do professor

```
python -X utf8 gerar_demos.py --retomar
python -X utf8 -c "import protocolo as P; print(P.hash_arquivo(P.DEMOS_PADRAO) == P.HASH_DEMOS_PADRAO)"
git checkout -- results/demos_v5
```

O primeiro comando gera 200 episódios (73.000 transições, arquivo de 378 MB em `results/demos_v5/`, fora do git).
O segundo **tem de imprimir `True`**: ele confere a soma SHA-256 do arquivo contra a registrada em `protocolo.py`.
O terceiro devolve os metadados versionados que o gerador reescreveu (o `config.json` versionado registra 16 min, de uma
geração anterior em máquina ocupada).

Se sair `False`, o ambiente ou o professor não são os da dissertação (versão do NumPy diferente é a causa mais comum).
**Não troque o hash para passar**: o clone se recusa a treinar com demonstrações diferentes, de propósito.

### Passo 2 — referências, clone e reforço sem HPO

```
python -X utf8 execucao/encadear_ate_rl_padrao.py --so-plano
python -X utf8 execucao/encadear_ate_rl_padrao.py
```

| Etapa | O que roda | Processos | Tempo |
|---|---|---|---|
| E1 | `referencias.py` (CPU) e `treinar_warmstart_il.py` com 5 sementes (GPU), em paralelo | 2 | 54 min |
| E2 | `bakeoff.py`: `ppo_il` e `ppo_zero`, 5 sementes cada, 1.000.000 de passos por execução | 4 | 4 h 05 |
| E3 | `bakeoff.py --retomar` com tudo pronto: grava `resumo.json` e a flag global | 1 | 1 min |

Saídas: `results/rodada3/referencias/referencias.json`, `results/rodada3/il_warmstart/`, `results/rodada3/rl_padrao/resumo.json`.

Para ver as curvas de validação, inclusive com a etapa em andamento:

```
python -X utf8 execucao/curvas_rl_padrao.py
```

### Passo 3 — estudo de AutoRL (TPE, busca aleatória, DEHB)

```
python -X utf8 execucao/encadear_hpo.py --so-plano
python -X utf8 execucao/encadear_hpo.py
```

| Fase | O que roda | Processos | Tempo |
|---|---|---|---|
| F1 | `tpe_r3_a`, `tpe_r3_b`, `tpe_r3_c`: 3 estudos × 2 trabalhadores | 6 | 31 h |
| F2 | `rs_r3_a`, `rs_r3_b`, `rs_r3_c`: 3 estudos × 2 trabalhadores | 6 | 29 h 30 |
| F3 | `pos_hpo.py`: reavalia o incumbente de cada estudo com 5 sementes, roda as 4 ablações e a análise | 4 | 20 h 30 |
| F4a | `dehb_r3_a` e `dehb_r3_b`, 3 avaliações simultâneas cada | 6 | 22 h |
| F4b | `dehb_r3_c` | 3 | 13 h |
| F5a | Reavaliação dos incumbentes do DEHB com 5 sementes | 4 | 6 h 20 |
| F5b | Análise dos 9 estudos | 1 | 1 min |

Cada estudo recebe **32 FRE** (1 FRE = uma execução completa de reforço = 1.000.000 de passos). As sementes de otimizador
são 1001, 1002 e 1003. O agente final (MimePlant) é definido na F3, como o candidato de maior média normalizada
(0 = aleatório, 1 = professor) em validação-objetivo entre os seis incumbentes de TPE e de busca aleatória
(`escolher_melhor_candidato`, em `hpo/reavaliar_topk.py`).

Saídas: `../experimentos/hpo/<estudo>.db` e `<estudo>/`; `results/rodada3/hpo_reavaliacao/resumo.json`;
`results/rodada3/hpo_ablacoes/resumo.json`; `results/rodada3/hpo_reavaliacao_dehb/resumo.json`;
`../experimentos/analise_hpo_rodada3/` e `../experimentos/analise_hpo_rodada3_dehb/`.

### Passo 4 — controle: reforço do zero com HPO

Fecha o desenho de quatro quadrantes (com e sem imitação × com e sem HPO).

```
python -X utf8 execucao/encadear_controle_zero.py --so-plano
python -X utf8 execucao/encadear_controle_zero.py
```

| Fase | O que roda | Processos | Tempo |
|---|---|---|---|
| C1 | `tpe_zero_r3` e `rs_zero_r3`, 3 trabalhadores cada, `--inicializacao zero` | 6 | 24 h 45 |
| C2 | Reavaliação do incumbente de cada estudo com 5 sementes | 4 | 4 h 25 |
| C3 | `controle_zero.py --so-comparar`: a tabela 2 × 2 | 1 | 1 min |

Este encadeador **espera** o Passo 3 terminar (no máximo 48 h; depois disso desiste e é preciso relançá-lo). Como o Passo 3
leva cerca de 123 h, só o deixe na fila durante as últimas 48 h do Passo 3, por exemplo depois do início da fase F4b. Para
começar de imediato, acrescente `--sem-esperar`. Ele depende de `results/rodada3/hpo_reavaliacao/resumo.json`, gerado na
fase F3 do Passo 3.

O espaço de busca do controle tem 8 hiperparâmetros, e não 12: os 4 que regulam o acoplamento com o clone não têm efeito
sem clone.

No texto da dissertação, o agente "reforço do zero com HPO" é o melhor, em validação-objetivo, dos três estudos do zero
(TPE, busca aleatória e GP-BO; este último vem do Passo 6).

Saída: `results/rodada3/controle_zero/comparacao.json`.

### Passo 5 — otimizadores adicionais (GP-BO e Sobol)

```
python -X utf8 execucao/encadear_extras.py --so-plano
python -X utf8 execucao/encadear_extras.py
```

| Fase | O que roda | Processos | Tempo |
|---|---|---|---|
| G1 | `gp_r3_a`, `gp_r3_b`, `gp_r3_c`: 3 estudos × 2 trabalhadores | 6 | 21 h 30 |
| G2, G3 | Reavaliação dos incumbentes com 5 sementes e análise | 4 | 4 h 35 |
| Q1 | `qmc_r3_a`, `qmc_r3_b`, `qmc_r3_c`: 3 estudos × 2 trabalhadores | 6 | 24 h 15 |
| Q2, Q3 | Reavaliação dos incumbentes com 5 sementes e análise | 4 | 6 h |

Este encadeador **espera** o Passo 4 terminar (até 72 h). Use `--sem-esperar` para começar de imediato e `--so-gp` para rodar
só o GP-BO.

Os dois otimizadores rodam por `execucao/hpo_sampler_extra.py`, que chama o `hpo/hpo_rl.py` sem modificá-lo e substitui
apenas o objeto amostrador. Assim o objetivo, a poda, a contabilidade de orçamento e as sementes são idênticos aos dos
estudos de TPE e de busca aleatória.

Saídas: `results/rodada3/hpo_reavaliacao_gp/resumo.json`, `results/rodada3/hpo_reavaliacao_qmc/resumo.json`.

### Passo 6 — controle adicional: reforço do zero com HPO por GP-BO

Acrescenta ao controle do Passo 4 o otimizador cuja melhor execução no IL + RL teve a maior recompensa em teste (`gp_r3_a`,
4.633; 4.630 em validação-objetivo). Essa escolha foi feita depois de conhecido esse resultado. O GP-BO não concorreu à
escolha do MimePlant (aplicada aos 15 incumbentes, a regra também não o escolheria).

```
python -X utf8 execucao/encadear_gp_zero.py --so-plano
python -X utf8 execucao/encadear_gp_zero.py
```

| Fase | O que roda | Processos | Tempo |
|---|---|---|---|
| Z1 | `gp_zero_r3`: um estudo, 3 trabalhadores, `--inicializacao zero` | 3 | 12 h 05 |
| Z2 | Reavaliação do incumbente com 5 sementes | 4 | 3 h |
| Z3, Z4 | Análise do estudo e tabela comparativa em recompensa bruta | 1 | 2 min |

O estudo espelha o `tpe_zero_r3` em tudo, menos no amostrador: 32 FRE, poda Hyperband, `n_startup = 5`, semente de
otimizador 1001, 3 trabalhadores, 8 hiperparâmetros e reavaliação do incumbente com as 5 sementes de treino. A busca só
precisa do Passo 2 concluído, mas a tabela comparativa (Z4) lê as reavaliações dos Passos 3, 4 e 5; se elas ainda não
existirem, a Z4 falha e basta rodar o mesmo comando de novo depois. Com `--execucoes 3`, o encadeador roda três buscas
independentes (`gp_zero_r3_a`, `_b` e `_c`, 2 trabalhadores cada), como no IL + RL; a dissertação usa a busca única.

Saídas: `../experimentos/hpo/gp_zero_r3.db`, `results/rodada3/hpo_reavaliacao_zero_gp/resumo.json`,
`../experimentos/analise_hpo_rodada3_zero_gp/` e `results/rodada3/controle_zero_gp/comparacao.json`.

### Passo 7 — tabelas e figuras da dissertação

```
python -X utf8 execucao/gerar_artefatos_final.py <pasta do projeto LaTeX>
```

Lê os JSON de `results/rodada3/` e grava, na pasta indicada, nove tabelas em `tabelas/*.tex` e três figuras
(`fig_cinco_condicoes.pdf`, `fig_otimizadores.pdf` e `fig_esforco.pdf`). Os valores calculados para o capítulo de resultados
ficam em `../experimentos/analise_final/numeros_final.json`. Sem argumento, usa `../final`. Exige os Passos 2 a 6 concluídos.
Os valores são recompensa bruta no conjunto de teste.

`execucao/gerar_artefatos_dissertacao.py` é a versão anterior do gerador (sete tabelas e quatro figuras, sem o GP no
reforço do zero). Continua no repositório porque o gerador atual o importa como módulo.

### Passo 8 — diagnóstico do regime seco

```
python -X utf8 analise/diag_regime_seco.py
```

Roda o professor e o agente aleatório nos três regimes, nas sementes de validação (500–549, 50 episódios; nunca as de teste),
e registra a cada passo a temperatura, a chuva e a umidade do solo. Grava `Resultados/diag_regime_seco/` (`episodios.json`,
`numeros_seco.json` e `RESUMO_SECO.txt`). A recompensa média do professor é conferida contra `referencias.json`. A saída
reproduz byte a byte os arquivos publicados. Não depende dos passos anteriores, exceto do `referencias.json` (Passo 2) ou da
cópia publicada em `Resultados/rodada3/referencias/`.

---

## 6. Executar à mão, sem os encadeadores

Os encadeadores apenas lançam os comandos abaixo, esperam as flags `_DONE.flag` e relançam o que cair. **Uma etapa só
começa depois de a anterior ter terminado.** Linhas da mesma etapa rodam ao mesmo tempo, cada uma em seu terminal.

Abreviações usadas abaixo, para as linhas caberem. Defina-as **em cada terminal** (a Seção 6 pede um terminal por linha de
uma mesma etapa):

```
# bash
export WS=results/rodada3/il_warmstart REFS=results/rodada3/referencias/referencias.json

# PowerShell (use $WS e $REFS nas linhas abaixo)
$WS = 'results/rodada3/il_warmstart'; $REFS = 'results/rodada3/referencias/referencias.json'
```

### Passo 2

```
# E1 (em paralelo)
python -X utf8 referencias.py --saida results/rodada3/referencias --retomar
python -X utf8 treinar_warmstart_il.py --seeds 42 123 2024 7 1337 --config lr_alto --saida $WS --retomar

# E2 (4 processos)
python -X utf8 bakeoff.py --bracos ppo_il   --seeds 42 123 2024 --orcamento 1000000 --saida results/rodada3/rl_padrao --warmstart $WS --referencias $REFS --retomar
python -X utf8 bakeoff.py --bracos ppo_il   --seeds 7 1337      --orcamento 1000000 --saida results/rodada3/rl_padrao --warmstart $WS --referencias $REFS --retomar
python -X utf8 bakeoff.py --bracos ppo_zero --seeds 42 123 2024 --orcamento 1000000 --saida results/rodada3/rl_padrao --warmstart $WS --referencias $REFS --retomar
python -X utf8 bakeoff.py --bracos ppo_zero --seeds 7 1337      --orcamento 1000000 --saida results/rodada3/rl_padrao --warmstart $WS --referencias $REFS --retomar

# E3 (não treina nada: só consolida o resumo)
python -X utf8 bakeoff.py --bracos ppo_il ppo_zero --seeds 42 123 2024 7 1337 --orcamento 1000000 --saida results/rodada3/rl_padrao --warmstart $WS --referencias $REFS --retomar
```

### Passo 3

Para cada estudo de TPE, com o par (`<S>`, `<SEMENTE>`) em (`a`, `1001`), (`b`, `1002`), (`c`, `1003`) e `<W>` em `0`, `1`
(6 processos ao todo):

```
python -X utf8 hpo/hpo_rl.py --study tpe_r3_<S> --sampler tpe --n-fre 32 --orcamento 1000000 --seed-sampler <SEMENTE> --n-startup 5 --paralelo --trabalhador <W> --warmstart $WS --referencias $REFS --retomar
```

Busca aleatória, mesma combinação de `<S>`, `<SEMENTE>` e `<W>`:

```
python -X utf8 hpo/hpo_rl.py --study rs_r3_<S> --sampler random --n-fre 32 --orcamento 1000000 --seed-sampler <SEMENTE> --paralelo --trabalhador <W> --warmstart $WS --referencias $REFS --retomar
```

Pós-HPO, depois que os seis estudos tiverem `_DONE.flag`:

```
python -X utf8 hpo/pos_hpo.py --studies tpe_r3_a tpe_r3_b tpe_r3_c rs_r3_a rs_r3_b rs_r3_c --k 1 --orcamento 1000000 --padroes ppo_il ppo_zero --max-horas 120 --retomar
```

DEHB, um processo por estudo (cada um lança 3 avaliações simultâneas); no máximo dois estudos ao mesmo tempo:

```
python -X utf8 hpo/hpo_dehb.py --study dehb_r3_<S> --n-fre 32 --orcamento 1000000 --paralelo 3 --niveis 2 --seed <SEMENTE> --warmstart $WS --referencias $REFS --retomar
```

Reavaliação do DEHB e análise dos nove estudos. **A análise só depois de a reavaliação terminar**, porque ela a lê:

```
python -X utf8 hpo/reavaliar_topk.py --study dehb_r3_a dehb_r3_b dehb_r3_c --k 1 --orcamento 1000000 --saida results/rodada3/hpo_reavaliacao_dehb --paralelo 4 --retomar
python -X utf8 hpo/analise_hpo.py --studies tpe_r3_a tpe_r3_b tpe_r3_c rs_r3_a rs_r3_b rs_r3_c dehb_r3_a dehb_r3_b dehb_r3_c --reavaliacao results/rodada3/hpo_reavaliacao_dehb --saida ../experimentos/analise_hpo_rodada3_dehb
```

### Passo 4

Com `<W>` em `0`, `1`, `2` (6 processos ao todo):

```
python -X utf8 hpo/hpo_rl.py --study tpe_zero_r3 --sampler tpe    --n-fre 32 --orcamento 1000000 --inicializacao zero --seed-sampler 1001 --n-startup 5 --paralelo --trabalhador <W> --warmstart $WS --referencias $REFS --retomar
python -X utf8 hpo/hpo_rl.py --study rs_zero_r3  --sampler random --n-fre 32 --orcamento 1000000 --inicializacao zero --seed-sampler 1001               --paralelo --trabalhador <W> --warmstart $WS --referencias $REFS --retomar
```

Depois:

```
python -X utf8 hpo/reavaliar_topk.py --study tpe_zero_r3 rs_zero_r3 --k 1 --orcamento 1000000 --saida results/rodada3/hpo_reavaliacao_zero --paralelo 4 --retomar
python -X utf8 hpo/controle_zero.py --so-comparar --n-fre 32 --orcamento 1000000
```

### Passo 5

Com `<AMOSTRADOR>` em `gp` ou `qmc`, e as mesmas combinações de `<S>`, `<SEMENTE>` e `<W>` do Passo 3 (6 processos por
família, uma família de cada vez):

```
python -X utf8 execucao/hpo_sampler_extra.py --sampler-extra <AMOSTRADOR> --n-startup 5 --study <AMOSTRADOR>_r3_<S> --n-fre 32 --orcamento 1000000 --seed-sampler <SEMENTE> --paralelo --trabalhador <W> --warmstart $WS --referencias $REFS --retomar
```

Depois, para cada família:

```
python -X utf8 hpo/reavaliar_topk.py --study gp_r3_a gp_r3_b gp_r3_c --k 1 --orcamento 1000000 --saida results/rodada3/hpo_reavaliacao_gp --paralelo 4 --retomar
python -X utf8 hpo/analise_hpo.py --studies gp_r3_a gp_r3_b gp_r3_c --reavaliacao results/rodada3/hpo_reavaliacao_gp --saida ../experimentos/analise_hpo_rodada3_gp

python -X utf8 hpo/reavaliar_topk.py --study qmc_r3_a qmc_r3_b qmc_r3_c --k 1 --orcamento 1000000 --saida results/rodada3/hpo_reavaliacao_qmc --paralelo 4 --retomar
python -X utf8 hpo/analise_hpo.py --studies qmc_r3_a qmc_r3_b qmc_r3_c --reavaliacao results/rodada3/hpo_reavaliacao_qmc --saida ../experimentos/analise_hpo_rodada3_qmc
```

### Passo 6

Com `<W>` em `0`, `1`, `2` (3 processos):

```
python -X utf8 execucao/hpo_sampler_extra.py --sampler-extra gp --n-startup 5 --study gp_zero_r3 --n-fre 32 --orcamento 1000000 --inicializacao zero --seed-sampler 1001 --paralelo --trabalhador <W> --warmstart $WS --referencias $REFS --retomar
```

Depois:

```
python -X utf8 hpo/reavaliar_topk.py --study gp_zero_r3 --k 1 --orcamento 1000000 --saida results/rodada3/hpo_reavaliacao_zero_gp --paralelo 4 --retomar
python -X utf8 hpo/analise_hpo.py --studies gp_zero_r3 --reavaliacao results/rodada3/hpo_reavaliacao_zero_gp --saida ../experimentos/analise_hpo_rodada3_zero_gp
python -X utf8 execucao/comparar_gp_zero.py
```

O último comando lê as reavaliações dos Passos 3, 4 e 5.

Três cuidados ao rodar à mão:

- **Espere 20 segundos entre o lançamento de cada trabalhador de um mesmo estudo** (são 2 nos Passos 3 e 5, e 3 nos Passos 4
  e 6). Se dois deles criarem o banco SQLite ao mesmo tempo, um morre com `table alembic_version already exists`. Basta
  relançá-lo.
- **Passe sempre `--n-startup 5`** nos estudos com modelo. O padrão do `hpo_rl.py` é 20, e omitir a flag muda o protocolo.
- **Passe `--paralelo` em todos os trabalhadores** de um estudo com mais de um processo.

---

## 7. Acompanhar, interromper e retomar

**Rodar sem deixar um terminal aberto.** Os encadeadores rodam em primeiro plano. Para lançá-los desanexados no Windows, de
modo que sobrevivam ao fechamento do terminal (PowerShell, ajuste os dois caminhos):

```powershell
$py  = "C:\caminho\para\.venv\Scripts\python.exe"
$dir = "C:\caminho\para\MimePlant-Dissertacao"
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$py`" -X utf8 `"$dir\execucao\encadear_hpo.py`""; CurrentDirectory = $dir }
```

**Onde olhar.**

| O quê | Onde |
|---|---|
| Em que fase a cadeia está | `../experimentos/_jobs/encadeador_*.log` |
| Estado de cada job (rodando, pronto, falhou) | `../experimentos/_jobs/estado*.json` |
| Saída completa de um job | `../experimentos/_jobs/<job>.log` |
| Progresso de um estudo de HPO | `../experimentos/hpo/<estudo>/PROGRESSO.json` (FRE consumidos, concluídos, podados, melhor valor) |
| Andamento de uma execução de reforço | `results/rodada3/.../seed_<s>/_STATUS.txt` |

**Interromper.** Encerre o processo do encadeador e depois os processos `python` dos jobs. Nada precisa ser limpo.

**Retomar.** Rode o mesmo comando de novo. A cadeia é idempotente: pula toda etapa que já tem `_DONE.flag` e relança o
resto com `--retomar`. Vale também depois de queda de energia ou reinício.

O que se perde numa queda é só o treinamento em andamento de cada processo. No Optuna, o trial interrompido fica marcado
como em execução e não é cobrado do orçamento. No DEHB, a avaliação interrompida é refeita, e o que se perdeu fica
registrado no `PROGRESSO.json` sem ser cobrado.

Cada encadeador relança sozinho um job que termine sem as flags, até três lançamentos, e desiste da cadeia depois disso.

---

## 8. Testes e fumaça

```
python -X utf8 -m pytest -q -m "not lento"      # 3–4 min; use sempre
python -X utf8 -m pytest -q -m lento            # ~6 min: avaliador real do DEHB, varredura de caos, treinos curtos
```

O `pytest.ini` não exclui os testes lentos por padrão: `pytest` sem `-m` roda tudo.

**Fumaça** (orçamentos mínimos, minutos; não é resultado). O `--smoke` encolhe o orçamento e a avaliação (e, no clone,
dispensa a conferência do hash) e **não muda a pasta de saída**. Sem `--saida` própria, a fumaça do clone gravaria
`_DONE.flag` em `results/rodada3/il_warmstart`, e a rodada de verdade pularia essa semente; sem `--study` próprio, a do
`hpo/hpo_rl.py` gravaria no estudo `tpe_r3`. Use sempre assim:

```
python -X utf8 treinar_warmstart_il.py --seeds 42 --smoke --saida results/_smoke/il_warmstart
python -X utf8 hpo/hpo_rl.py --study smoke_rl --smoke --warmstart results/_smoke/il_warmstart --referencias results/rodada3/referencias/referencias.json
python -X utf8 execucao/hpo_sampler_extra.py --sampler-extra gp --study smoke_gp --smoke --n-startup 2 --warmstart results/_smoke/il_warmstart --referencias results/rodada3/referencias/referencias.json
```

A primeira precisa das demonstrações do Passo 1; as duas últimas precisam também do `referencias.json` do Passo 2. Depois,
apague `results/_smoke/` e, em `../experimentos/hpo/`, as pastas e os bancos `smoke_*`.

---

## 9. Protocolo experimental

| Papel | Sementes | O que decide |
|---|---|---|
| Treino | 42, 123, 2024, 7, 1337 | pesos, coleta e embaralhamento (cada trial de HPO usa só a 42) |
| Validação-rollback | 500–519 | melhor bloco, melhor rodada de DAgger e poda |
| Validação-objetivo | 520–549 | objetivo da HPO e escolha do agente final |
| Teste | 999–1048 | **apenas reportar** |
| Demonstrações | 10.000–10.199 | episódios do professor |

- **Nada é escolhido no conjunto de teste.**
- **Orçamento:** 1.000.000 de passos por execução de reforço (1 FRE); 32 FRE por execução de HPO. Um trial podado cobra só
  a fração que treinou. Cada estudo termina com 32,8 a 34,9 FRE (32,8 a 33,9 no IL + RL e 34,5 a 34,9 no reforço do zero),
  porque os trials em andamento quando o orçamento acaba são concluídos.
- **Poda:** Hyperband com fidelidade mínima de um terço do orçamento e fator 3. O bloco final nunca é podado.
- **Espaço de busca:** 12 hiperparâmetros da fase de reforço (`agents/rl/hp.py`). A inicialização não é sorteada. No
  reforço do zero, os 4 que regulam o acoplamento com o clone saem do espaço (8 hiperparâmetros).
- **Objetivo da HPO:** média, nos três regimes climáticos, da recompensa normalizada em validação-objetivo (0 = aleatório,
  1 = professor). A normalização iguala o peso dos regimes na busca; os resultados são reportados em recompensa bruta.
- **Avaliação:** política amostrada da distribuição mascarada, com gerador próprio por episódio. O treinamento ocorre só no
  regime normal; os regimes seco e úmido são avaliação fora da distribuição.
- **Agregação:** média interquartil (IQM) dos retornos por episódio, agrupados entre as sementes de treino (250 por regime nos
  agentes treinados e no clone; 50 no professor e no agente aleatório), e depois a média simples dos três regimes. Os testes
  usam a recompensa média de cada semente de treino: *t* pareado quando as duas execuções partem dos mesmos pesos (efeito da
  HPO dentro de um braço, ablações) e *t* de Welch quando se comparam agentes com e sem imitação (inclusive a interação,
  medida entre os ganhos da HPO por semente nos dois braços).
  O teste de Friedman entre as cinco condições (blocos: os 150 episódios de teste) entra só como descrição, e o Wilcoxon de postos
  sinalizados não é usado entre as sementes, porque com cinco pares o menor *p* exato é 0,0625.
- **Reprodutibilidade:** as constantes do ambiente, das sementes, do professor e do orçamento por execução vivem em
  `protocolo.py`; os parâmetros das buscas (32 FRE, `n_startup`, sementes 1001–1003, trabalhadores, k = 1) ficam nos
  encadeadores (`execucao/encadear_*.py`); cada execução grava versões e configuração; o arquivo de demonstrações é
  identificado por SHA-256.

---

## 10. Problemas conhecidos

| Sintoma | Causa | O que fazer |
|---|---|---|
| O hash das demonstrações não confere | NumPy diferente de 2.2.6, ou ambiente alterado | Reinstalar as versões da Seção 3. Não trocar o hash |
| `table alembic_version already exists` ao iniciar um estudo | Dois trabalhadores criaram o banco ao mesmo tempo | Relançar o que morreu. Os encadeadores já espaçam os lançamentos em 20 s |
| `CUDA out of memory` | Processos demais para a GPU | Reduzir o número de trabalhadores e retomar. Seis processos cabem em 8 GB |
| Um teste lento falha em `bat['pid'] == corridas['pid_baixa']` | No Windows, o `python.exe` de um ambiente virtual abre o interpretador real como processo filho, e os dois PIDs diferem | Ignorar. Só afeta essa asserção; a retomada do DEHB usa travas do sistema operacional, não PIDs |
| Um teste do DEHB falha às vezes na suíte rápida (por exemplo, `test_espelho_e_projecao_do_jornal_pode_ser_refeito_e_nunca_derruba_o_estudo`) | Disputa de tempo entre os subprocessos do avaliador falso sob carga | Rodar de novo; isolado, o teste passa |
| `ExperimentalWarning` do `GPSampler` | Aviso do Optuna | Inofensivo |
| A cadeia não avança depois de um reinício | Os processos morreram | Rodar o mesmo encadeador de novo |
| `experimentos/` apareceu num lugar inesperado | Ela é criada ao lado do repositório | Mover o repositório para dentro de uma pasta própria |

Para reproduzir os experimentos, o guia válido é este arquivo.
