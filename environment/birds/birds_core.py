"""Aves do FarmGym.

FarmGym v4 (06/09/2026): a população de aves passa a depender do ambiente. Até a v3 ela era um passeio aleatório sem
viés: `nascimentos = int(pop * 0,02)` e `mortes = int(pop * 0,015)` davam ZERO para toda população alcançável (o teto é
50), então a única dinâmica era `np.random.randint(-2, 3)`. Não havia dependência de estação, alimento, clima nem
migração, e o espantalho **exterminava** a população em dois ou três passos, o que apagava a diferença entre o
espantalho básico e o avançado.

Agora cada célula converge, de forma logística, para uma capacidade de suporte que depende da estação, da onda
migratória, do alimento disponível na célula e da repelência dos espantalhos. A repelência age sobre a **capacidade**,
não sobre a população: qualquer redução multiplicativa aplicada a cada passo, por menor que seja, levaria a população a
zero e reproduziria o extermínio da v3.

Este módulo também absorve o que sobrou de `MigracaoSazonal` (removida na v4): aquele módulo lia `dados_clima['passo']`
enquanto o ambiente escreve `'step'`, então era inerte e mantinha um segundo modelo de aves, paralelo a este, que só
alimentava a observação. A onda migratória virou um estado deste módulo, e é a única parte da dinâmica de aves que não
é dedutível do passo do episódio — por isso ocupa uma dimensão própria da observação.
"""
import numpy as np
from typing import Dict, Tuple, List
import random

# ── FarmGym v4: capacidade de suporte (substitui natalidade/mortalidade, que eram código morto) ──
_CAPACIDADE_BASE = 6.0          # aves/célula sem alimento nenhum, em estação neutra
_GANHO_ALIMENTO = 1.5           # capacidade × (1 + 1,5 × alimento): célula com fruto (alimento ~0,95) vale ~2,4×
_FATOR_ESTACAO = {'primavera': 0.7, 'verao': 1.2, 'outono': 1.2, 'inverno': 0.5}
_TAXA_CRESCIMENTO = 0.12        # convergência logística para a capacidade (meia-vida ~6 passos)
_IMIGRACAO = 0.3                # chegada de fora por passo: impede que uma célula fique extinta para sempre
_RUIDO = 0.8                    # desvio-padrão do ruído gaussiano por passo
# Onda migratória (o que sobrou de `MigracaoSazonal`): nas estações de pico, uma chegada de fora que dura 2–3 semanas
# e não é dedutível do passo do episódio.
_P_INICIO_ONDA = 0.02           # por passo, quando _FATOR_ESTACAO >= 1 (≈ 2 ondas por episódio)
_DURACAO_ONDA = (12, 26)        # passos
_GANHO_ONDA = 0.8               # durante a onda, a capacidade sobe 80 %
# Espantalho: repelência multiplicativa sobre a capacidade = exp(−_SENSIBILIDADE_ESPANTALHO × força efetiva)
_SENSIBILIDADE_ESPANTALHO = 0.7  # força 1,0 (básico) → 0,497 | força 2,0 (avançado) → 0,247
_ALCANCE_VIZINHO = 0.5          # o espantalho também repele nas 4 células ortogonais, com metade da força


class Passaros:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade

        self.populacao_minima = 0.0
        self.populacao_maxima = 50.0
        self.media_populacao_base = 15

        self.resetar()

    def resetar(self):
        self.populacao_passaros = {}
        self.passo_tempo = 0
        self.passos_onda = 0            # v4: passos restantes da onda migratória em curso
        self.pressao_migratoria = 0.0   # v4: 1,0 durante uma onda; é uma dimensão da observação
        self._gerar_populacao_inicial()

    def _gerar_populacao_inicial(self):
        # v4: começa disperso em torno da capacidade de uma célula sem cultura na primavera, para não haver um
        # transiente artificial no início do episódio (a v3 sorteava 0..24, longe de qualquer equilíbrio).
        limite = 2.0 * _CAPACIDADE_BASE * _FATOR_ESTACAO['primavera']
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                self.populacao_passaros[(x, y)] = float(np.random.uniform(0.0, limite))

    def _atualizar_onda_migratoria(self, fator_estacao: float):
        """v4: substitui o módulo `MigracaoSazonal`. Nas estações de pico, uma onda de chegada de fora começa com
        probabilidade `_P_INICIO_ONDA` por passo e dura 12–26 passos, elevando a capacidade em 80 %."""
        if self.passos_onda > 0:
            self.passos_onda -= 1
        elif fator_estacao >= 1.0 and np.random.random() < _P_INICIO_ONDA:
            self.passos_onda = int(np.random.randint(*_DURACAO_ONDA))
        self.pressao_migratoria = 1.0 if self.passos_onda > 0 else 0.0

    def _forca_espantalho_efetiva(self, pos: Tuple[int, int], instalacoes: Dict) -> float:
        """v4: o espantalho da célula soma força total; os das 4 células ortogonais somam `_ALCANCE_VIZINHO` da sua.
        Sem alcance, `colocar_espantalho` (que sorteia a célula) protegeria 1 de 16 células por ação e a decisão não
        teria escala."""
        forca = instalacoes.get(pos, {}).get('forca_espantalho', 0.0) or 0.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            vizinho = instalacoes.get((pos[0] + dx, pos[1] + dy), {})
            forca += _ALCANCE_VIZINHO * (vizinho.get('forca_espantalho', 0.0) or 0.0)
        return float(forca)

    def atualizar(self, instalacoes: Dict, clima: Dict = None, alimento: Dict = None):
        """v4: cada célula converge (logística) para uma capacidade de suporte = base × estação × onda migratória
        × (1 + ganho × alimento) × repelência do espantalho.

        `clima` fornece a estação; `alimento` é o dicionário por célula de
        `AmbienteFazendaGym._calcular_disponibilidade_alimento()` — até a v3 esse dicionário só era consumido pela
        `MigracaoSazonal`, que era inerte."""
        self.passo_tempo += 1
        clima = clima or {}
        alimento = alimento or {}

        fator_estacao = _FATOR_ESTACAO.get(clima.get('estacao', 'primavera'), 1.0)
        self._atualizar_onda_migratoria(fator_estacao)
        fator_estacao *= (1.0 + _GANHO_ONDA * self.pressao_migratoria)

        for pos in list(self.populacao_passaros.keys()):
            pop = float(self.populacao_passaros[pos])
            repelencia = float(np.exp(-_SENSIBILIDADE_ESPANTALHO * self._forca_espantalho_efetiva(pos, instalacoes)))
            capacidade = max(0.2, _CAPACIDADE_BASE * fator_estacao
                             * (1.0 + _GANHO_ALIMENTO * float(alimento.get(pos, 0.0))) * repelencia)
            nova_pop = (pop
                        + _TAXA_CRESCIMENTO * pop * (1.0 - pop / capacidade)
                        + _IMIGRACAO * fator_estacao * repelencia
                        + np.random.normal(0.0, _RUIDO))
            self.populacao_passaros[pos] = float(np.clip(nova_pop, self.populacao_minima, self.populacao_maxima))

    def obter_observacao(self) -> Dict:
        return self.populacao_passaros.copy()

    def obter_populacao_em(self, x: int, y: int) -> float:
        pos = (x, y)
        return float(self.populacao_passaros.get(pos, 0.0))

    def obter_populacao_total(self) -> float:
        return float(sum(self.populacao_passaros.values()))

    def definir_parametros_populacao(self, pop_min: float, pop_max: float, media_base: float):
        self.populacao_minima = float(pop_min)
        self.populacao_maxima = float(pop_max)
        self.media_populacao_base = media_base
