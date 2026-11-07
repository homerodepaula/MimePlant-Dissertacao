"""Polinizadores do FarmGym.

FarmGym v3 (06/09/2026, D-24 / S10): o clima e as instalações passam a modular o CRESCIMENTO da população, não o estoque.
Na v2, `nova = (atual + cresc − mort) × fator_clima × fator_instalacao` multiplicava a densidade inteira por 0,5 em qualquer
dia frio, quente, chuvoso ou ventoso, e a população colapsava a zero em todo episódio (`polinizadores_final = 0` para os
dois oráculos; a cerca viva não mudava nada mensurável). Agora a cerca viva (×1,5 no crescimento local, ×1,1 por vizinha)
e o pesticida (mortalidade) são decisões com efeito no florescimento (ver `plant.py`).
"""
import numpy as np
from typing import Dict, Tuple, List
import random

_TAXA_CRESCIMENTO = 0.04   # v3: 0,03 -> 0,04 (recuperacao apos defensivos)
_MORTALIDADE_BASE = 0.01


class Polinizadores:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade

        self.densidade_minima = 0.0
        self.densidade_maxima = 1.0
        self.media_densidade_base = 0.3

        self.resetar()

    def resetar(self):
        self.densidade_polinizadores = {}
        self.passo_tempo = 0
        self._gerar_densidade_inicial()

    def _gerar_densidade_inicial(self):
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                densidade = np.random.uniform(self.densidade_minima, self.densidade_maxima)
                densidade = 0.7 * densidade + 0.3 * self.media_densidade_base
                densidade = np.clip(densidade, self.densidade_minima, self.densidade_maxima)
                self.densidade_polinizadores[(x, y)] = densidade

    def atualizar(self, clima: Dict, instalacoes: Dict, defensivos: Dict = None):
        self.passo_tempo += 1

        if defensivos is None:
            defensivos = {}

        fator_clima = self._calcular_fator_clima(clima)

        for pos in list(self.densidade_polinizadores.keys()):
            atual = self.densidade_polinizadores[pos]

            fator_instalacao = self._calcular_fator_instalacao(pos, instalacoes)
            crescimento = atual * _TAXA_CRESCIMENTO * (1.0 - atual) * fator_clima * fator_instalacao
            mortalidade = atual * _MORTALIDADE_BASE

            # v3: o herbicida (via `quantidade_defensivo`, residual de qualquer defensivo) pesa 0,1, nao 0,3: e o inseticida que
            # mata polinizadores; o herbicida so tira forrageio. O efeito chega a colheita pela polinizacao (plant.POLINIZACAO_PISO).
            quantidade_pesticida = defensivos.get(pos, {}).get('efeito_pesticida', 0) + \
                               defensivos.get(pos, {}).get('quantidade_defensivo', 0) * 0.1
            mortalidade_pesticida = atual * min(0.8, quantidade_pesticida * 0.2)   # v3: 0,6 -> 0,2 (uma aplicacao custa ~25 %, nao ~60 %)

            nova_densidade = atual + crescimento - mortalidade - mortalidade_pesticida
            self.densidade_polinizadores[pos] = float(np.clip(nova_densidade, self.densidade_minima, self.densidade_maxima))

    def _calcular_fator_clima(self, clima: Dict) -> float:
        fator = 1.0

        temp = clima.get('temperatura', 20)
        if temp < 10 or temp > 35:
            fator *= 0.5
        elif 15 <= temp <= 25:
            fator *= 1.2

        chuva = clima.get('chuva', 0)
        if chuva > 0:
            fator *= max(0.2, 1.0 - chuva / 20.0)

        vento = clima.get('vento', 0.1)
        if vento > 0.8:
            fator *= 0.6

        return fator

    def _calcular_fator_instalacao(self, pos: Tuple[int, int], instalacoes: Dict) -> float:
        fator = 1.0

        if instalacoes.get(pos, {}).get('tem_cerca', instalacoes.get(pos, {}).get('tipo') == 'cerca_viva'):
            fator *= 1.5

        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                viz = instalacoes.get((x + dx, y + dy))
                if viz and viz.get('tem_cerca', viz.get('tipo') == 'cerca_viva'):
                    fator *= 1.1

        return fator

    def obter_observacao(self) -> Dict:
        return self.densidade_polinizadores.copy()

    def obter_densidade_em(self, x: int, y: int) -> float:
        pos = (x, y)
        return self.densidade_polinizadores.get(pos, 0.0)

    def obter_densidade_media(self) -> float:
        if not self.densidade_polinizadores:
            return 0.0
        return sum(self.densidade_polinizadores.values()) / len(self.densidade_polinizadores)

    def definir_parametros_densidade(self, densidade_minima: float, densidade_maxima: float, media_base: float):
        self.densidade_minima = densidade_minima
        self.densidade_maxima = densidade_maxima
        self.media_densidade_base = media_base
