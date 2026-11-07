import numpy as np
from typing import Dict, Tuple, List
import random

# FarmGym v2 (S6, D-17): taxa de crescimento base das ervas 0,05 → 0,15. Com 0,05 a densidade de equilíbrio
# ficava em ~0,27 e as ervas custavam ~5 % do rendimento — o herbicida nunca compensava. Ver
# versao final/experimentos/ANALISE_RECOMPENSA_E_AMBIENTE.md.
TAXA_CRESCIMENTO_BASE = 0.15

class ErvasDaninhas:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade

        self.taxa_crescimento_base = TAXA_CRESCIMENTO_BASE
        self.densidade_maxima = 1.0
        self.taxa_dispersao_sementes = 0.1

        self.resetar()

    def resetar(self):
        self.densidade_ervas = {}
        self.passo_tempo = 0
        self._inicializar_ervas()

    def _inicializar_ervas(self):
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                densidade = np.random.uniform(0.0, 0.2)
                self.densidade_ervas[(x, y)] = densidade

    def atualizar(self, clima: Dict, solo: Dict, defensivos: Dict):
        self.passo_tempo += 1

        nova_densidade = {}

        for pos in list(self.densidade_ervas.keys()):
            densidade_atual = self.densidade_ervas[pos]

            fator_crescimento = self._calcular_fator_crescimento(pos, clima, solo, defensivos)

            nova_densidade_ervas = densidade_atual + fator_crescimento * self.taxa_crescimento_base

            nova_densidade_ervas *= (1 - densidade_atual * 0.5)

            fator_dispersao = self._calcular_fator_dispersao(pos)
            nova_densidade_ervas += fator_dispersao

            efeito_defensivo = self._calcular_efeito_defensivo(pos, defensivos)
            nova_densidade_ervas *= (1 - efeito_defensivo)

            nova_densidade_ervas = np.clip(nova_densidade_ervas, 0.0, self.densidade_maxima)

            nova_densidade[pos] = nova_densidade_ervas

        self.densidade_ervas = nova_densidade

    def _calcular_fator_crescimento(self, pos: Tuple[int, int], clima: Dict, solo: Dict, defensivos: Dict) -> float:
        fator = 1.0

        temp = clima.get('temperatura', 20)
        chuva = clima.get('chuva', 0)

        if 15 <= temp <= 25:
            fator *= 1.2
        elif temp < 5 or temp > 35:
            fator *= 0.3

        if chuva > 0:
            fator *= min(1.5, 1.0 + 0.3 * min(chuva / 10.0, 1.0))

        dados_solo = solo.get(pos, {})
        umidade = dados_solo.get('umidade', 0.5)
        qualidade = dados_solo.get('qualidade', 0.5)

        fator *= (0.5 + umidade)
        fator *= (0.3 + 0.7 * qualidade)

        return fator

    def _calcular_fator_dispersao(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        dispersao = 0.0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                pos_vizinho = (x + dx, y + dy)
                if pos_vizinho in self.densidade_ervas:
                    densidade_vizinho = self.densidade_ervas[pos_vizinho]
                    if densidade_vizinho > 0.5:
                        dispersao += densidade_vizinho * self.taxa_dispersao_sementes * 0.1

        return dispersao

    def _calcular_efeito_defensivo(self, pos: Tuple[int, int], defensivos: Dict) -> float:
        dados_defensivo = defensivos.get(pos, {})

        efeito_herbicida = dados_defensivo.get('efeito_herbicida', 0)

        quantidade_defensivo = dados_defensivo.get('quantidade_defensivo', 0)
        efeito_geral = quantidade_defensivo * 0.3

        efeito_total = efeito_herbicida + efeito_geral
        return min(0.9, efeito_total)

    def obter_observacao(self) -> Dict:
        obs = {}
        for pos, densidade in self.densidade_ervas.items():
            obs[pos] = {'densidade': float(densidade)}
        return obs

    def obter_densidade_em(self, x: int, y: int) -> float:
        pos = (x, y)
        return self.densidade_ervas.get(pos, 0.0)

    def obter_cobertura_total(self) -> float:
        if not self.densidade_ervas:
            return 0.0
        return sum(self.densidade_ervas.values()) / len(self.densidade_ervas)

    def remover_ervas(self, x: int, y: int, eficiencia_remocao: float = 0.8):
        pos = (x, y)
        if pos in self.densidade_ervas:
            self.densidade_ervas[pos] *= (1 - eficiencia_remocao)

    def definir_parametros_crescimento(self, taxa_crescimento: float, densidade_maxima: float, taxa_dispersao: float):
        self.taxa_crescimento_base = taxa_crescimento
        self.densidade_maxima = densidade_maxima
        self.taxa_dispersao_sementes = taxa_dispersao
