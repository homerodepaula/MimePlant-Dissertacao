import numpy as np
from typing import Dict, Tuple, List
import math

# FarmGym v2 (S7, D-17): sensibilidade da vida microbiana ao defensivo 0,5 → 0.2. Com 0,5, uma aplicação
# diária de herbicida derrubava a qualidade do solo (e a colheita) mais do que as ervas custavam.
THETA_0 = 0.1
THETA_DEFENSIVO = 0.2
# FarmGym v3 (S9): decaimento por passo dos nutrientes 0,01 -> 0,004. Com 0,01 os quatro nutrientes zeravam nos primeiros
# 20-40 dias e o fertilizante (2,0 x eficiencia, liberado em ~20 dias) nao chegava ao rendimento. O rendimento passa a
# depender do nutriente mais escasso (lei do minimo) em `plant.py`.
DECAIMENTO_NUTRIENTE = 0.004

class Solo:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade
        self.resetar()

    def resetar(self):
        self.estado_solo = {}
        self.passo_tempo = 0
        self._inicializar_solo()

    def _inicializar_solo(self):
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                self.estado_solo[(x, y)] = EstadoSolo()

    def atualizar(self, clima: Dict, defensivos: Dict):
        self.passo_tempo += 1

        for pos, solo in self.estado_solo.items():
            self._atualizar_umidade(solo, clima)
            self._atualizar_nutrientes(solo, defensivos.get(pos, {}))
            self._atualizar_vida_microbiana(solo, defensivos.get(pos, {}))
            solo.atualizar_qualidade()

    def _atualizar_umidade(self, solo: 'EstadoSolo', clima: Dict):
        chuva = clima.get('chuva', 0)
        temperatura = clima.get('temperatura', 20)
        umidade_ar = clima.get('umidade_ar', 0.5)
        vento = clima.get('vento', 0.3)

        solo.umidade += min(chuva / 50.0, 0.3) * 0.6

        taxa_evaporacao = 0.05 * (temperatura / 20) * (1 - umidade_ar) + vento * 0.01
        solo.umidade -= taxa_evaporacao

        if solo.umidade > 0.1:
            solo.umidade -= 0.02

        solo.umidade = np.clip(solo.umidade, 0.0, 1.0)

    _EFICIENCIA_FERTILIZANTE = {'N': 0.30, 'P': 0.25, 'K': 0.25, 'C': 0.20}   # v3: P 0,15 -> 0,25 (era o gargalo: 5 aplicacoes/ano so de P)

    def _atualizar_nutrientes(self, solo: 'EstadoSolo', dados_defensivos: Dict):
        for nutriente in ['N', 'P', 'K', 'C']:
            chave_fertilizante = f'fertilizante_{nutriente}'
            if chave_fertilizante in dados_defensivos:
                eficiencia = self._EFICIENCIA_FERTILIZANTE[nutriente]
                solo.nutrientes[nutriente] += dados_defensivos[chave_fertilizante] * eficiencia

        for nutriente in solo.nutrientes:
            solo.nutrientes[nutriente] -= DECAIMENTO_NUTRIENTE
            solo.nutrientes[nutriente] = min(1.0, max(0.0, solo.nutrientes[nutriente]))

    def _atualizar_vida_microbiana(self, solo: 'EstadoSolo', dados_defensivos: Dict):
        p_permanencia = self._calcular_probabilidade_sobrevivencia_microbiana(solo, dados_defensivos)

        if np.random.random() > p_permanencia:
            solo.densidade_microbiana *= 0.9
        else:
            if 0.3 < solo.umidade < 0.8:
                solo.densidade_microbiana = min(1.0, solo.densidade_microbiana * 1.05)

    def _calcular_probabilidade_sobrevivencia_microbiana(self, solo: 'EstadoSolo', dados_defensivos: Dict) -> float:
        theta_0 = THETA_0
        theta_defensivo = THETA_DEFENSIVO
        quantidade_defensivo = dados_defensivos.get('quantidade_defensivo', 0)
        distancia = quantidade_defensivo
        p_permanencia = math.exp(-(theta_0 + theta_defensivo * distancia))
        return max(0.01, min(1.0, p_permanencia))

    def obter_observacao(self) -> Dict:
        obs = {}
        for pos, solo in self.estado_solo.items():
            obs[pos] = {
                'umidade': solo.umidade,
                'nutrientes': solo.nutrientes.copy(),
                'densidade_microbiana': solo.densidade_microbiana,
                'qualidade': solo.qualidade
            }
        return obs

    def obter_solo_em(self, x: int, y: int) -> 'EstadoSolo':
        pos = (x, y)
        return self.estado_solo.get(pos, EstadoSolo())

    def adicionar_nutrientes(self, x: int, y: int, nutrientes: Dict[str, float]):
        pos = (x, y)
        if pos in self.estado_solo:
            for nutriente, quantidade in nutrientes.items():
                if nutriente in self.estado_solo[pos].nutrientes:
                    self.estado_solo[pos].nutrientes[nutriente] += quantidade
                    self.estado_solo[pos].nutrientes[nutriente] = min(1.0,
                        self.estado_solo[pos].nutrientes[nutriente])

class EstadoSolo:
    def __init__(self):
        self.umidade = 0.5
        self.nutrientes = {
            'N': 0.3,
            'P': 0.2,
            'K': 0.2,
            'C': 0.4
        }
        self.densidade_microbiana = 0.5
        self.qualidade = 0.5

    def atualizar_qualidade(self):
        media_nutrientes = sum(self.nutrientes.values()) / len(self.nutrientes)

        self.qualidade = (
            0.3 * media_nutrientes +
            0.3 * self.umidade +
            0.2 * self.densidade_microbiana +
            0.2 * min(1.0, media_nutrientes * self.densidade_microbiana)
        )

        self.qualidade = np.clip(self.qualidade, 0.0, 1.0)
