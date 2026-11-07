import numpy as np
from typing import Dict, Tuple, List
import random

_INTENSIDADE_CHUVA = {
    'primavera': {'media': 8.0,  'forma': 1.5},
    'verao': {'media': 5.0,  'forma': 1.2},
    'outono': {'media': 10.0, 'forma': 1.8},
    'inverno': {'media': 7.0,  'forma': 1.5},
}
_CHUVA_MAX_MM = 50.0

class Clima:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade

        self.estacao = 'primavera'
        self.temperatura_base = 20
        self.umidade_ar_base = 0.5
        self.probabilidade_chuva_base = 0.3
        # modo_ood: None (baseline sazonal), 'dry' ou 'humid'.
        # Quando definido, _gerar_clima usa os atributos *_base DIRETAMENTE
        # em vez do regime sazonal, criando deslocamento OOD genuíno.
        self.modo_ood = None

        self.resetar()

    def resetar(self):
        self.passo_tempo = 0
        self.estacao = 'primavera'   # FarmGym v2: a estação não vaza mais de um episódio para o seguinte
        self.clima_atual = self._gerar_clima()

    def _gerar_clima(self) -> Dict:
        # Modo OOD: usa *_base diretamente (gera deslocamento de distribuição real)
        if self.modo_ood is not None:
            return self._gerar_clima_ood()

        temp_sazonal = self._obter_temperatura_sazonal()
        prob_chuva_sazonal = self._obter_probabilidade_chuva_sazonal()

        if self.passo_tempo == 0:
            temp = np.random.normal(temp_sazonal, 5)
            umidade_ar = np.random.uniform(0.2, 0.8)
            vento = np.random.uniform(0, 1)
        else:
            temp_anterior = self.clima_atual.get('temperatura', temp_sazonal)
            umidade_ar_anterior = self.clima_atual.get('umidade_ar', 0.5)
            vento_anterior = self.clima_atual.get('vento', 0.3)

            temp = 0.7 * temp_anterior + 0.3 * np.random.normal(temp_sazonal, 3)
            umidade_ar = 0.6 * umidade_ar_anterior + 0.4 * np.random.uniform(0.2, 0.8)
            vento = np.clip(0.7 * vento_anterior + 0.3 * np.random.uniform(0, 1), 0.0, 1.0)

        if np.random.random() < prob_chuva_sazonal:
            parametros = _INTENSIDADE_CHUVA[self.estacao]
            escala = parametros['media'] / parametros['forma']
            chuva_mm = np.random.gamma(parametros['forma'], escala)
            chuva = float(np.clip(chuva_mm, 0.0, _CHUVA_MAX_MM))
        else:
            chuva = 0.0

        return {
            'temperatura': float(np.clip(temp, -5, 40)),
            'umidade_ar': float(np.clip(umidade_ar, 0.0, 1.0)),
            'chuva': chuva,
            'vento': float(vento),
            'estacao': self.estacao
        }

    def _gerar_clima_ood(self) -> Dict:
        """Gera clima usando temperatura_base/umidade_ar_base/probabilidade_chuva_base
        diretamente. Aplicado em cada step quando self.modo_ood != None — produz
        deslocamento de distribuição persistente, não apenas no reset."""
        # Persistência leve (auto-regressivo) para evitar choque a cada step
        if self.passo_tempo == 0 or not self.clima_atual:
            temp = np.random.normal(self.temperatura_base, 2.5)
            umidade_ar = np.clip(np.random.normal(self.umidade_ar_base, 0.05), 0.0, 1.0)
            vento = np.random.uniform(0.3, 0.7)
        else:
            temp_anterior = self.clima_atual.get('temperatura', self.temperatura_base)
            umid_anterior = self.clima_atual.get('umidade_ar', self.umidade_ar_base)
            vento_anterior = self.clima_atual.get('vento', 0.5)
            temp = 0.7 * temp_anterior + 0.3 * np.random.normal(self.temperatura_base, 2.5)
            umidade_ar = 0.7 * umid_anterior + 0.3 * np.clip(
                np.random.normal(self.umidade_ar_base, 0.05), 0.0, 1.0)
            vento = np.clip(0.7 * vento_anterior + 0.3 * np.random.uniform(0.3, 0.7), 0.0, 1.0)

        if np.random.random() < self.probabilidade_chuva_base:
            parametros = _INTENSIDADE_CHUVA[self.estacao]
            escala = parametros['media'] / parametros['forma']
            chuva_mm = np.random.gamma(parametros['forma'], escala)
            chuva = float(np.clip(chuva_mm, 0.0, _CHUVA_MAX_MM))
        else:
            chuva = 0.0

        return {
            'temperatura': float(np.clip(temp, -5, 40)),
            'umidade_ar': float(np.clip(umidade_ar, 0.0, 1.0)),
            'chuva': chuva,
            'vento': float(vento),
            'estacao': self.estacao,
        }

    def _obter_temperatura_sazonal(self) -> float:
        temperaturas_sazonais = {
            'primavera': 18,
            'verao': 25,
            'outono': 15,
            'inverno': 8
        }
        return temperaturas_sazonais.get(self.estacao, 20)

    def _obter_probabilidade_chuva_sazonal(self) -> float:
        chuva_sazonal = {
            'primavera': 0.4,
            'verao': 0.2,
            'outono': 0.5,
            'inverno': 0.3
        }
        return chuva_sazonal.get(self.estacao, 0.3)

    def atualizar(self):
        self.passo_tempo += 1

        if self.passo_tempo % 90 == 0:
            self._avancar_estacao()

        self.clima_atual = self._gerar_clima()

    def _avancar_estacao(self):
        estacoes = ['primavera', 'verao', 'outono', 'inverno']
        indice_atual = estacoes.index(self.estacao)
        self.estacao = estacoes[(indice_atual + 1) % 4]

    def obter_observacao(self) -> Dict:
        return self.clima_atual.copy()

    def obter_temperatura_em(self, x: int, y: int) -> float:
        return self.clima_atual.get('temperatura', 20)

    def esta_congelando(self) -> bool:
        return self.clima_atual.get('temperatura', 20) < 0

    def esta_seca(self) -> bool:
        return (self.clima_atual.get('chuva', 0) < 1.0 and
                self.clima_atual.get('umidade_ar', 0.5) < 0.3)

    def definir_estacao(self, estacao: str):
        if estacao in ['primavera', 'verao', 'outono', 'inverno']:
            self.estacao = estacao
