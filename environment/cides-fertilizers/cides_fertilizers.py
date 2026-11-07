import numpy as np
from typing import Dict, Tuple, List, Optional
from enum import Enum

class TipoNutriente(Enum):
    NITROGENIO = "N"
    FOSFORO = "P"
    POTASSIO = "K"
    CARBONO = "C"

class TipoDefensivo(Enum):
    HERBICIDA = "herbicida"
    PESTICIDA = "pesticida"
    FUNGICIDA = "fungicida"
    GERAL = "geral"

class DefensivosFertilizantes:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade

        self.velocidade_absorcao_base = 0.1
        self.tamanho_saco = 5.0

        self.resetar()

    def resetar(self):
        self.aplicacoes = {}
        self.passo_tempo = 0

        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                self.aplicacoes[(x, y)] = []

    def aplicar_fertilizante(self, x: int, y: int, tipo_nutriente: TipoNutriente, quantidade: float):
        pos = (x, y)
        if pos in self.aplicacoes:
            aplicacao = Aplicacao(
                tipo='fertilizante',
                tipo_nutriente=tipo_nutriente,
                quantidade=quantidade,
                velocidade_absorcao=self.velocidade_absorcao_base
            )
            self.aplicacoes[pos].append(aplicacao)

    def aplicar_defensivo(self, x: int, y: int, tipo_defensivo: TipoDefensivo, forca_efeito: float, quantidade: float):
        pos = (x, y)
        if pos in self.aplicacoes:
            aplicacao = Aplicacao(
                tipo='defensivo',
                tipo_defensivo=tipo_defensivo,
                forca_efeito=forca_efeito,
                quantidade=quantidade,
                velocidade_absorcao=self.velocidade_absorcao_base
            )
            self.aplicacoes[pos].append(aplicacao)

    def espalhamento(self, x: int, y: int, tipo_produto: str, quantidade: float, **kwargs):
        if tipo_produto == 'fertilizante':
            tipo_nutriente = kwargs.get('tipo_nutriente', TipoNutriente.NITROGENIO)
            self.aplicar_fertilizante(x, y, tipo_nutriente, quantidade)
        elif tipo_produto == 'defensivo':
            tipo_defensivo = kwargs.get('tipo_defensivo', TipoDefensivo.GERAL)
            forca_efeito = kwargs.get('forca_efeito', 0.5)
            self.aplicar_defensivo(x, y, tipo_defensivo, forca_efeito, quantidade)

    def espalhamento_saco(self, x: int, y: int, tipo_produto: str, **kwargs):
        self.espalhamento(x, y, tipo_produto, self.tamanho_saco, **kwargs)

    def atualizar(self):
        self.passo_tempo += 1

        for pos in self.aplicacoes:
            aplicacoes_para_remover = []

            for i, app in enumerate(self.aplicacoes[pos]):
                liberado = app.liberar()

                if app.esta_esgotada():
                    aplicacoes_para_remover.append(i)

            for i in reversed(aplicacoes_para_remover):
                del self.aplicacoes[pos][i]

    def obter_contribuicoes_solo(self) -> Dict:
        contribuicoes = {}

        for pos, aplicacoes in self.aplicacoes.items():
            contribuicoes[pos] = {
                'fertilizante_N': 0.0,
                'fertilizante_P': 0.0,
                'fertilizante_K': 0.0,
                'fertilizante_C': 0.0,
                'efeito_herbicida': 0.0,
                'efeito_pesticida': 0.0,
                'efeito_fungicida': 0.0,
                'quantidade_defensivo': 0.0
            }

            for app in aplicacoes:
                if app.tipo == 'fertilizante' and app.tipo_nutriente:
                    chave = f'fertilizante_{app.tipo_nutriente.value}'
                    contribuicoes[pos][chave] += app.obter_taxa_liberacao_atual()
                elif app.tipo == 'defensivo' and app.tipo_defensivo:
                    if app.tipo_defensivo == TipoDefensivo.HERBICIDA:
                        contribuicoes[pos]['efeito_herbicida'] += app.forca_efeito * app.obter_taxa_liberacao_atual()
                    elif app.tipo_defensivo == TipoDefensivo.PESTICIDA:
                        contribuicoes[pos]['efeito_pesticida'] += app.forca_efeito * app.obter_taxa_liberacao_atual()
                    elif app.tipo_defensivo == TipoDefensivo.FUNGICIDA:
                        contribuicoes[pos]['efeito_fungicida'] += app.forca_efeito * app.obter_taxa_liberacao_atual()

                    contribuicoes[pos]['quantidade_defensivo'] += app.obter_taxa_liberacao_atual()

        return contribuicoes

    def obter_observacao(self) -> Dict:
        obs = {}

        for pos, aplicacoes in self.aplicacoes.items():
            obs[pos] = {
                'aplicacoes_ativas': len(aplicacoes),
                'total_fertilizante': sum(app.quantidade for app in aplicacoes if app.tipo == 'fertilizante'),
                'total_defensivo': sum(app.quantidade for app in aplicacoes if app.tipo == 'defensivo'),
                'aplicacoes': [
                    {
                        'tipo': app.tipo,
                        'tipo_nutriente': app.tipo_nutriente.value if app.tipo_nutriente else None,
                        'tipo_defensivo': app.tipo_defensivo.value if app.tipo_defensivo else None,
                        'quantidade_restante': app.quantidade,
                        'forca_efeito': getattr(app, 'forca_efeito', None)
                    }
                    for app in aplicacoes
                ]
            }

        return obs

    def definir_velocidade_absorcao(self, velocidade: float):
        self.velocidade_absorcao_base = velocidade

    def definir_tamanho_saco(self, tamanho: float):
        self.tamanho_saco = tamanho

_MEIA_VIDA_DEFENSIVO = {
    TipoDefensivo.HERBICIDA: 14,
    TipoDefensivo.PESTICIDA: 7,
    TipoDefensivo.FUNGICIDA: 10,
    TipoDefensivo.GERAL: 10,
}


class Aplicacao:
    def __init__(self, tipo: str, quantidade: float, velocidade_absorcao: float,
                 tipo_nutriente: Optional[TipoNutriente] = None,
                 tipo_defensivo: Optional[TipoDefensivo] = None,
                 forca_efeito: float = 0.5):
        self.tipo = tipo
        self.quantidade = quantidade
        self.quantidade_inicial = quantidade
        self.velocidade_absorcao = velocidade_absorcao
        self.tipo_nutriente = tipo_nutriente
        self.tipo_defensivo = tipo_defensivo
        self.forca_efeito = forca_efeito
        self.idade = 0

        if tipo == 'defensivo' and tipo_defensivo is not None:
            meia_vida = _MEIA_VIDA_DEFENSIVO.get(tipo_defensivo, 10)
            self._decaimento_por_passo = 1.0 - 0.5 ** (1.0 / meia_vida)
        else:
            self._decaimento_por_passo = 0.0

    def liberar(self) -> float:
        if self.quantidade <= 0:
            return 0.0

        if self._decaimento_por_passo > 0:
            liberado = self.quantidade * self._decaimento_por_passo
        else:
            liberado = min(self.quantidade, self.velocidade_absorcao)

        self.quantidade = max(0.0, self.quantidade - liberado)
        self.idade += 1
        return liberado

    def obter_taxa_liberacao_atual(self) -> float:
        if self.quantidade <= 0:
            return 0.0
        if self._decaimento_por_passo > 0:
            return self.quantidade * self._decaimento_por_passo
        return min(self.velocidade_absorcao, self.quantidade)

    def esta_esgotada(self) -> bool:
        return self.quantidade <= 0.001
