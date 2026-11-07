import numpy as np
from typing import Dict, List, Tuple, Optional
import random

class EventosClimaticosExtremos:
    """FarmGym v3 (06/09/2026, D-24): os eventos passam a AGIR -- `obter_impacto_extremo(categoria)` e somado ao custo da
    acao executada durante o evento (ver `env.step`). Na v2 o metodo nunca era chamado e os eventos so entravam na
    observacao. Limiares de calor/frio alinhados ao estresse da planta (35 C / 5 C); na v2 eram 40 C e -5 C, fora do
    intervalo do gerador de clima, e duas dimensoes da observacao ficavam constantes. Calibrado em 06/09: calor 33 C (regime
    seco, T = 32 +- 1), frio 8 C (inverno in-dist, T = 8 +- 1,3)."""

    def __init__(self):
        self.eventos_extremos = []
        self.impactos_atuais = {}
        self.dados_historicos = []
        self.probabilidade_evento = 0.05   # gate por passo e por tipo; a duracao real e (min, max) dias -- ver o bug do passo abaixo

        self.tipos_evento = {
            'estresse_calor': {
                'limiar_temp': 33,
                'limiar_umidade': 0.3,
                'duracao': (3, 7),
                'impactos': {
                    'plantar': -5.0,
                    'irrigar': -3.0,
                    'colher': -8.0,
                    'fertilizar': -2.0,
                    'controle_pragas': -1.0
                }
            },
            'estresse_frio': {
                'limiar_temp': 8,
                'limiar_umidade': 0.8,
                'duracao': (2, 5),
                'impactos': {
                    'plantar': -4.0,
                    'irrigar': -1.0,
                    'colher': -6.0,
                    'fertilizar': -1.5,
                    'controle_pragas': -0.5
                }
            },
            'seca': {
                'limiar_temp': 30,
                'limiar_umidade': 0.27,   # v3: 0,2 nunca ocorria (umidade do ar do regime seco = 0,25 +- 0,02)
                'duracao': (5, 10),
                'impactos': {
                    'plantar': -6.0,
                    'irrigar': 0.0,       # v3: era -5; irrigar E a resposta certa a seca (a escassez de agua nao e modelada)
                    'colher': -9.0,
                    'fertilizar': -3.0,
                    'controle_pragas': -2.0
                }
            },
            'tempestade': {
                'limiar_temp': 15,
                'limiar_vento': 0.7,      # v3: 0,8 e 30 mm nunca ocorriam (vento ~ 0,5 +- 0,12; chuva gama com media 5-10 mm)
                'limiar_chuva': 25,
                'duracao': (1, 3),
                'impactos': {
                    'plantar': -7.0,
                    # v4: era +2,0 -- o unico termo por passo estritamente POSITIVO fora da colheita, e portanto um
                    # atrator: regar durante tempestade pagava. Nenhuma acao paga por passo (criterio 21 do checklist).
                    'irrigar': 0.0,
                    'colher': -10.0,
                    'fertilizar': -4.0,
                    'controle_pragas': -3.0
                }
            }
        }

        self.eventos_ativos = []

    def verificar_condicoes_extremas(self, dados_clima: Dict) -> bool:
        tem_extremo = False

        for tipo_evento, limiares in self.tipos_evento.items():
            if self._verificar_condicoes_evento(dados_clima, tipo_evento, limiares):
                if tipo_evento not in [e['tipo'] for e in self.eventos_ativos]:
                    duracao = random.randint(*limiares['duracao'])
                    # v3 (06/09/2026): o ambiente passa o passo na chave 'step'; este modulo lia 'passo' (sempre ausente), logo
                    # passo_inicio era 0, `_atualizar_eventos_ativos(0)` nunca expirava nada e todo evento durava ate o fim do
                    # episodio (uma seca no regime seco custava -400/ep ao oraculo). Aceita as duas chaves.
                    evento = {
                        'tipo': tipo_evento,
                        'passo_inicio': dados_clima.get('step', dados_clima.get('passo', 0)),
                        'duracao': duracao,
                        'severidade': self._calcular_severidade(dados_clima, limiares),
                        'impactos': limiares['impactos'].copy()
                    }
                    self.eventos_ativos.append(evento)
                    self.eventos_extremos.append(evento)
                    tem_extremo = True

        self._atualizar_eventos_ativos(dados_clima.get('step', dados_clima.get('passo', 0)))

        return tem_extremo

    def _verificar_condicoes_evento(self, dados_clima: Dict, tipo_evento: str, limiares: Dict) -> bool:
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)
        vento = dados_clima.get('vento', 0.5)
        chuva = dados_clima.get('chuva', 0)

        if random.random() > self.probabilidade_evento:
            return False

        if tipo_evento == 'estresse_calor':
            return temp > limiares['limiar_temp'] and umidade_ar < limiares['limiar_umidade']
        elif tipo_evento == 'estresse_frio':
            return temp < limiares['limiar_temp']
        elif tipo_evento == 'seca':
            return temp > limiares['limiar_temp'] and umidade_ar < limiares['limiar_umidade'] and chuva < 1.0
        elif tipo_evento == 'tempestade':
            return vento > limiares.get('limiar_vento', 15) or chuva > limiares.get('limiar_chuva', 20)

        return False

    def _calcular_severidade(self, dados_clima: Dict, limiares: Dict) -> float:
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)

        severidade = 0.5
        if 'limiar_temp' in limiares:
            if limiares['limiar_temp'] > 0:
                excesso = max(0, temp - limiares['limiar_temp'])
                severidade += min(0.5, excesso / 20)
            else:
                deficit = max(0, limiares['limiar_temp'] - temp)
                severidade += min(0.5, deficit / 20)

        return min(1.0, severidade)

    def _atualizar_eventos_ativos(self, passo_atual: int):
        self.eventos_ativos = [
            evento for evento in self.eventos_ativos
            if passo_atual - evento['passo_inicio'] < evento['duracao']
        ]

    def obter_impacto_extremo(self, nome_acao: str) -> float:
        impacto_total = 0

        for evento in self.eventos_ativos:
            impacto = evento['impactos'].get(nome_acao, 0)
            impacto_total += impacto * evento['severidade']

        return impacto_total

    def obter_impactos_atuais(self) -> List[float]:
        impactos = []

        for tipo_evento in self.tipos_evento.keys():
            ativo = any(e['tipo'] == tipo_evento for e in self.eventos_ativos)
            if ativo:
                evento = next(e for e in self.eventos_ativos if e['tipo'] == tipo_evento)
                impactos.append(evento['severidade'])
            else:
                impactos.append(0.0)

        if self.eventos_ativos:
            severidade_media = np.mean([e['severidade'] for e in self.eventos_ativos])
            impactos.append(severidade_media)
        else:
            impactos.append(0.0)

        return impactos

    def resetar(self):
        self.eventos_extremos = []
        self.eventos_ativos = []
        self.impactos_atuais = {}
        self.dados_historicos = []

    def obter_resumo_eventos(self) -> Dict:
        resumo = {
            'eventos_ativos': len(self.eventos_ativos),
            'total_eventos': len(self.eventos_extremos),
            'tipos_evento': [e['tipo'] for e in self.eventos_ativos],
            'severidade_media': np.mean([e['severidade'] for e in self.eventos_ativos]) if self.eventos_ativos else 0.0
        }
        return resumo
