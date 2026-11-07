"""Instalações do FarmGym: espantalhos e cerca viva.

FarmGym v3 (06/09/2026, D-24): espantalho e cerca viva COEXISTEM na mesma célula. Na v2 os dois compartilhavam um único
dicionário `instalacoes[pos]` e um sobrescrevia o outro (colocar cerca apagava todos os espantalhos e vice-versa).
Irrigação e estufa continuam definidas como tipos, sem ação que as instale.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from enum import Enum


class TipoInstalacao(Enum):
    ESPANTALHO = "espantalho"
    CERCA_VIVA = "cerca_viva"
    IRRIGACAO = "irrigacao"
    ESTUFA = "estufa"


class TipoEspantalho(Enum):
    BASICO = "basico"
    AVANCADO = "avancado"


_DECAIMENTO_ESPANTALHO_POR_PASSO = 0.001


class Instalacoes:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade
        self.forcas_espantalho = {
            TipoEspantalho.BASICO: 1.0,
            TipoEspantalho.AVANCADO: 2.0,
        }
        self.resetar()

    def resetar(self):
        self.espantalhos: Dict[Tuple[int, int], 'Instalacao'] = {}
        self.cercas: Dict[Tuple[int, int], 'Instalacao'] = {}
        self.outras: Dict[Tuple[int, int], 'Instalacao'] = {}
        self.passo_tempo = 0

    def _dentro(self, x: int, y: int) -> bool:
        return 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]

    # ── colocar / remover ──
    def colocar_espantalho(self, x: int, y: int, tipo_espantalho: TipoEspantalho):
        if self._dentro(x, y):
            self.espantalhos[(x, y)] = Instalacao(tipo=TipoInstalacao.ESPANTALHO, tipo_espantalho=tipo_espantalho,
                                                  forca_espantalho=self.forcas_espantalho[tipo_espantalho])

    def colocar_cerca_viva(self, x: int, y: int):
        if self._dentro(x, y):
            self.cercas[(x, y)] = Instalacao(tipo=TipoInstalacao.CERCA_VIVA)

    def colocar_irrigacao(self, x: int, y: int, eficiencia: float = 0.8):
        if self._dentro(x, y):
            self.outras[(x, y)] = Instalacao(tipo=TipoInstalacao.IRRIGACAO, eficiencia_irrigacao=eficiencia)

    def remover_espantalho(self, x: int, y: int):
        self.espantalhos.pop((x, y), None)

    def remover_instalacao(self, x: int, y: int):
        pos = (x, y)
        self.espantalhos.pop(pos, None)
        self.cercas.pop(pos, None)
        self.outras.pop(pos, None)

    # ── dinâmica ──
    def atualizar(self):
        self.passo_tempo += 1
        for inst in self.espantalhos.values():
            inst.forca_espantalho = max(0.1, inst.forca_espantalho * (1.0 - _DECAIMENTO_ESPANTALHO_POR_PASSO))
            inst.idade += 1
        for inst in self.cercas.values():
            inst.idade += 1

    # ── consultas ──
    def obter_observacao(self) -> Dict:
        """Por célula com alguma instalação: `forca_espantalho` (0 se não há), `tipo_espantalho`, `tem_cerca` e `tipo`
        (compatibilidade: 'cerca_viva' se há cerca, senão 'espantalho' se há espantalho)."""
        obs = {}
        for pos in set(self.espantalhos) | set(self.cercas) | set(self.outras):
            esp = self.espantalhos.get(pos)
            obs[pos] = {
                'tipo': 'cerca_viva' if pos in self.cercas else ('espantalho' if esp else self.outras[pos].tipo.value),
                'forca_espantalho': esp.forca_espantalho if esp else 0.0,
                'tipo_espantalho': esp.tipo_espantalho if esp else None,
                'tem_cerca': pos in self.cercas,
                'eficiencia_irrigacao': self.outras[pos].eficiencia_irrigacao if pos in self.outras else None,
            }
        return obs

    def obter_instalacao_em(self, x: int, y: int) -> Optional['Instalacao']:
        pos = (x, y)
        return self.espantalhos.get(pos) or self.cercas.get(pos) or self.outras.get(pos)

    def tem_espantalho(self, x: int, y: int) -> bool:
        return (x, y) in self.espantalhos

    def obter_forca_espantalho(self, x: int, y: int) -> float:
        inst = self.espantalhos.get((x, y))
        return inst.forca_espantalho if inst else 0.0

    def tem_cerca_viva(self, x: int, y: int) -> bool:
        return (x, y) in self.cercas

    def tem_irrigacao(self, x: int, y: int) -> bool:
        inst = self.outras.get((x, y))
        return inst is not None and inst.tipo == TipoInstalacao.IRRIGACAO

    def obter_eficiencia_irrigacao(self, x: int, y: int) -> float:
        inst = self.outras.get((x, y))
        return inst.eficiencia_irrigacao if inst is not None and inst.tipo == TipoInstalacao.IRRIGACAO else 0.0

    def obter_todas_posicoes_por_tipo(self, tipo_instalacao: TipoInstalacao) -> List[Tuple[int, int]]:
        if tipo_instalacao == TipoInstalacao.ESPANTALHO:
            return list(self.espantalhos)
        if tipo_instalacao == TipoInstalacao.CERCA_VIVA:
            return list(self.cercas)
        return [pos for pos, inst in self.outras.items() if inst.tipo == tipo_instalacao]

    def n_espantalhos(self) -> int:
        return len(self.espantalhos)

    def n_cercas(self) -> int:
        return len(self.cercas)

    def definir_forca_espantalho(self, tipo_espantalho: TipoEspantalho, forca: float):
        self.forcas_espantalho[tipo_espantalho] = forca
        for inst in self.espantalhos.values():
            if inst.tipo_espantalho == tipo_espantalho:
                inst.forca_espantalho = forca


class Instalacao:
    def __init__(self, tipo: TipoInstalacao, tipo_espantalho: Optional[TipoEspantalho] = None,
                 forca_espantalho: float = 0, eficiencia_irrigacao: float = 0.8):
        self.tipo = tipo
        self.tipo_espantalho = tipo_espantalho
        self.forca_espantalho = forca_espantalho
        self.eficiencia_irrigacao = eficiencia_irrigacao
        self.idade = 0
