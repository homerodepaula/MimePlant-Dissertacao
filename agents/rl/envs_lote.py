"""envs_lote.py — V ambientes FarmGym avançando em passo de lote (uma inferência da LSTM por passo para todos).

Cada ambiente é semeado explicitamente (`seed_base + contador`) a cada reset, então a coleta é reproduzível dada a
semente de treino. Episódios têm horizonte fixo (365, truncado): o fim é tratado como terminal (sem bootstrap), como no
código da qualificação; a observação inclui `passo/365`, então a função de valor pode aprender o efeito do horizonte.
Ao terminar um episódio, o ambiente reinicia sozinho e `fim[v] = True` sinaliza que `obs[v]` já é o dia 0 do próximo.

RNG por ambiente: o FarmGym sorteia dos geradores GLOBAIS (`np.random`, `random`), então V ambientes no mesmo processo
entrelaçariam seus sorteios e cada `reset(seed)` re-semearia os outros. `EnvLote` guarda um estado de RNG por ambiente e o
troca no gerador global em cada reset/step (624 inteiros; custo desprezível). Resultado: o episódio de semente s é o mesmo
aqui e em `avaliacao.avaliar` (testado), e o estado global de quem chamou fica intacto.
"""
import random
from typing import Callable, Dict, List, Optional

import numpy as np

import protocolo as P


class EnvLote:
    def __init__(self, n_envs: int, fabrica_env: Callable, seed_base: int):
        self.n = int(n_envs)
        self.envs = [fabrica_env() for _ in range(self.n)]
        self.seed_base = int(seed_base)
        self.contador = 0
        self._estados_rng = [None] * self.n     # (estado np.random, estado random) de cada ambiente
        self.obs = np.zeros((self.n, P.OBS_DIM), dtype=np.float32)
        self.mascaras = np.ones((self.n, P.N_ACOES), dtype=np.float32)
        self.sementes_atuais = [None] * self.n
        self._retorno = np.zeros(self.n)
        self._passos = np.zeros(self.n, dtype=np.int64)
        self._comp = [dict() for _ in range(self.n)]
        self._acoes = np.zeros((self.n, P.N_ACOES), dtype=np.int64)
        self.episodios_concluidos: List[Dict] = []   # {semente, retorno, passos, comp, acoes}
        self.reset_todos()

    def _proxima_semente(self) -> int:
        s = self.seed_base + self.contador
        self.contador += 1
        return s

    def _entrar_rng(self, v: int):
        """Troca o gerador global pelo estado do ambiente v; devolve o estado de quem chamou."""
        externo = (np.random.get_state(), random.getstate())
        if self._estados_rng[v] is not None:
            np.random.set_state(self._estados_rng[v][0])
            random.setstate(self._estados_rng[v][1])
        return externo

    def _sair_rng(self, v: int, externo):
        self._estados_rng[v] = (np.random.get_state(), random.getstate())
        np.random.set_state(externo[0])
        random.setstate(externo[1])

    def _reset_env(self, v: int):
        s = self._proxima_semente()
        externo = self._entrar_rng(v)
        try:
            obs, info = self.envs[v].reset(seed=s)      # semeia o global = o fluxo deste ambiente
        finally:
            self._sair_rng(v, externo)
        self.obs[v] = obs
        self.mascaras[v] = info['mascara_acoes']
        self.sementes_atuais[v] = s
        self._retorno[v] = 0.0
        self._passos[v] = 0
        self._comp[v] = {}
        self._acoes[v] = 0

    def reset_todos(self):
        for v in range(self.n):
            self._reset_env(v)
        return self.obs.copy(), self.mascaras.copy()

    def _step_env(self, v: int, a: int):
        externo = self._entrar_rng(v)
        try:
            return self.envs[v].step(a)
        finally:
            self._sair_rng(v, externo)

    def step(self, acoes):
        """acoes: (V,) inteiros. Devolve obs (V,148), mascaras (V,15), recompensas (V,), fins (V,) bool, infos (lista)."""
        recompensas = np.zeros(self.n, dtype=np.float64)
        fins = np.zeros(self.n, dtype=bool)
        infos = []
        for v in range(self.n):
            a = int(acoes[v])
            obs, r, terminado, truncado, info = self._step_env(v, a)
            recompensas[v] = r
            self._retorno[v] += r
            self._passos[v] += 1
            self._acoes[v, a] += 1
            for k, val in info.get('recompensa_comp', {}).items():
                self._comp[v][k] = self._comp[v].get(k, 0.0) + float(val)
            infos.append(info)
            if terminado or truncado:
                fins[v] = True
                self.episodios_concluidos.append({
                    'semente': self.sementes_atuais[v], 'retorno': float(self._retorno[v]), 'passos': int(self._passos[v]),
                    'comp': dict(self._comp[v]), 'acoes': self._acoes[v].tolist(),
                    'kpis': {k: float(x) for k, x in info.get('kpis', {}).items()},
                })
                self._reset_env(v)
            else:
                self.obs[v] = obs
                self.mascaras[v] = info['mascara_acoes']
        return self.obs.copy(), self.mascaras.copy(), recompensas, fins, infos

    def coletar_concluidos(self) -> List[Dict]:
        """Devolve (e esvazia) a lista de episódios terminados desde a última chamada."""
        out, self.episodios_concluidos = self.episodios_concluidos, []
        return out


def fabrica_env_padrao():
    """Ambiente oficial (FarmGym v3 + máscara)."""
    P.preparar_sys_path()
    from env import AmbienteFazendaGym
    from mascara_acoes import InvolucroMascaraAcoes
    return InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=P.PASSOS_MAXIMOS))
