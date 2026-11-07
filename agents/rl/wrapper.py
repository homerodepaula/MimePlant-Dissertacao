"""wrapper.py — transformação da recompensa para o treino de RL (S5), FORA do ambiente.

`TransformadorRecompensa(escala, gamma, pbrs)`: r_treino = r_bruta/escala (+ F, se `pbrs`), onde F = γ·Φ(s') − Φ(s) é o
shaping por potencial de Ng et al. (1999), que não muda a política ótima. Φ(obs) = Σ_células presente × w(estágio) × saúde,
com os pesos do bônus de estágio da v1 (germinação 0,05 … colheita 0,30), calculada só da observação. Desligado por padrão.
O retorno BRUTO é sempre o que se reporta (EnvLote registra por episódio).
"""
import numpy as np

import protocolo as P

# Estágios (EstagioPlanta): 0 semente, 1 germinação, 2 crescimento, 3 florescimento, 4 frutificação, 5 colheita, 6 morta;
# na observação: índice / 7.
PESOS_ESTAGIO = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.0], dtype=np.float32)
# v4: vem do protocolo (que espelha env.py). Ate a v3 estes numeros eram literais aqui, e um vetor de tamanho
# diferente NAO levantava excecao: o reshape lia campos trocados em silencio.
INICIO_GRADE = P.INICIO_GRADE
TAM_CELULA = P.TAM_CELULA
N_CELULAS = P.N_CELULAS


def potencial(obs: np.ndarray) -> np.ndarray:
    """Φ para obs de forma (..., OBS_DIM)."""
    obs = np.asarray(obs, dtype=np.float32)
    grade = obs[..., INICIO_GRADE:INICIO_GRADE + N_CELULAS * TAM_CELULA].reshape(*obs.shape[:-1], N_CELULAS, TAM_CELULA)
    presente = grade[..., 0]
    saude = grade[..., 1]
    idx = np.clip(np.rint(grade[..., 2] * 7.0).astype(int), 0, 6)
    return (presente * saude * PESOS_ESTAGIO[idx]).sum(axis=-1)


class TransformadorRecompensa:
    def __init__(self, escala: float = 100.0, gamma: float = 0.99, pbrs: bool = False):
        self.escala = float(escala)
        self.gamma = float(gamma)
        self.pbrs = bool(pbrs)

    def __call__(self, r_bruta, obs_antes, obs_depois, fim):
        """Vetorizado sobre V ambientes. `fim[v]` = episódio terminou (Φ(s') = 0)."""
        r = np.asarray(r_bruta, dtype=np.float64) / self.escala
        if self.pbrs:
            phi_antes = potencial(obs_antes)
            phi_depois = potencial(obs_depois) * (~np.asarray(fim, dtype=bool))
            r = r + (self.gamma * phi_depois - phi_antes)
        return r
