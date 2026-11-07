"""Rede LSTM e dataset compartilhados pelo BC (bc.py) e pelo DAgger (dagger.py).

Contem:
  - Constantes da arquitetura (OBS_DIM, N_ACOES, dimensoes da rede).
  - ConjuntoDadosBC: organiza episodios em janelas de tamanho fixo para o LSTM.
  - RedeBCLSTM: a politica (encoder FC -> LSTM -> cabeca), com mascaramento de
    acoes invalidas (logits -1e8 antes do softmax/argmax).

Como BC e DAgger compartilham exatamente a mesma rede e o mesmo formato de dados,
estes componentes ficam em um modulo unico, importado pelos dois.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

# --- Constantes ---------------------------------------------------------------
OBS_DIM       = 176   # FarmGym v4 (env.py e a fonte; protocolo.py espelha)
N_ACOES       = 14    # FarmGym v4: 'observar' saiu
LEN_SEQUENCIA = 32   # janela temporal para treino do LSTM

# Hiperparâmetros de rede
DIM_ENCODER  = 256
DIM_LSTM     = 512
N_CAMADAS    = 2


# --- Dataset ------------------------------------------------------------------
# ── Normalização da observação (D-28, 06/09/2026) ───────────────────────────────────────────────
# As estatísticas saem das demonstrações, que são só do regime de treino (in-dist). Três dimensões da observação
# (estresse de calor, seca e podridão de raiz) têm desvio EXATAMENTE zero nessas demonstrações, porque os eventos
# correspondentes não ocorrem in-dist. Sem piso, `1/(0 + 1e-8) = 1e8`: quando a dimensão sai de zero fora da
# distribuição, a entrada do encoder explode. Medido no ambiente v3 com o professor: |z| = 6,9e7 no regime seco
# (dimensões de seca e calor) e 2,5e7 no úmido (podridão de raiz). Isso satura o encoder — Linear → GELU → LayerNorm
# devolve uma representação praticamente constante — e a política vira uma distribuição fixa. Ou seja, TODA medida
# fora da distribuição feita antes desta correção mede uma rede saturada, não generalização.
# O piso e o recorte são a prática padrão (VecNormalize) e quase não tocam o regime de treino: in-dist só 0,02 % dos
# valores normalizados passam de |10|.
PISO_STD_OBS = 1e-2
CLIP_OBS_NORM = 10.0


def estatisticas_obs(observacoes) -> Tuple[np.ndarray, np.ndarray]:
    """Média e desvio (com piso) das observações. Único lugar onde o desvio da normalização é calculado."""
    a = np.asarray(observacoes, dtype=np.float32)
    return a.mean(axis=0).astype(np.float32), np.maximum(a.std(axis=0), PISO_STD_OBS).astype(np.float32)


def normalizar_obs(obs, media, std):
    """`(obs − média) / max(desvio, PISO)`, recortado em ±CLIP. Aceita NumPy e torch. O piso é reaplicado aqui de
    propósito: assim um checkpoint gravado antes da D-28, com desvio zero em alguma dimensão, também fica são."""
    if torch.is_tensor(obs):
        med = media if torch.is_tensor(media) else torch.as_tensor(media, device=obs.device, dtype=obs.dtype)
        dev = std if torch.is_tensor(std) else torch.as_tensor(std, device=obs.device, dtype=obs.dtype)
        return torch.clamp((obs - med) / torch.clamp(dev, min=PISO_STD_OBS), -CLIP_OBS_NORM, CLIP_OBS_NORM)
    return np.clip((np.asarray(obs, np.float32) - media) / np.maximum(std, PISO_STD_OBS),
                   -CLIP_OBS_NORM, CLIP_OBS_NORM).astype(np.float32)


class ConjuntoDadosBC(Dataset):
    """
    Organiza episódios em janelas de comprimento fixo para treino do LSTM.
    Nunca cruza fronteiras de episódio → hidden state zerado entre janelas.
    """
    def __init__(
        self,
        episodios: List[List[Dict]],
        obs_media: np.ndarray,
        obs_std: np.ndarray,
        len_seq: int = LEN_SEQUENCIA,
    ):
        self.obs_media = obs_media
        self.obs_std   = obs_std

        self.sequencias: List[Tuple] = []
        for ep in episodios:
            n = len(ep)
            inicios = list(range(0, n - len_seq + 1, len_seq))
            # A última janela é alinhada ao FIM do episódio quando n não é múltiplo de len_seq (06/09/2026): antes, os
            # n mod len_seq passos finais (13 de 365 com janela 32) nunca entravam no treino — justamente os dias em
            # que o oráculo para de plantar e só colhe (criticas.md, §4.1). A sobreposição parcial é intencional.
            if n >= len_seq and n % len_seq != 0:
                inicios.append(n - len_seq)
            for inicio in inicios:
                janela   = ep[inicio : inicio + len_seq]
                obs_seq  = np.array([t['obs']     for t in janela], dtype=np.float32)
                mask_seq = np.array([t['mascara'] for t in janela], dtype=np.float32)
                acao_seq = np.array([t['acao']    for t in janela], dtype=np.int64)
                self.sequencias.append((obs_seq, mask_seq, acao_seq))

    def __len__(self) -> int:
        return len(self.sequencias)

    def __getitem__(self, idx: int):
        obs_seq, mask_seq, acao_seq = self.sequencias[idx]
        obs_norm = normalizar_obs(obs_seq, self.obs_media, self.obs_std)
        return (
            torch.FloatTensor(obs_norm),
            torch.FloatTensor(mask_seq),
            torch.LongTensor(acao_seq),
        )


# --- Rede Política ------------------------------------------------------------
class RedeBCLSTM(nn.Module):
    """
    Política LSTM para Behavioral Cloning (compartilhada com o DAgger).

    Forward: (obs_seq, mask_seq, hidden?) → (logits_mascarados, nova_hidden)
      obs_seq:  (B, T, obs_dim)
      mask_seq: (B, T, n_acoes) — 1=válida, 0=inválida
    """
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_acoes: int = N_ACOES,
        dim_enc: int = DIM_ENCODER,
        dim_lstm: int = DIM_LSTM,
        n_camadas: int = N_CAMADAS,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, dim_enc),
            nn.GELU(),
            nn.LayerNorm(dim_enc),
            nn.Linear(dim_enc, dim_enc),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            dim_enc, dim_lstm, n_camadas,
            batch_first=True, dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim_lstm),
            nn.Linear(dim_lstm, dim_enc),
            nn.GELU(),
            nn.Linear(dim_enc, n_acoes),
        )
        self.dim_lstm  = dim_lstm
        self.n_camadas = n_camadas

    def forward(
        self,
        obs_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        B, T, _ = obs_seq.shape
        enc = self.encoder(obs_seq.view(B * T, -1)).view(B, T, -1)
        lstm_out, hidden = self.lstm(enc, hidden)
        logits = self.head(lstm_out)                           # (B, T, n_acoes)
        masked_logits = logits + (1.0 - mask_seq) * (-1e8)   # mascara ações inválidas
        return masked_logits, hidden

    def inicializar_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple:
        return (
            torch.zeros(self.n_camadas, batch_size, self.dim_lstm, device=device),
            torch.zeros(self.n_camadas, batch_size, self.dim_lstm, device=device),
        )
