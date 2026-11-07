"""politica.py — ator-crítico recorrente da fase de RL (Fase 3, D-19).

Ator = `RedeBCLSTM` (a mesma rede do IL; logits mascarados), carregada do checkpoint do warm-start quando existe.
Crítico = tronco próprio (encoder + LSTM, cópia dos pesos do IL) + cabeça de valor `LayerNorm → Linear → GELU → Linear(1)`
com a última camada zerada (V(s) = 0 no início; o aquecimento do crítico ajusta antes de liberar o ator). O crítico não
compartilha pesos com o ator: o gradiente de valor não toca a política.

`AgenteRL` embrulha o ator na interface do IL (`.rede`, `.dispositivo`, `.selecionar_acao`, `.salvar`/`.carregar`,
`obs_media`/`obs_std`), então `avaliacao.PoliticaRede(AgenteRL)` funciona igual ao BC/DAgger e o checkpoint tem o mesmo formato.
"""
import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import protocolo as P

P.preparar_sys_path()
from rede_lstm import RedeBCLSTM, OBS_DIM, N_ACOES, DIM_ENCODER, DIM_LSTM, N_CAMADAS  # noqa: E402
from bc import AgenteBC  # noqa: E402


class AtorCritico(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_acoes: int = N_ACOES, dim_enc: int = DIM_ENCODER,
                 dim_lstm: int = DIM_LSTM, n_camadas: int = N_CAMADAS):
        super().__init__()
        self.ator = RedeBCLSTM(obs_dim, n_acoes, dim_enc, dim_lstm, n_camadas)
        self.critico_tronco = RedeBCLSTM(obs_dim, n_acoes, dim_enc, dim_lstm, n_camadas)   # a cabeça de ação dele não é usada
        self.cabeca_valor = nn.Sequential(nn.LayerNorm(dim_lstm), nn.Linear(dim_lstm, dim_enc), nn.GELU(), nn.Linear(dim_enc, 1))
        nn.init.zeros_(self.cabeca_valor[-1].weight)
        nn.init.zeros_(self.cabeca_valor[-1].bias)
        self.dim_lstm, self.n_camadas = dim_lstm, n_camadas

    # ── carga do warm-start ──
    def carregar_il(self, caminho: str, dispositivo: str, temperatura: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Copia os pesos do IL para o ator e para o tronco do crítico; devolve (obs_media, obs_std).

        `temperatura` > 1 suaviza a política clonada ao carregá-la (ver `aplicar_temperatura`). Em `ppo.treinar_rl` a
        carga é feita com τ = 1, a referência do KL é congelada a partir do clone tal como treinado, e só depois a
        temperatura é aplicada ao ator (06/09/2026) — assim KL(π_IL ‖ π) é de fato em relação ao professor clonado, e
        não à sua versão suavizada."""
        ckpt = torch.load(caminho, map_location=dispositivo, weights_only=False)
        self.ator.load_state_dict(ckpt['estado_rede'])
        self.critico_tronco.load_state_dict(ckpt['estado_rede'])
        self.aplicar_temperatura(temperatura)
        return np.asarray(ckpt['obs_media'], np.float32), np.asarray(ckpt['obs_std'], np.float32)

    def aplicar_temperatura(self, temperatura: float) -> None:
        """Divide os logits do ator por τ (escala a última camada da cabeça por 1/τ). O argmax não muda; a entropia
        sobe, e o PPO passa a explorar em vez de repetir o professor. Hiperparâmetro `temperatura_inicial`; 1,0 = clone."""
        if float(temperatura) != 1.0:
            with torch.no_grad():
                self.ator.head[-1].weight.div_(float(temperatura))
                self.ator.head[-1].bias.div_(float(temperatura))

    # ── passes ──
    def logits(self, obs_seq, mask_seq, hidden=None):
        return self.ator(obs_seq, mask_seq, hidden)

    def valor(self, obs_seq, hidden=None):
        B, T, _ = obs_seq.shape
        enc = self.critico_tronco.encoder(obs_seq.reshape(B * T, -1)).reshape(B, T, -1)
        saida, hidden = self.critico_tronco.lstm(enc, hidden)
        return self.cabeca_valor(saida).squeeze(-1), hidden

    def inicializar_hidden(self, batch: int, dispositivo: str):
        return (self.ator.inicializar_hidden(batch, dispositivo), self.critico_tronco.inicializar_hidden(batch, dispositivo))

    def parametros_ator(self):
        return list(self.ator.parameters())

    def parametros_critico(self):
        return list(self.critico_tronco.parameters()) + list(self.cabeca_valor.parameters())


def desligar_dropout(modulo: nn.Module) -> None:
    """Zera o dropout das LSTMs mantendo o modo de treino (o cuDNN só faz o backward de RNN em modo train;
    eval() desligaria o dropout mas quebraria o backward na GPU). LayerNorm/GELU não dependem do modo."""
    for m in modulo.modules():
        if isinstance(m, nn.LSTM):
            m.dropout = 0.0
    modulo.train()


def referencia_congelada(ator: RedeBCLSTM) -> RedeBCLSTM:
    """Cópia congelada da política IL para a penalidade KL(π_IL ‖ π)."""
    ref = copy.deepcopy(ator)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref


def kl_referencia(logits_ref: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(π_ref ‖ π) por posição, com logits já mascarados (as ações inválidas têm prob. 0 nos dois)."""
    logp_ref = torch.log_softmax(logits_ref, -1)
    logp = torch.log_softmax(logits, -1)
    p_ref = logp_ref.exp()
    return (p_ref * (logp_ref - logp)).sum(-1)


class AgenteRL(AgenteBC):
    """Interface do IL sobre o ator de RL (avaliação, salvar/carregar no mesmo formato)."""

    def __init__(self, ator: RedeBCLSTM, obs_media: np.ndarray, obs_std: np.ndarray, dispositivo: Optional[str] = None,
                 dim_lstm: int = DIM_LSTM):
        self.dispositivo = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rede = ator.to(self.dispositivo)
        self.obs_media = np.asarray(obs_media, np.float32)
        self.obs_std = np.asarray(obs_std, np.float32)
        from collections import defaultdict
        self.historico = defaultdict(list)

    @staticmethod
    def carregar_de(caminho: str, dispositivo: Optional[str] = None, dim_lstm: int = DIM_LSTM) -> 'AgenteRL':
        dispositivo = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(caminho, map_location=dispositivo, weights_only=False)
        cfg = ckpt.get('config', {})
        ator = RedeBCLSTM(cfg.get('obs_dim', OBS_DIM), cfg.get('n_acoes', N_ACOES), cfg.get('dim_enc', DIM_ENCODER),
                          cfg.get('dim_lstm', dim_lstm), cfg.get('n_camadas', N_CAMADAS))
        ator.load_state_dict(ckpt['estado_rede'])
        return AgenteRL(ator, ckpt['obs_media'], ckpt['obs_std'], dispositivo)
