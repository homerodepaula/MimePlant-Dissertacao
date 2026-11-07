"""
Agente de Behavioral Cloning (BC) com LSTM
==========================================

BC e a forma mais direta de Aprendizado por Imitacao: treinamento supervisionado
puro nas trajetorias do oraculo (BotPerfeito). A rede aprende a reproduzir a acao
do especialista para cada observacao, minimizando cross-entropy mascarada
(acoes invalidas recebem logit -1e8).

A classe AgenteBC reune a infraestrutura compartilhada do agente de imitacao
(rede, normalizacao, avaliacao, selecao de acao, persistencia) e o treino de BC
(treinar_bc). O DAgger (dagger.py) estende esta classe adicionando a coleta
iterativa de dados.

Arquitetura (em rede_lstm.py):
  obs(148) -> Encoder FC(256, GELU, LN) -> LSTM(512, 2 camadas) -> Head -> logits(15)

Uso:
  python bc.py                 # treina BC e avalia
  python bc.py --n-epocas 30
"""

import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# --- Paths --------------------------------------------------------------------
_DIR     = os.path.dirname(os.path.abspath(__file__))
_ENV_DIR = os.path.join(_DIR, '../../environment')
sys.path.insert(0, _ENV_DIR)
sys.path.insert(0, _DIR)
for _sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
             'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(_ENV_DIR, _sub))

from rede_lstm import (OBS_DIM, N_ACOES, LEN_SEQUENCIA, DIM_ENCODER, DIM_LSTM,
                       N_CAMADAS, ConjuntoDadosBC, RedeBCLSTM,
                       PISO_STD_OBS, CLIP_OBS_NORM, estatisticas_obs, normalizar_obs)


# --- Agente BC ----------------------------------------------------------------
class AgenteBC:
    """
    Agente de Imitation Learning por Behavioral Cloning.

    Uso típico:
        agente = AgenteBC()
        episodios, _ = agente.carregar_trajetorias('results/bot_perfeito/**/*.json')
        agente.calcular_normalizacao(episodios)
        agente.treinar_bc(episodios, n_epocas=50)
        agente.salvar('results/il_agent/il_bc.pt')
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_acoes: int = N_ACOES,
        dim_encoder: int = DIM_ENCODER,
        dim_lstm: int = DIM_LSTM,
        n_camadas: int = N_CAMADAS,
        dispositivo: Optional[str] = None,
    ):
        self.dispositivo = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rede = RedeBCLSTM(obs_dim, n_acoes, dim_encoder, dim_lstm, n_camadas)
        self.rede.to(self.dispositivo)
        self.obs_media = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std   = np.ones(obs_dim, dtype=np.float32)
        self.historico: Dict[str, List] = defaultdict(list)

        n_params = sum(p.numel() for p in self.rede.parameters())
        print(f"[rede] {n_params:,} parâmetros | dispositivo: {self.dispositivo}")

    # -- Dados -----------------------------------------------------------------

    def carregar_trajetorias(self, padrao_glob: str) -> Tuple[List[List[Dict]], str]:
        """Carrega o arquivo de trajetórias mais recente."""
        arquivos = sorted(glob.glob(padrao_glob, recursive=True))
        if not arquivos:
            raise FileNotFoundError(f"Nenhuma trajetória em: {padrao_glob}")
        caminho = arquivos[-1]
        with open(caminho, encoding='utf-8') as f:
            raw = json.load(f)
        episodios = []
        for ep_raw in raw:
            ep = [{
                'obs':     np.array(t['obs'],     dtype=np.float32),
                'acao':    int(t['acao']),
                'mascara': np.array(t['mascara'], dtype=np.float32),
            } for t in ep_raw]
            episodios.append(ep)
        n_trans = sum(len(e) for e in episodios)
        print(f"[dados] {len(episodios)} episódios, {n_trans} transições — {os.path.basename(os.path.dirname(caminho))}")
        return episodios, caminho

    def calcular_normalizacao(self, episodios: List[List[Dict]]):
        """Calcula média/std por feature em todas as observações."""
        todas = np.stack([t['obs'] for ep in episodios for t in ep])
        self.obs_media, self.obs_std = estatisticas_obs(todas)   # D-28: desvio com piso
        n_piso = int((todas.std(axis=0) < PISO_STD_OBS).sum())
        print(f"[norm] obs_dim={todas.shape[1]} | média global={self.obs_media.mean():.3f} "
              f"| std global={self.obs_std.mean():.3f} | {n_piso} dims no piso ({PISO_STD_OBS})")

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return normalizar_obs(obs, self.obs_media, self.obs_std)

    # -- Behavioral Cloning ----------------------------------------------------

    def treinar_bc(
        self,
        episodios: List[List[Dict]],
        n_epocas: int = 50,
        batch_size: int = 64,
        lr: float = 3e-4,
        len_seq: int = LEN_SEQUENCIA,
        env_aval=None,
        n_eps_aval: int = 10,
        freq_aval: int = 10,
    ) -> Dict:
        """
        Behavioral Cloning por supervisão direta nas trajetórias do bot.

        Loss = cross-entropy(logits_mascarados, acao_especialista)
        """
        dataset = ConjuntoDadosBC(episodios, self.obs_media, self.obs_std, len_seq)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        n_seqs  = len(dataset)

        otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr, weight_decay=1e-5)
        agendador  = torch.optim.lr_scheduler.CosineAnnealingLR(
            otimizador, T_max=n_epocas, eta_min=lr * 0.01,
        )

        _sep = '-' * 62
        print(f"\n{_sep}")
        print(f"  BEHAVIORAL CLONING — {n_epocas} épocas")
        print(f"  Sequências: {n_seqs} × {len_seq} passos | batch: {batch_size} | LR: {lr:.0e}")
        print(_sep)

        t0_total = time.time()
        hist: Dict[str, List] = defaultdict(list)

        for epoca in range(1, n_epocas + 1):
            t0_ep = time.time()
            self.rede.train()
            perdas, acertos, total = [], 0, 0

            for obs_seq, mask_seq, acao_seq in loader:
                obs_seq  = obs_seq.to(self.dispositivo)
                mask_seq = mask_seq.to(self.dispositivo)
                acao_seq = acao_seq.to(self.dispositivo)

                hidden = self.rede.inicializar_hidden(obs_seq.size(0), self.dispositivo)
                logits, _ = self.rede(obs_seq, mask_seq, hidden)

                B, T, A = logits.shape
                loss = F.cross_entropy(logits.view(B * T, A), acao_seq.view(B * T))

                otimizador.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.rede.parameters(), 1.0)
                otimizador.step()

                perdas.append(loss.item())
                acertos += (logits.argmax(-1) == acao_seq).sum().item()
                total   += B * T

            agendador.step()

            perda_m  = float(np.mean(perdas))
            acuracia = acertos / total
            tempo_ep = time.time() - t0_ep
            lr_atual = agendador.get_last_lr()[0]

            hist['perda'].append(perda_m)
            hist['acuracia'].append(acuracia)
            hist['tempo_ep'].append(tempo_ep)
            self.historico['bc_perda'].append(perda_m)
            self.historico['bc_acuracia'].append(acuracia)

            if epoca % 5 == 0 or epoca == 1:
                t_total = time.time() - t0_total
                print(f"  Época {epoca:3d}/{n_epocas} | "
                      f"Perda: {perda_m:.4f} | "
                      f"Acur: {acuracia:.1%} | "
                      f"LR: {lr_atual:.2e} | "
                      f"Ep: {tempo_ep:.1f}s | Total: {t_total:.0f}s")

            if env_aval is not None and epoca % freq_aval == 0:
                stats = self.avaliar(env_aval, n_eps_aval)
                hist['aval_recompensa'].append(stats['media'])
                hist['aval_epoca'].append(epoca)
                self.historico['bc_aval_recompensa'].append(stats['media'])
                print(f"    → avaliação: {stats['media']:.1f} ± {stats['std']:.1f}")

        t_bc = time.time() - t0_total
        print(f"\n  BC concluído em {t_bc:.1f}s ({t_bc/60:.1f} min)")
        print(f"  Acurácia final: {hist['acuracia'][-1]:.1%} | Perda: {hist['perda'][-1]:.4f}")
        return dict(hist)

    # -- Avaliação --------------------------------------------------------------

    def avaliar(self, env_wrapped, n_episodios: int = 20) -> Dict:
        """Avalia o agente no ambiente real, mantendo hidden state por episódio."""
        self.rede.eval()
        recompensas = []

        for _ in range(n_episodios):
            obs, info = env_wrapped.reset()
            hidden = self.rede.inicializar_hidden(1, self.dispositivo)
            rec_ep = 0.0
            done   = False

            while not done:
                mask = info.get('mascara_acoes', np.ones(N_ACOES, dtype=np.float32))
                obs_t  = torch.FloatTensor(self._norm(obs)).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.dispositivo)

                with torch.no_grad():
                    logits, hidden = self.rede(obs_t, mask_t, hidden)

                acao = int(logits.squeeze().argmax().item())
                obs, r, done_f, trunc, info = env_wrapped.step(acao)
                rec_ep += r
                done    = done_f or trunc

            recompensas.append(rec_ep)

        return {
            'media': float(np.mean(recompensas)),
            'std':   float(np.std(recompensas)),
            'min':   float(np.min(recompensas)),
            'max':   float(np.max(recompensas)),
        }

    def selecionar_acao(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        hidden: Optional[Tuple],
        temperatura: float = 0.0,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, Tuple]:
        """Interface de inferência: retorna (acao, nova_hidden).

        Args:
            temperatura: softmax temperature τ. Se τ>0, amostra de softmax(logits/τ)
                          sobre ações válidas. τ=0 ⇒ argmax (determinístico).
            rng: gerador NumPy próprio para a amostragem (D-26, 06/09/2026). Com `rng`, a amostragem não
                 consome o RNG global do torch nem `np.random`/`random` (que o ambiente usa): a avaliação
                 fica determinística dada a semente do episódio e o fluxo do ambiente não muda com a política.
                 Sem `rng` (coleta do DAgger), amostra pelo torch como antes.
            epsilon: ε-greedy. Se ε>0, com prob ε escolhe ação UNIFORME entre as
                      válidas; caso contrário usa temperatura/argmax.

        Os dois mecanismos são compostos: ε-greedy é avaliado primeiro
        (com prob ε escolhe uniforme). Se NÃO entrar no ε-branch, então
        avalia τ (sample softmax ou argmax).
        """
        self.rede.eval()
        # 1) ε-greedy: com prob ε, ação UNIFORME entre as válidas
        if epsilon > 0.0 and np.random.random() < epsilon:
            validas = np.where(mask > 0.5)[0]
            acao_eps = int(np.random.choice(validas)) if len(validas) else 14
            # ainda precisamos atualizar hidden para a próxima chamada
            obs_t  = torch.FloatTensor(self._norm(obs)).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
            with torch.no_grad():
                _, hidden = self.rede(obs_t, mask_t, hidden)
            return acao_eps, hidden

        # 2) Forward da rede + mascaramento
        obs_t  = torch.FloatTensor(self._norm(obs)).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
        with torch.no_grad():
            logits, hidden = self.rede(obs_t, mask_t, hidden)
        logits = logits.squeeze()                                # (n_acoes,)
        logits = logits.masked_fill(mask_t.squeeze() < 0.5, -1e8)  # garante validade

        # 3) Softmax com temperatura ou argmax
        if temperatura > 0.0:
            probs = torch.nn.functional.softmax(logits / temperatura, dim=-1)
            if rng is not None:
                p = probs.detach().cpu().numpy().astype(np.float64)
                p = p / p.sum()
                acao = int(rng.choice(len(p), p=p))
            else:
                acao = int(torch.distributions.Categorical(probs).sample().item())
        else:
            acao = int(logits.argmax().item())
        return acao, hidden

    def inicializar_hidden(self) -> Tuple:
        return self.rede.inicializar_hidden(1, self.dispositivo)

    # -- Persistência -----------------------------------------------------------

    def salvar(self, caminho: str):
        os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
        torch.save({
            'estado_rede': self.rede.state_dict(),
            'obs_media':   self.obs_media,
            'obs_std':     self.obs_std,
            'historico':   dict(self.historico),
            # v2: dimensões REAIS da rede (antes gravava as constantes do módulo, erradas para redes de outro tamanho)
            'config': dict(
                obs_dim=self.rede.encoder[0].in_features, n_acoes=self.rede.head[-1].out_features,
                dim_enc=self.rede.encoder[0].out_features, dim_lstm=self.rede.lstm.hidden_size,
                n_camadas=self.rede.lstm.num_layers,
            ),
        }, caminho)
        print(f"[salvo] {caminho}")

    def carregar(self, caminho: str):
        ckpt = torch.load(caminho, map_location=self.dispositivo, weights_only=False)
        self.rede.load_state_dict(ckpt['estado_rede'])
        self.obs_media = ckpt['obs_media']
        self.obs_std   = ckpt['obs_std']
        if 'historico' in ckpt:
            self.historico = defaultdict(list, ckpt['historico'])
        print(f"[carregado] {caminho}")


# --- Script principal (demo BC) -----------------------------------------------

if __name__ == '__main__':
    import argparse
    from env import AmbienteFazendaGym
    from mascara_acoes import InvolucroMascaraAcoes

    ap = argparse.ArgumentParser(description='Agente IL — Behavioral Cloning')
    ap.add_argument('--n-epocas', type=int, default=50)
    ap.add_argument('--n-aval',   type=int, default=10, help='Episódios de avaliação')
    args = ap.parse_args()

    env_wrapped = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=365))
    agente = AgenteBC()

    padrao = os.path.join(_DIR, '../../results/bot_perfeito/**/trajetorias_especialista.json')
    episodios, _ = agente.carregar_trajetorias(padrao)
    agente.calcular_normalizacao(episodios)

    print(f"\n[início] {time.strftime('%H:%M:%S')}")
    agente.treinar_bc(
        episodios,
        n_epocas=args.n_epocas,
        env_aval=env_wrapped,
        n_eps_aval=args.n_aval,
        freq_aval=10,
    )

    stats_bc = agente.avaliar(env_wrapped, args.n_aval * 2)
    print(f"\n[BC] avaliação final: {stats_bc['media']:.1f} ± {stats_bc['std']:.1f}")

    dir_saida = os.path.join(_DIR, '../../results/il_agent')
    agente.salvar(os.path.join(dir_saida, 'il_bc.pt'))
