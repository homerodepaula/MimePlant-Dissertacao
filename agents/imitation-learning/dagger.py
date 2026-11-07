"""
Agente DAgger (Dataset Aggregation) com LSTM
============================================

DAgger estende o Behavioral Cloning: parte do modelo treinado por BC e, em
rodadas iterativas, coleta dados nos estados que o PROPRIO agente visita,
rotulados pelo oraculo (BotPerfeito). Os dados sao agregados ao conjunto de
treino e a rede e refinada. Isso trata o covariate shift (acumulo de erro) que
o BC puro nao cobre.

A classe AgenteDAgger herda toda a infraestrutura de AgenteBC (bc.py) — rede,
normalizacao, treino BC, avaliacao, selecao de acao, persistencia — e adiciona:
  - _coletar_dagger : executa o agente e rotula os estados com o oraculo
  - treinar_dagger  : laco de rodadas (coleta -> agrega -> fine-tuning)

Uso:
  python dagger.py             # BC + DAgger
  python dagger.py --so-bc     # so Behavioral Cloning
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

# Temperatura da coleta do DAgger (auditoria de 06/09). Amostrar com tau = 1 e a MESMA decodificacao com que o agente e
# avaliado (D-26) e a unica fonte de exploracao que sobrou depois que o professor virou deterministico (D-27): com
# argmax, e com beta indo a zero, cada rodada do DAgger revisita quase os mesmos estados, entao ele deixa de cobrir a
# distribuicao que a politica realmente induz quando e usada. Coletar com argmax e avaliar amostrando tambem seria um
# descasamento entre treino e uso.
TAU_COLETA = 1.0
from typing import List, Dict, Tuple, Optional

# --- Paths --------------------------------------------------------------------
_DIR     = os.path.dirname(os.path.abspath(__file__))
_ENV_DIR = os.path.join(_DIR, '../../environment')
sys.path.insert(0, _ENV_DIR)
sys.path.insert(0, _DIR)
for _sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
             'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(_ENV_DIR, _sub))

from rede_lstm import N_ACOES, LEN_SEQUENCIA, ConjuntoDadosBC
from bc import AgenteBC


# --- Agente DAgger ------------------------------------------------------------
class AgenteDAgger(AgenteBC):
    """
    Agente de Imitation Learning com Behavioral Cloning + DAgger.

    Herda de AgenteBC (toda a infra + treinar_bc) e adiciona a fase DAgger.

    Uso típico:
        agente = AgenteDAgger()
        episodios, _ = agente.carregar_trajetorias('results/bot_perfeito/**/*.json')
        agente.calcular_normalizacao(episodios)
        agente.treinar_bc(episodios, n_epocas=50)                    # fase 1 (herdada)
        agente.treinar_dagger(episodios, env_wrapped, bot, n_rounds=5)  # fase 2
        agente.salvar('results/il_agent/il_dagger.pt')
    """

    @staticmethod
    def _env_bruto(env_wrapped) -> 'AmbienteFazendaGym':
        """Retorna o AmbienteFazendaGym desembrulhado para o bot."""
        env = env_wrapped
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _coletar_dagger(
        self,
        env_wrapped,
        bot: 'BotPerfeito',
        n_passos: int,
        beta: float = 0.0,
        temperatura: float = 0.0,
        epsilon: float = 0.0,
    ) -> List[List[Dict]]:
        """
        Coleta n_passos transições para DAgger.

        O agente executa no ambiente (com prob 1-beta).
        O bot SEMPRE fornece o label correto (oráculo).
        beta=0.5 → agente e bot compartilham execução (início do treino)
        beta=0.0 → só o agente executa (DAgger padrão)

        Exploração do agente (apenas durante coleta DAgger):
          temperatura > 0 → amostra de softmax(logits/τ) sobre ações válidas
          epsilon > 0     → ε-greedy: ação uniforme entre válidas com prob ε

        Retorna lista de episódios (listas de transições) para uso no dataset.
        """
        self.rede.eval()
        env_raw = self._env_bruto(env_wrapped)

        episodios: List[List[Dict]] = []
        ep_atual:  List[Dict] = []

        obs, info = env_wrapped.reset()
        bot.resetar_episodio()
        hidden = self.rede.inicializar_hidden(1, self.dispositivo)
        passos = 0

        while passos < n_passos:
            mask = info.get('mascara_acoes', np.ones(N_ACOES, dtype=np.float32))

            # Bot rotula o estado atual
            acao_bot = bot.selecionar_acao(env_raw)

            # Agente computa sua própria ação (com possível exploração)
            acao_agente, hidden = self.selecionar_acao(
                obs, mask, hidden,
                temperatura=temperatura,
                epsilon=epsilon,
            )

            # Executa: bot ou agente?
            acao_exec = acao_bot if np.random.random() < beta else acao_agente

            # Salva transição com label do bot (oráculo)
            ep_atual.append({
                'obs':     obs.copy(),
                'acao':    acao_bot,      # label do especialista
                'mascara': mask.copy(),
            })
            passos += 1

            obs, _, done, trunc, info = env_wrapped.step(acao_exec)

            if done or trunc:
                if ep_atual:
                    episodios.append(ep_atual)
                ep_atual = []
                obs, info = env_wrapped.reset()
                bot.resetar_episodio()
                hidden = self.rede.inicializar_hidden(1, self.dispositivo)

        if ep_atual:
            episodios.append(ep_atual)

        return episodios

    def treinar_dagger(
        self,
        episodios_base: List[List[Dict]],
        env_wrapped,
        bot: 'BotPerfeito',
        n_rounds: int = 5,
        passos_por_round: int = 3650,
        epocas_por_round: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        beta_inicial: float = 0.5,
        beta_final: float = 0.0,
        env_aval=None,
        n_eps_aval: int = 10,
        len_seq: int = LEN_SEQUENCIA,
        callback_rodada=None,
    ) -> Dict:
        """
        DAgger — agrega dados do agente rotulados pelo bot iterativamente.

        A cada round:
          1. Agente executa no ambiente (parcialmente guiado pelo bot via beta)
          2. Bot rotula cada estado visitado com ação ótima
          3. Dados são agregados ao conjunto de treino
          4. Fine-tuning da rede nas transições agregadas
          5. (v2) `callback_rodada(rodada, agente)`, se dado: o chamador avalia em VALIDAÇÃO e guarda o
             melhor checkpoint (escolher a melhor rodada, não a última). `len_seq` é a janela do LSTM.
        """
        _sep = '-' * 62
        print(f"\n{_sep}")
        print(f"  DAGGER — {n_rounds} rounds × {passos_por_round} passos/round")
        print(f"  Épocas/round: {epocas_por_round} | LR: {lr:.0e} | beta: {beta_inicial:.1f}→{beta_final:.1f}")
        print(_sep)

        t0_total = time.time()
        hist: Dict[str, List] = defaultdict(list)

        # Dataset acumulado: começa com dados do especialista
        episodios_agg = list(episodios_base)

        for rnd in range(1, n_rounds + 1):
            beta = (beta_inicial + (beta_final - beta_inicial) * (rnd - 1) / max(n_rounds - 1, 1))
            t0_rnd = time.time()

            print(f"\n  -- Round {rnd}/{n_rounds} (beta={beta:.2f}) --")

            # 1. Coleta
            t0_col = time.time()
            novos_eps = self._coletar_dagger(env_wrapped, bot, passos_por_round, beta=beta, temperatura=TAU_COLETA)
            n_novos   = sum(len(e) for e in novos_eps)
            episodios_agg.extend(novos_eps)
            n_total = sum(len(e) for e in episodios_agg)
            print(f"     Coleta: {n_novos} transições em {time.time()-t0_col:.1f}s | "
                  f"acumulado: {n_total}")

            # 2. Dataset e DataLoader sobre dados agregados
            dataset = ConjuntoDadosBC(episodios_agg, self.obs_media, self.obs_std, len_seq)
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr, weight_decay=1e-5)

            # 3. Fine-tuning
            t0_treino = time.time()
            for epoca in range(1, epocas_por_round + 1):
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

            perda_m  = float(np.mean(perdas))
            acuracia = acertos / total
            t_treino = time.time() - t0_treino
            print(f"     Treino ({epocas_por_round} épocas): perda={perda_m:.4f} "
                  f"acur={acuracia:.1%} em {t_treino:.1f}s")
            hist[f'r{rnd}_perda'].append(perda_m)
            hist[f'r{rnd}_acuracia'].append(acuracia)
            self.historico['dagger_perda'].append(perda_m)
            self.historico['dagger_acuracia'].append(acuracia)

            # 4. Callback do chamador (v2) e avaliação opcional do round
            if callback_rodada is not None:
                callback_rodada(rnd, self)
            if env_aval is not None:
                stats = self.avaliar(env_aval, n_eps_aval)
                hist[f'r{rnd}_aval'] = stats['media']
                self.historico['dagger_aval'].append(stats['media'])
                self.historico['dagger_aval_round'].append(rnd)
                t_rnd = time.time() - t0_rnd
                print(f"     Avaliação: {stats['media']:.1f} ± {stats['std']:.1f} "
                      f"| round em {t_rnd:.0f}s")

        t_dag = time.time() - t0_total
        print(f"\n  DAgger concluído em {t_dag:.1f}s ({t_dag/60:.1f} min)")
        return dict(hist)


# --- Script principal (demo BC + DAgger) --------------------------------------

def _baseline_random(env_wrapped, n_episodios: int = 10) -> Dict:
    recompensas = []
    for _ in range(n_episodios):
        obs, info = env_wrapped.reset()
        total = 0.0
        done  = False
        while not done:
            acao = env_wrapped.action_space.sample()
            obs, r, done_f, trunc, info = env_wrapped.step(acao)
            total += r
            done   = done_f or trunc
        recompensas.append(total)
    return {'media': float(np.mean(recompensas)), 'std': float(np.std(recompensas))}


def _resumo_final(stats_bc, stats_dag, stats_rnd):
    _sep = '=' * 62
    print(f"\n{_sep}")
    print(f"  RESUMO FINAL")
    print(_sep)
    print(f"  IL após BC                 : {stats_bc['media']:>8.1f} ± {stats_bc['std']:.1f}")
    if stats_dag is not None:
        print(f"  IL após DAgger             : {stats_dag['media']:>8.1f} ± {stats_dag['std']:.1f}")
    print(f"  Random        (referência) : {stats_rnd['media']:>8.1f} ± {stats_rnd['std']:.1f}")
    print(_sep)


if __name__ == '__main__':
    import argparse
    from env import AmbienteFazendaGym
    from mascara_acoes import InvolucroMascaraAcoes
    from bot_perfeito import BotPerfeito

    ap = argparse.ArgumentParser(description='Agente IL (BC + DAgger)')
    ap.add_argument('--so-bc',           action='store_true', help='Só BC, sem DAgger')
    ap.add_argument('--n-epocas-bc',     type=int, default=50)
    ap.add_argument('--n-rounds-dagger', type=int, default=5)
    ap.add_argument('--passos-dagger',   type=int, default=3650, help='Passos por round DAgger')
    ap.add_argument('--epocas-dagger',   type=int, default=10,   help='Épocas de fine-tuning/round')
    ap.add_argument('--n-aval',          type=int, default=10,   help='Episódios de avaliação')
    ap.add_argument('--carregar',        type=str, default=None,  help='Checkpoint .pt para continuar')
    args = ap.parse_args()

    # Ambiente e wrappers
    env_base    = AmbienteFazendaGym(passos_maximos=365)
    env_wrapped = InvolucroMascaraAcoes(env_base)
    # D-27: o professor e o oraculo SEM pausas (p=0), o mesmo das demos. A cadeia oficial usa
    # treinar_warmstart_il.py (que constroi o professor por protocolo.novo_professor()); este runner e avulso.
    bot         = BotPerfeito(imperfeicao=0.0, lazy_steps=8, seed=12345)

    # Agente
    agente = AgenteDAgger()
    if args.carregar:
        agente.carregar(args.carregar)

    # Trajetórias
    padrao = os.path.join(_DIR, '../../results/bot_perfeito/**/trajetorias_especialista.json')
    episodios, _ = agente.carregar_trajetorias(padrao)
    agente.calcular_normalizacao(episodios)

    dir_saida = os.path.join(_DIR, '../../results/il_agent')
    os.makedirs(dir_saida, exist_ok=True)

    t_inicio = time.time()
    print(f"\n[início] {time.strftime('%H:%M:%S')}")

    # -- Fase 1: BC (herdada de AgenteBC) --------------------------------------
    agente.treinar_bc(
        episodios,
        n_epocas=args.n_epocas_bc,
        env_aval=env_wrapped,
        n_eps_aval=args.n_aval,
        freq_aval=10,
    )

    stats_bc = agente.avaliar(env_wrapped, args.n_aval * 2)
    print(f"\n[BC] avaliação final: {stats_bc['media']:.1f} ± {stats_bc['std']:.1f}")
    agente.salvar(os.path.join(dir_saida, 'il_bc.pt'))

    stats_dag = None

    # -- Fase 2: DAgger --------------------------------------------------------
    if not args.so_bc:
        agente.treinar_dagger(
            episodios,
            env_wrapped,
            bot,
            n_rounds=args.n_rounds_dagger,
            passos_por_round=args.passos_dagger,
            epocas_por_round=args.epocas_dagger,
            env_aval=env_wrapped,
            n_eps_aval=args.n_aval,
        )
        stats_dag = agente.avaliar(env_wrapped, args.n_aval * 2)
        print(f"\n[DAgger] avaliação final: {stats_dag['media']:.1f} ± {stats_dag['std']:.1f}")
        agente.salvar(os.path.join(dir_saida, 'il_dagger.pt'))

    # -- Baseline random -------------------------------------------------------
    print("\n[random] calculando baseline...")
    stats_rnd = _baseline_random(env_wrapped, args.n_aval)

    t_total = time.time() - t_inicio
    print(f"\n[tempo total] {t_total:.0f}s ({t_total/60:.1f} min)")

    _resumo_final(stats_bc, stats_dag, stats_rnd)
