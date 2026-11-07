"""ppo.py — braço A (PPO-IL) e braço C (PPO do zero) da Fase 3, pela interface única `treinar_rl` (D-19).

Desenho (experimentos/DESENHO_FASE3_BAKEOFF_RL.md §3.2):
  - atualização por lote de V episódios COMPLETOS (365 passos cada): todos os ambientes começam juntos no dia 0 e
    terminam juntos, então não há fronteira de episódio dentro do lote e o retorno é exato (sem bootstrap);
  - BPTT truncado: cada episódio vira 5 segmentos de 73 passos com o hidden do início do segmento guardado na coleta;
  - crítico aquecido por `k_aquec` atualizações (só a perda de valor, rollouts da política IL) antes de liberar o ator;
  - penalidade β_KL·KL(π_IL ‖ π) com β decaindo por atualização até β_min (β_min só vale se β_0 > 0); π_IL é o clone
    tal como treinado — a `temperatura_inicial` é aplicada ao ator DEPOIS de congelar a referência; recompensa escalada
    (÷ `escala`), vantagem normalizada por lote; entropia; clipe de gradiente; dropout zerado (modo train, exigido pelo
    cuDNN) para logp_old e logp_new coincidirem;
  - a cada bloco de `orcamento_passos/blocos` passos chama `callback(passos, agente)`; se ele devolver True, para (poda).
  - sem warm-start (`politica_il=None` ou hp['inicializacao'] = 'zero'): ator aleatório, β_KL = 0, k_aquec = 0, normalização das demos.
"""
import os
import json
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import protocolo as P

P.preparar_sys_path()
from politica import AtorCritico, AgenteRL, referencia_congelada, kl_referencia, desligar_dropout  # noqa: E402
from envs_lote import EnvLote, fabrica_env_padrao                               # noqa: E402
from wrapper import TransformadorRecompensa                                      # noqa: E402

HP_PADRAO: Dict = {
    'V': 12, 'len_seq': 73, 'lr': 5e-5, 'lr_critico': 3e-4, 'clip': 0.2, 'c_ent': 3e-3, 'c_v': 0.5,
    'gamma': 0.99, 'lambda_gae': 0.95, 'n_epocas': 4, 'minilote_seqs': 20,
    'beta_kl0': 0.1, 'beta_kl_decay': 0.98, 'beta_kl_min': 1e-3,
    'escala': 100.0, 'pbrs': False, 'k_aquec': 10, 'max_grad_norm': 0.5, 'blocos': 9, 'temperatura_inicial': 1.0,
    'inicializacao': 'professor',   # 'professor' = parte da política IL dada; 'zero' = ignora politica_il (D-21)
    'dim_lstm': 512, 'dim_enc': 256, 'n_camadas': 2,
}


def _normalizacao_das_demos(caminho: str):
    from rede_lstm import estatisticas_obs
    with open(caminho, encoding='utf-8') as f:
        raw = json.load(f)
    return estatisticas_obs([t['obs'] for ep in raw for t in ep])   # D-28: desvio com piso


def _gae(recompensas, valores, gamma, lam):
    """recompensas, valores: (T, V) de episódios completos (terminal em T-1). Devolve (vantagens, retornos)."""
    T, V = recompensas.shape
    adv = np.zeros((T, V), np.float32)
    ultimo = np.zeros(V, np.float32)
    for t in reversed(range(T)):
        v_prox = valores[t + 1] if t + 1 < T else np.zeros(V, np.float32)
        delta = recompensas[t] + gamma * v_prox - valores[t]
        ultimo = delta + gamma * lam * ultimo
        adv[t] = ultimo
    return adv, adv + valores


def treinar_rl(politica_il: Optional[str], hp: Optional[Dict], seed: int, orcamento_passos: int,
               callback: Optional[Callable] = None, fabrica_env: Callable = fabrica_env_padrao,
               log: Callable = print, dispositivo: Optional[str] = None, demos_para_normalizacao: Optional[str] = None) -> Dict:
    hp = {**HP_PADRAO, **(hp or {})}
    assert P.PASSOS_MAXIMOS % hp['len_seq'] == 0, 'len_seq deve dividir 365 (ex.: 73 ou 5)'
    t0 = time.time()
    P.fixar_sementes(seed)
    dev = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
    V, T, L = int(hp['V']), P.PASSOS_MAXIMOS, int(hp['len_seq'])
    n_seg = T // L

    modelo = AtorCritico(dim_enc=hp['dim_enc'], dim_lstm=hp['dim_lstm'], n_camadas=hp['n_camadas']).to(dev)
    if hp.get('inicializacao', 'professor') == 'zero':
        politica_il = None   # D-21: a inicialização é hiperparâmetro; 'zero' treina do zero mesmo com checkpoint disponível
    if politica_il:
        media, std = modelo.carregar_il(politica_il, dev, temperatura=1.0)
        ref = referencia_congelada(modelo.ator)                 # âncora = o clone do professor, antes da temperatura (06/09)
        modelo.aplicar_temperatura(float(hp['temperatura_inicial']))
        beta_kl = float(hp['beta_kl0']); k_aquec = int(hp['k_aquec'])
    else:
        media, std = _normalizacao_das_demos(demos_para_normalizacao or P.DEMOS_PADRAO)
        ref, beta_kl, k_aquec = None, 0.0, 0
    desligar_dropout(modelo)   # dropout 0 em modo train: logp_old e logp_new coincidem e o cuDNN faz o backward
    from rede_lstm import PISO_STD_OBS, CLIP_OBS_NORM
    media_t = torch.as_tensor(media, device=dev)
    std_t = torch.clamp(torch.as_tensor(std, device=dev), min=PISO_STD_OBS)   # D-28

    opt = torch.optim.Adam([{'params': modelo.parametros_ator(), 'lr': hp['lr']},
                            {'params': modelo.parametros_critico(), 'lr': hp['lr_critico']}])
    transf = TransformadorRecompensa(hp['escala'], hp['gamma'], hp['pbrs'])
    lote = EnvLote(V, fabrica_env, seed_base=int(seed) * 100_000)
    agente = AgenteRL(modelo.ator, media, std, dev)   # avaliação: selecionar_acao chama rede.eval() (sem dropout de qualquer forma)

    passos, atualizacao, curva, episodios_treino = 0, 0, [], []
    tamanho_bloco = max(1, int(orcamento_passos) // int(hp['blocos']))
    proximo_bloco = tamanho_bloco
    parar = False

    def coletar():
        """V episódios completos com a política atual (amostrada). Devolve tensores (T, V, ...) e hiddens por segmento."""
        obs_n = np.zeros((T, V, P.OBS_DIM), np.float32); masc = np.zeros((T, V, P.N_ACOES), np.float32)
        acoes = np.zeros((T, V), np.int64); logp = np.zeros((T, V), np.float32); val = np.zeros((T, V), np.float32)
        rec = np.zeros((T, V), np.float32); rec_bruta = np.zeros((T, V), np.float32)
        h_seg_a, h_seg_c = [], []
        obs, mk = lote.reset_todos()
        h_a, h_c = modelo.inicializar_hidden(V, dev)
        with torch.no_grad():
            for t in range(T):
                if t % L == 0:
                    h_seg_a.append((h_a[0].clone(), h_a[1].clone())); h_seg_c.append((h_c[0].clone(), h_c[1].clone()))
                o = torch.clamp((torch.as_tensor(obs, device=dev) - media_t) / std_t,
                                -CLIP_OBS_NORM, CLIP_OBS_NORM).unsqueeze(1)                       # (V,1,148); D-28
                m = torch.as_tensor(mk, device=dev).unsqueeze(1)
                lg, h_a = modelo.logits(o, m, h_a)
                vl, h_c = modelo.valor(o, h_c)
                dist = torch.distributions.Categorical(logits=lg.squeeze(1))
                a = dist.sample()
                obs_n[t] = o.squeeze(1).cpu().numpy(); masc[t] = mk; acoes[t] = a.cpu().numpy()
                logp[t] = dist.log_prob(a).cpu().numpy(); val[t] = vl.squeeze(1).cpu().numpy()
                obs_dep, mk_dep, r, fins, _ = lote.step(acoes[t])
                rec_bruta[t] = r
                rec[t] = transf(r, obs, obs_dep, fins)
                obs, mk = obs_dep, mk_dep
        assert fins.all(), 'os V episódios devem terminar juntos (horizonte fixo)'
        return obs_n, masc, acoes, logp, val, rec, rec_bruta, h_seg_a, h_seg_c

    def atualizar(dados, so_critico: bool):
        nonlocal beta_kl
        modelo.train()   # a avaliação (callback) deixa o ator em eval(); o cuDNN exige train() no backward (dropout já é 0)
        obs_n, masc, acoes, logp_old, val, rec, rec_bruta, h_seg_a, h_seg_c = dados
        adv, ret = _gae(rec, val, hp['gamma'], hp['lambda_gae'])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # segmentos (n_seg*V) de comprimento L, com o hidden do início de cada um
        def seg(x):  # (T, V, ...) -> (n_seg*V, L, ...)
            x = torch.as_tensor(x, device=dev)
            return x.reshape(n_seg, L, V, *x.shape[2:]).transpose(0, 1).reshape(L, n_seg * V, *x.shape[2:]).transpose(0, 1)
        S_obs, S_m, S_a, S_lp, S_adv, S_ret = seg(obs_n), seg(masc), seg(acoes), seg(logp_old), seg(adv), seg(ret)
        H_a = tuple(torch.cat([h[i] for h in h_seg_a], dim=1) for i in range(2))   # (camadas, n_seg*V, dim)
        H_c = tuple(torch.cat([h[i] for h in h_seg_c], dim=1) for i in range(2))
        n_total = n_seg * V
        estat = {'perda_valor': [], 'perda_pg': [], 'entropia': [], 'kl': [], 'clipfrac': []}
        for _ in range(int(hp['n_epocas'])):
            perm = torch.randperm(n_total, device=dev)
            for ini in range(0, n_total, int(hp['minilote_seqs'])):
                idx = perm[ini:ini + int(hp['minilote_seqs'])]
                ha = (H_a[0][:, idx].contiguous(), H_a[1][:, idx].contiguous()); hc = (H_c[0][:, idx].contiguous(), H_c[1][:, idx].contiguous())
                v_pred, _ = modelo.valor(S_obs[idx], hc)
                perda_v = 0.5 * F.mse_loss(v_pred, S_ret[idx])
                if so_critico:
                    perda = hp['c_v'] * perda_v
                else:
                    lg, _ = modelo.logits(S_obs[idx], S_m[idx], ha)
                    dist = torch.distributions.Categorical(logits=lg)
                    lp = dist.log_prob(S_a[idx])
                    razao = torch.exp(lp - S_lp[idx])
                    a_ = S_adv[idx]
                    perda_pg = -torch.min(razao * a_, razao.clamp(1 - hp['clip'], 1 + hp['clip']) * a_).mean()
                    ent = dist.entropy().mean()
                    perda = perda_pg + hp['c_v'] * perda_v - hp['c_ent'] * ent
                    if ref is not None and beta_kl > 0:
                        with torch.no_grad():
                            lg_ref, _ = ref(S_obs[idx], S_m[idx], ha)
                        kl = kl_referencia(lg_ref, lg).mean()
                        perda = perda + beta_kl * kl
                        estat['kl'].append(float(kl))
                    estat['perda_pg'].append(float(perda_pg)); estat['entropia'].append(float(ent))
                    estat['clipfrac'].append(float(((razao - 1).abs() > hp['clip']).float().mean()))
                opt.zero_grad(); perda.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), hp['max_grad_norm'])
                opt.step()
                estat['perda_valor'].append(float(perda_v))
        return {k: (float(np.mean(v)) if v else None) for k, v in estat.items()}

    # ── aquecimento do crítico ──
    for k in range(k_aquec):
        dados = coletar()
        passos += T * V
        eps = lote.coletar_concluidos(); episodios_treino.extend(eps)
        est = atualizar(dados, so_critico=True)
        log(f"  [aquec {k + 1}/{k_aquec}] passos {passos} | retorno bruto médio {np.mean([e['retorno'] for e in eps]):.0f} | perda_valor {est['perda_valor']:.4f}")
        curva.append({'passos': passos, 'fase': 'aquecimento', 'retorno_treino': float(np.mean([e['retorno'] for e in eps])), **est})

    # ── PPO ──
    while passos < orcamento_passos and not parar:
        dados = coletar()
        passos += T * V
        eps = lote.coletar_concluidos(); episodios_treino.extend(eps)
        est = atualizar(dados, so_critico=False)
        atualizacao += 1
        # o piso beta_kl_min só vale quando há penalidade (beta_kl0 > 0); com beta_kl0 = 0 (ablação "sem KL") o KL fica
        # desligado o treino inteiro — antes, o piso o religava na 2ª atualização (criticas.md, §5.4; 06/09/2026)
        beta_kl = max(hp['beta_kl_min'], beta_kl * hp['beta_kl_decay']) if (ref is not None and float(hp['beta_kl0']) > 0) else 0.0
        ret_treino = float(np.mean([e['retorno'] for e in eps]))
        curva.append({'passos': passos, 'fase': 'ppo', 'atualizacao': atualizacao, 'retorno_treino': ret_treino, 'beta_kl': beta_kl, **est})
        log(f"  [ppo {atualizacao}] passos {passos}/{orcamento_passos} | retorno treino {ret_treino:.0f} | pg {est['perda_pg']:.4f} "
            f"v {est['perda_valor']:.4f} ent {est['entropia']:.3f} kl {est['kl'] if est['kl'] is not None else 0:.4f} clip {est['clipfrac']:.2f} | β {beta_kl:.4f} | {time.time() - t0:.0f}s")
        if callback is not None and passos >= proximo_bloco:
            proximo_bloco += tamanho_bloco
            parar = bool(callback(passos, agente))
    return {'agente': agente, 'modelo': modelo, 'curva': curva, 'passos': passos, 'tempo_s': time.time() - t0, 'hp': hp,
            'episodios_treino': episodios_treino, 'podado': parar}
