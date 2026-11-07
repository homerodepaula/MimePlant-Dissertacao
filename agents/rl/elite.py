"""elite.py — braço B: auto-imitação de elite (MimeCustom v7, reescrito sobre a interface única `treinar_rl`; D-19).

Ideia (v7): partir da política IL; coletar rollouts próprios com exploração (softmax/τ, ε-greedy, anti-pausa opcional);
manter um buffer de ELITE (top-E episódios por pontuação = retorno − pause_pen × nº de pausas); treinar por clonagem
comportamental sobre janelas da elite + uma fração de janelas das demos + âncora KL(π_IL ‖ π); "desconectar" (decair
demos e KL) quando a elite supera o nível da própria política IL, medido nos rollouts iniciais (na v7 era um número fixo).
Correções v2: nenhuma avaliação interna em 999+i (o harness avalia em validação-rollback pelo `callback`); `pause_pen`
= 0 por padrão (na v2 pausar já custa retorno; é hiperparâmetro/ablação); decaimentos por iteração; RNG por ambiente.
"""
import json
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import protocolo as P

P.preparar_sys_path()
from politica import AgenteRL, referencia_congelada, kl_referencia   # noqa: E402
from envs_lote import EnvLote, fabrica_env_padrao                    # noqa: E402
from rede_lstm import RedeBCLSTM                                     # noqa: E402

HP_PADRAO: Dict = {
    'lr': 1e-4, 'tam_elite': 40, 'n_eps_coleta': 8, 'updates_por_iter': 12, 'batch': 64, 'len_seq': 32,
    'fracao_demo0': 0.5, 'fracao_demo_min': 0.05, 'decay_demo': 0.967,
    'lambda_kl0': 0.05, 'lambda_kl_min': 0.003, 'decay_kl': 0.979,
    'tau0': 0.7, 'tau_min': 0.3, 'decay_tau': 0.994,
    'epsilon0': 0.08, 'epsilon_min': 0.0, 'decay_eps': 0.99,
    'p_antipausa0': 0.0, 'p_antipausa_min': 0.0, 'decay_antipausa': 0.998, 'pause_pen': 0.0,
    'max_grad_norm': 1.0, 'blocos': 9, 'dim_lstm': 512, 'dim_enc': 256, 'n_camadas': 2,
}


def _janelas(arrs, L):
    T = len(arrs[0])
    return [tuple(a[i:i + L] for a in arrs) for i in range(0, T - L + 1, L)]


def treinar_rl(politica_il: Optional[str], hp: Optional[Dict], seed: int, orcamento_passos: int,
               callback: Optional[Callable] = None, fabrica_env: Callable = fabrica_env_padrao,
               log: Callable = print, dispositivo: Optional[str] = None, demos: Optional[str] = None) -> Dict:
    if not politica_il:
        raise ValueError('o braço B (auto-imitação de elite) exige a política IL de warm-start')
    hp = {**HP_PADRAO, **(hp or {})}
    t0 = time.time()
    P.fixar_sementes(seed)
    rng = np.random.default_rng(seed)
    dev = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
    L, V, T = int(hp['len_seq']), int(hp['n_eps_coleta']), P.PASSOS_MAXIMOS

    agente = AgenteRL.carregar_de(politica_il, dev, dim_lstm=hp['dim_lstm'])
    ator: RedeBCLSTM = agente.rede
    ref = referencia_congelada(ator)
    from rede_lstm import PISO_STD_OBS, CLIP_OBS_NORM, normalizar_obs
    media, std = agente.obs_media, agente.obs_std
    media_t = torch.as_tensor(media, device=dev)
    std_t = torch.clamp(torch.as_tensor(std, device=dev), min=PISO_STD_OBS)   # D-28

    # janelas das demos (normalizadas)
    with open(demos or P.DEMOS_PADRAO, encoding='utf-8') as f:
        raw = json.load(f)
    win_demo = []
    for ep in raw:
        o = normalizar_obs(np.asarray([t['obs'] for t in ep], np.float32), media, std)   # D-28
        m = np.asarray([t['mascara'] for t in ep], np.float32)
        a = np.asarray([t['acao'] for t in ep], np.int64)
        win_demo.extend(_janelas([o, m, a], L))
    del raw

    lote = EnvLote(V, fabrica_env, seed_base=int(seed) * 100_000)
    opt = torch.optim.Adam(ator.parameters(), lr=hp['lr'])
    tau, eps_g, p_anti = hp['tau0'], hp['epsilon0'], hp['p_antipausa0']
    frac_demo, lam_kl = hp['fracao_demo0'], hp['lambda_kl0']
    passos, it, curva, episodios_treino = 0, 0, [], []
    tamanho_bloco = max(1, int(orcamento_passos) // int(hp['blocos'])); proximo_bloco = tamanho_bloco
    parar, desconectado, nivel_il = False, False, None

    def coletar(tau_, eps_, p_anti_):
        """V episódios completos com exploração; devolve lista de (retorno, n_esperar, (O, M, A)) e episódios brutos."""
        ator.eval()
        O = np.zeros((T, V, P.OBS_DIM), np.float32); M = np.zeros((T, V, P.N_ACOES), np.float32); A = np.zeros((T, V), np.int64)
        obs, mk = lote.reset_todos()
        h = ator.inicializar_hidden(V, dev)
        with torch.no_grad():
            for t in range(T):
                o = torch.clamp((torch.as_tensor(obs, device=dev) - media_t) / std_t,
                                -CLIP_OBS_NORM, CLIP_OBS_NORM).unsqueeze(1)   # D-28
                m = torch.as_tensor(mk, device=dev).unsqueeze(1)
                lg, h = ator(o, m, h)
                lg = lg.squeeze(1)
                if tau_ > 0:
                    a = torch.distributions.Categorical(logits=lg / tau_).sample().cpu().numpy()
                else:
                    a = lg.argmax(-1).cpu().numpy()
                for v in range(V):
                    validas = np.where(mk[v] > 0.5)[0]
                    if eps_ > 0 and rng.random() < eps_:
                        a[v] = rng.choice(validas)
                    if p_anti_ > 0 and a[v] == P.ACAO_ESPERAR and len(validas) > 1 and rng.random() < p_anti_:
                        a[v] = rng.choice(validas[validas != P.ACAO_ESPERAR])
                O[t] = o.squeeze(1).cpu().numpy(); M[t] = mk; A[t] = a
                obs, mk, _, fins, _ = lote.step(a)
        assert fins.all()
        eps = lote.coletar_concluidos()
        saida = []
        for v, e in enumerate(eps):
            n_esp = int((A[:, v] == P.ACAO_ESPERAR).sum())
            saida.append((e['retorno'], n_esp, (O[:, v], M[:, v], A[:, v])))
        return saida, eps

    def pontuacao(e):
        return e[0] - hp['pause_pen'] * e[1]

    # elite inicial: rollouts quase-gulosos da política IL (também define o nível do IL nos rollouts de treino)
    elite, eps0 = coletar(hp['tau_min'], 0.0, 0.0)
    passos += T * V; episodios_treino.extend(eps0)
    nivel_il = float(np.mean([e[0] for e in elite]))
    elite.sort(key=lambda e: -pontuacao(e)); elite = elite[:hp['tam_elite']]
    log(f"  [elite] nível do IL nos rollouts (τ={hp['tau_min']}): {nivel_il:.0f}")

    while passos < orcamento_passos and not parar:
        novos, eps = coletar(tau, eps_g, p_anti)
        passos += T * V; episodios_treino.extend(eps)
        elite.extend(novos); elite.sort(key=lambda e: -pontuacao(e)); elite = elite[:hp['tam_elite']]
        ret_elite = float(np.mean([e[0] for e in elite]))
        win_elite = []
        for _r, _n, (O, M, A) in elite:
            win_elite.extend(_janelas([O, M, A], L))

        ator.train()
        perdas, kls = [], []
        for _ in range(int(hp['updates_por_iter'])):
            ie = rng.integers(0, len(win_elite), min(hp['batch'], len(win_elite)))
            ob = torch.from_numpy(np.stack([win_elite[i][0] for i in ie])).to(dev)
            mb = torch.from_numpy(np.stack([win_elite[i][1] for i in ie])).to(dev)
            ab = torch.from_numpy(np.stack([win_elite[i][2] for i in ie])).to(dev)
            lg, _ = ator(ob, mb)
            perda = F.cross_entropy(lg.reshape(-1, P.N_ACOES), ab.reshape(-1))
            if lam_kl > 0:
                with torch.no_grad():
                    lg_ref, _ = ref(ob, mb)
                kl = kl_referencia(lg_ref, lg).mean()
                perda = perda + lam_kl * kl; kls.append(float(kl))
            if frac_demo > 0 and win_demo:
                nb = max(1, int(hp['batch'] * frac_demo)); jd = rng.integers(0, len(win_demo), nb)
                od = torch.from_numpy(np.stack([win_demo[i][0] for i in jd])).to(dev)
                md = torch.from_numpy(np.stack([win_demo[i][1] for i in jd])).to(dev)
                ad = torch.from_numpy(np.stack([win_demo[i][2] for i in jd])).to(dev)
                lgd, _ = ator(od, md)
                perda = perda + F.cross_entropy(lgd.reshape(-1, P.N_ACOES), ad.reshape(-1))
            opt.zero_grad(); perda.backward()
            torch.nn.utils.clip_grad_norm_(ator.parameters(), hp['max_grad_norm'])
            opt.step(); perdas.append(float(perda))
        ator.eval()
        it += 1
        if not desconectado and ret_elite >= nivel_il:
            desconectado = True
            log(f"  [elite] >>> desconexão do IL (elite {ret_elite:.0f} ≥ nível do IL {nivel_il:.0f}) na iteração {it}")
        if desconectado:
            frac_demo = max(hp['fracao_demo_min'], frac_demo * hp['decay_demo'])
            lam_kl = max(hp['lambda_kl_min'], lam_kl * hp['decay_kl'])
        tau = max(hp['tau_min'], tau * hp['decay_tau']); eps_g = max(hp['epsilon_min'], eps_g * hp['decay_eps'])
        p_anti = max(hp['p_antipausa_min'], p_anti * hp['decay_antipausa'])
        ret_treino = float(np.mean([e['retorno'] for e in eps]))
        curva.append({'passos': passos, 'fase': 'elite', 'iteracao': it, 'retorno_treino': ret_treino, 'retorno_elite': ret_elite,
                      'perda': float(np.mean(perdas)), 'kl': float(np.mean(kls)) if kls else None, 'fracao_demo': frac_demo,
                      'lambda_kl': lam_kl, 'tau': tau, 'epsilon': eps_g, 'p_antipausa': p_anti, 'desconectado': desconectado})
        log(f"  [elite {it}] passos {passos}/{orcamento_passos} | treino {ret_treino:.0f} | elite {ret_elite:.0f} | perda {np.mean(perdas):.4f} "
            f"| demo {frac_demo:.2f} kl {lam_kl:.3f} τ {tau:.2f} | {'RL' if desconectado else 'IL'} | {time.time() - t0:.0f}s")
        if callback is not None and passos >= proximo_bloco:
            proximo_bloco += tamanho_bloco
            parar = bool(callback(passos, agente))
    return {'agente': agente, 'curva': curva, 'passos': passos, 'tempo_s': time.time() - t0, 'hp': hp,
            'episodios_treino': episodios_treino, 'podado': parar, 'nivel_il_rollouts': nivel_il}
