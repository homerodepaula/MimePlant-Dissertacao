"""gerar_demos.py — demonstrações do PROFESSOR (sem pausas, D-27; manejo intensivo, D-33) para o warm-start IL (FarmGym v4).

Cada episódio i usa a semente de ambiente `protocolo.sementes_demos()[i]` (10000+i) e pausas semeadas por episódio
(S13), então o arquivo é reproduzível bit a bit. Formato igual ao da qualificação (lista de episódios; transição =
{obs, acao, recompensa, prox_obs, feito, mascara}) para `bc.carregar_trajetorias` e `nucleo.carregar_episodios_demos`.

Uso:  python -X utf8 gerar_demos.py [--saida results/demos_v5] [--n-eps 200] [--retomar] [--p 0 --L 8]
      A política é a de `protocolo.PROFESSOR_VERSAO`; a pasta padrão é `protocolo.DIR_DEMOS`.
      `--p` é a imperfeição do professor; o padrão é `protocolo.PROFESSOR_IMPERFEICAO` (= 0, sem pausas). Só passe
      `--p 0.14` para reproduzir a variante degradada, e nesse caso grave em outra pasta: o hash muda e o
      `treinar_warmstart_il.py` recusa demos com hash diferente do protocolo.
Resumível: cada episódio vai para <saida>/parciais/ep_<semente>.json; o arquivo final e o config.json (com o hash
SHA-256 das demos) são escritos ao terminar; _DONE.flag no fim, _FAILED.flag em erro.
"""
import os
import sys
import json
import time
import argparse
import traceback

import numpy as np

import protocolo as P

P.preparar_sys_path()
from env import AmbienteFazendaGym                # noqa: E402
from mascara_acoes import InvolucroMascaraAcoes   # noqa: E402
from bot_perfeito import NOMES_ACOES              # noqa: E402


def gerar_episodio(env, bot, seed):
    bruto = P.env_bruto(env)
    obs, info = env.reset(seed=int(seed))
    bot.resetar_episodio(seed_episodio=int(seed))
    traj, total, colheitas = [], 0.0, 0
    for _ in range(P.PASSOS_MAXIMOS):
        mascara = info['mascara_acoes']
        acao = int(bot.selecionar_acao(bruto))
        obs_ant = obs.copy()
        obs, r, terminado, truncado, info = env.step(acao)
        total += float(r); colheitas += int(acao == P.ACAO_COLHER)
        traj.append({'obs': obs_ant.tolist(), 'acao': acao, 'recompensa': float(r), 'prox_obs': obs.tolist(),
                     'feito': bool(terminado or truncado), 'mascara': mascara.tolist()})
        if terminado or truncado:
            break
    return traj, total, colheitas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saida', default=os.path.relpath(P.DIR_DEMOS, P.RAIZ))
    ap.add_argument('--n-eps', type=int, default=P.N_DEMOS)
    ap.add_argument('--p', type=float, default=P.PROFESSOR_IMPERFEICAO)
    ap.add_argument('--L', type=int, default=P.ORACULO_LAZY_STEPS)
    ap.add_argument('--retomar', action='store_true')
    args = ap.parse_args()

    saida = os.path.abspath(args.saida)
    parciais = os.path.join(saida, 'parciais')
    os.makedirs(parciais, exist_ok=True)
    final = os.path.join(saida, 'trajetorias_especialista.json')
    if args.retomar and os.path.exists(final) and os.path.exists(os.path.join(saida, '_DONE.flag')):
        print('[demos] já concluído:', final); return 0
    P.limpar_flags(saida)

    sementes = P.sementes_demos()[:args.n_eps]
    t0 = time.time()
    P.escrever_status(saida, 'inicio', f'n_eps={len(sementes)} p={args.p} L={args.L}')
    try:
        env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=P.PASSOS_MAXIMOS))
        bot = P.novo_bot(imperfeicao=args.p, lazy_steps=args.L)   # a política do professor (protocolo.PROFESSOR_VERSAO, D-33)
        episodios, stats = [], []
        with P.rng_isolado():
            for i, s in enumerate(sementes):
                arq = os.path.join(parciais, f'ep_{s}.json')
                if args.retomar and os.path.exists(arq):
                    d = json.load(open(arq, encoding='utf-8'))
                else:
                    traj, total, colh = gerar_episodio(env, bot, s)
                    d = {'semente': s, 'recompensa': total, 'colheitas': colh, 'trajetoria': traj}
                    with open(arq, 'w', encoding='utf-8') as f:
                        json.dump(d, f)
                episodios.append(d['trajetoria'])
                stats.append({'semente': d['semente'], 'recompensa': d['recompensa'], 'colheitas': d['colheitas']})
                if (i + 1) % 10 == 0:
                    ult = [x['recompensa'] for x in stats[-10:]]
                    print(f'  ep {i + 1:4d}/{len(sementes)} | média(10) {np.mean(ult):8.1f} | {time.time() - t0:5.0f}s', flush=True)
                    P.escrever_status(saida, 'progresso', f'{i + 1}/{len(sementes)}')
        with open(final, 'w', encoding='utf-8') as f:
            json.dump(episodios, f)
        acoes = np.zeros(P.N_ACOES, dtype=np.int64)
        for ep in episodios:
            for t in ep:
                acoes[t['acao']] += 1
        n_trans = int(sum(len(ep) for ep in episodios))
        recs = np.array([x['recompensa'] for x in stats])
        estat = {'n_episodios': len(episodios), 'n_transicoes': n_trans, 'sementes': sementes,
                 'media_recompensa': float(recs.mean()), 'std_recompensa': float(recs.std()),
                 'media_colheitas': float(np.mean([x['colheitas'] for x in stats])),
                 'acoes_total': {NOMES_ACOES[i]: int(acoes[i]) for i in range(P.N_ACOES)},
                 'acoes_pct': {NOMES_ACOES[i]: round(100 * float(acoes[i] / acoes.sum()), 1) for i in range(P.N_ACOES)},
                 'por_episodio': stats, 'tempo_s': time.time() - t0}
        with open(os.path.join(saida, 'estatisticas.json'), 'w', encoding='utf-8') as f:
            json.dump(estat, f, indent=1, ensure_ascii=False)
        h = P.hash_arquivo(final)
        P.registrar_config(saida, experimento='demos_v2', arquivo=final, hash_demos=h, n_episodios=len(episodios),
                           n_transicoes=n_trans, oraculo={'versao': P.PROFESSOR_VERSAO, 'classe': type(bot).__name__,
                                                          'imperfeicao': args.p, 'lazy_steps': args.L, 'seed': P.ORACULO_SEED,
                                                          'pausas_semeadas_por_episodio': True,
                                                          'papel': 'professor' if args.p == P.PROFESSOR_IMPERFEICAO else 'variante'},
                           sementes_ambiente=[sementes[0], sementes[-1]], tempo_s=time.time() - t0)
        print(f"[demos] {len(episodios)} episódios, {n_trans} transições | média {recs.mean():.1f} ± {recs.std():.1f} | "
              f"colheitas {estat['media_colheitas']:.1f} | esperar {estat['acoes_pct']['esperar']}% | hash {h[:16]} | {time.time() - t0:.0f}s")
        P.escrever_flag(saida, '_DONE.flag', f'hash={h}')
        return 0
    except Exception:
        traceback.print_exc()
        P.escrever_flag(saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
