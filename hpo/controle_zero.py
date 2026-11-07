"""hpo/controle_zero.py — o braço de CONTROLE do MimePlant: HPO sobre o RL treinado DO ZERO (D-32, 11/09/2026).

POR QUE ESTE EXPERIMENTO EXISTE. A rodada 2 mediu três dos quatro quadrantes do desenho:

                    | configuração padrão | otimizada por HPO
    com imitação    |        0,874        |   1,696 (MimePlant)
    do zero         |        1,481        |      faltando

Sem o quarto quadrante, a afirmação "a imitação ajuda" fica confundida com a otimização: o braço com imitação foi
otimizado por 64 FRE e o braço do zero rodou na configuração padrão. A ablação `abl_zero` também não fecha a
lacuna, porque ela treina do zero com os hiperparâmetros que foram otimizados PARA partir da imitação — o que
desfavorece o controle por construção. Este script fecha a célula: mesma busca, mesmo orçamento, mesmo podador,
mesmo objetivo, mesmas sementes, mesma avaliação, treinando do zero.

O QUE É MANTIDO IDÊNTICO ao estudo do método (`tpe_v1` / `rs_v1`): orçamento de 64 FRE por método; 250.000 passos
por trial; uma semente de treino por trial (42); `HyperbandPruner(min_resource=N/3, max_resource=N, eta=3)` com o
bloco terminal fora da decisão de poda; objetivo = média nos 3 regimes da normalizada em validação-objetivo
(520-549); proxy de poda em validação-rollback in-dist (500-519); referências de `results/referencias_v2`; semente
do amostrador derivada por crc32 do nome do estudo; dois métodos de busca (TPE com poda e busca aleatória com a
mesma poda); reavaliação do top-3 com as 5 sementes de treino e avaliação em teste (999-1048).

A ÚNICA DIFERENÇA, e ela é declarada: o espaço tem 8 hiperparâmetros em vez de 12. `ppo.treinar_rl` anula
`temperatura_inicial`, `beta_kl0`, `beta_kl_decay` e `k_aquec` quando não há política de partida (ppo.py:73-82),
então sorteá-los gastaria orçamento em dimensões inertes. Não existe escolha neutra aqui: com 8 dimensões o
controle resolve um problema de busca mais fácil sob o mesmo orçamento; com 12, desperdiça parte do orçamento.
`--espaco-completo` roda a alternativa, para quem preferir igualar a dimensão. Seja qual for a escolha, ela tem de
aparecer no texto — e, se o controle vencer com 8 dimensões, a ressalva de que a busca dele era mais fácil é
obrigatória.

NÃO É um quinto candidato do MimePlant. O MimePlant continua sendo imitação seguida de RL (D-30); este estudo é
termo de comparação, e nada aqui é selecionado em sementes de teste.

Uso (a partir da raiz do código):
  python -X utf8 hpo/controle_zero.py [--n-fre 64] [--paralelo 3] [--retomar]
  python -X utf8 hpo/controle_zero.py --espaco-completo --sufixo compl   # a alternativa de 12 dimensões
  python -X utf8 hpo/controle_zero.py --so-comparar                      # só recomputa a tabela 2x2
Saída: ../experimentos/hpo/{tpe_zero_v1,rs_zero_v1}.db, results/hpo_reavaliacao_zero/ e
       results/controle_zero/comparacao.json com a tabela 2x2 e as diferenças pareadas por semente.
"""
import os
import sys
import json
import time
import zlib
import argparse
import subprocess

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocolo as P                                  # noqa: E402

VF = os.path.dirname(RAIZ)
DIR_HPO = os.path.join(VF, 'experimentos', 'hpo')
PY = [sys.executable, '-X', 'utf8']

# Os quadrantes já medidos na rodada 2, para a tabela final. Nomes como aparecem em hpo_reavaliacao/resumo.json.
QUADRANTES_RODADA2 = {
    'imitacao_padrao': 'padrao_ppo_il',
    'zero_padrao': 'padrao_ppo_zero',
}


def executar(etapa, cmd, log, saida_flag=None):
    """Roda uma etapa e para a cadeia se ela falhar. Cada etapa é resumível por conta própria (--retomar)."""
    log(f'=== {etapa}: {" ".join(cmd)}')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=RAIZ, env=dict(os.environ, PYTHONUTF8='1'))
    log(f'=== {etapa}: exit={r.returncode} em {(time.time() - t0) / 60:.0f} min')
    if r.returncode != 0:
        raise SystemExit(f'{etapa} falhou (exit {r.returncode})')
    if saida_flag and not os.path.exists(saida_flag):
        raise SystemExit(f'{etapa} terminou com exit 0 mas sem {saida_flag}')


def comparar(args, log):
    """Monta a tabela 2x2 em TESTE e as diferenças pareadas por semente de treino.

    O MimePlant e o controle otimizado são cada um o MELHOR do seu estudo pela VALIDAÇÃO-objetivo com 5 sementes
    (`escolher_melhor_candidato`), nunca pelo teste. As diferenças são pareadas por semente: a semente s do
    MimePlant contra a semente s do controle, o que remove a variação de inicialização da rede comum às duas."""
    from reavaliar_topk import escolher_melhor_candidato
    res_met = json.load(open(os.path.join(P.DIR_REAVALIACAO, 'resumo.json'), encoding='utf-8'))   # o estudo do MÉTODO, mesma rodada
    res_zero = json.load(open(os.path.join(args.saida_reav, 'resumo.json'), encoding='utf-8'))

    melhor_met = escolher_melhor_candidato(res_met, n_sementes=len(P.SEMENTES_TREINO))
    melhor_zero = escolher_melhor_candidato(res_zero, n_sementes=len(P.SEMENTES_TREINO))
    if melhor_met is None or melhor_zero is None:
        raise SystemExit('falta o melhor candidato de um dos lados; a reavaliação dos dois precisa estar completa')

    def bloco(res, nome):
        d = res[nome]['teste']['normalizada_media_3_regimes']
        return {'candidato': nome, 'media': d['media'], 'std_amostral': float(np.std(d['por_semente'], ddof=1)),
                'por_semente': list(d['por_semente'])}

    quad = {
        'imitacao_otimizado': bloco(res_met, melhor_met),
        'zero_otimizado': bloco(res_zero, melhor_zero),
        'imitacao_padrao': bloco(res_met, QUADRANTES_RODADA2['imitacao_padrao']),
        'zero_padrao': bloco(res_met, QUADRANTES_RODADA2['zero_padrao']),
    }

    def pareado(a, b):
        d = np.asarray(quad[a]['por_semente']) - np.asarray(quad[b]['por_semente'])
        out = {'diferenca_media': float(d.mean()), 'n': int(d.size),
               'd_cohen_pareado': float(d.mean() / (d.std(ddof=1) + 1e-12)),
               'vence_em_sementes': int((d > 0).sum())}
        try:
            from scipy import stats
            t, p = stats.ttest_rel(quad[a]['por_semente'], quad[b]['por_semente'])
            out['t'], out['p'] = float(t), float(p)
        except Exception as e:
            out['aviso_teste'] = f'scipy indisponível: {e}'
        return out

    comp = {
        'o_que_a_imitacao_acrescenta_com_HPO': pareado('imitacao_otimizado', 'zero_otimizado'),
        'o_que_a_imitacao_acrescenta_sem_HPO': pareado('imitacao_padrao', 'zero_padrao'),
        'o_que_a_HPO_acrescenta_com_imitacao': pareado('imitacao_otimizado', 'imitacao_padrao'),
        'o_que_a_HPO_acrescenta_do_zero': pareado('zero_otimizado', 'zero_padrao'),
    }
    out = {'gerado_em': time.strftime('%Y-%m-%d %H:%M:%S'), 'conjunto': 'teste (999-1048), 5 sementes de treino',
           'escala': 'recompensa normalizada, média dos 3 regimes; 1,0 = professor',
           'selecao': 'cada lado é o melhor do seu estudo em validação-objetivo (520-549); teste só reportado',
           'espaco_do_controle': '12 hiperparâmetros' if args.espaco_completo else '8 hiperparâmetros efetivos',
           'quadrantes': quad, 'comparacoes': comp}
    os.makedirs(args.saida, exist_ok=True)
    with open(os.path.join(args.saida, 'comparacao.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    log('')
    log('TABELA 2x2 — recompensa normalizada em teste (1,0 = professor)')
    log('%-16s %-22s %-22s' % ('', 'configuração padrão', 'otimizada por HPO'))
    for linha, (a, b) in (('com imitação', ('imitacao_padrao', 'imitacao_otimizado')),
                          ('do zero', ('zero_padrao', 'zero_otimizado'))):
        log('%-16s %-22s %-22s' % (linha,
                                   '%.3f ± %.3f' % (quad[a]['media'], quad[a]['std_amostral']),
                                   '%.3f ± %.3f' % (quad[b]['media'], quad[b]['std_amostral'])))
    log('')
    for k, v in comp.items():
        log('%-38s %+.3f (vence em %d/%d sementes%s)' % (
            k, v['diferenca_media'], v['vence_em_sementes'], v['n'],
            '' if 'p' not in v else ', p=%.4f' % v['p']))
    log(f"[ok] {os.path.join(args.saida, 'comparacao.json')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-fre', type=float, default=64.0, help='mesmo orçamento do estudo do método')
    ap.add_argument('--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS)
    ap.add_argument('--n-startup', type=int, default=10, help='mesmo valor usado em tpe_v1')
    ap.add_argument('--paralelo', type=int, default=3, help='processos por estudo (a rodada 2 fechou com 3)')
    ap.add_argument('--sufixo', default=P.SUFIXO_STUDIES_ZERO, help='sufixo dos studies: tpe_zero_<sufixo>, rs_zero_<sufixo>')
    ap.add_argument('--espaco-completo', action='store_true', help='sortear as 12 dimensões também no controle')
    ap.add_argument('--saida', default=P.DIR_CONTROLE_ZERO)
    ap.add_argument('--saida-reav', default=P.DIR_REAVALIACAO_ZERO)
    ap.add_argument('--so-comparar', action='store_true', help='não roda HPO nem reavaliação; só refaz a tabela')
    ap.add_argument('--retomar', action='store_true')
    args = ap.parse_args()
    args.saida, args.saida_reav = os.path.abspath(args.saida), os.path.abspath(args.saida_reav)
    os.makedirs(args.saida, exist_ok=True)
    if not args.so_comparar:
        P.limpar_flags(args.saida)

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    estudos = [('tpe_zero_' + args.sufixo, 'tpe'), ('rs_zero_' + args.sufixo, 'random')]
    try:
        if args.so_comparar:
            comparar(args, log)
            return 0

        log(f'CONTROLE DO ZERO (D-32) | {args.n_fre} FRE por método | espaço de '
            f'{"12" if args.espaco_completo else "8"} hiperparâmetros | {args.paralelo} processos por estudo')

        # ── 1. os dois estudos, em paralelo, espelhando a rodada 2 ──
        vivos = []
        for nome, sampler in estudos:
            for i in range(max(1, args.paralelo)):
                cmd = PY + [os.path.join(RAIZ, 'hpo', 'hpo_rl.py'), '--study', nome, '--sampler', sampler,
                            '--n-fre', str(args.n_fre), '--orcamento', str(args.orcamento),
                            '--inicializacao', 'zero', '--retomar']
                if sampler == 'tpe':
                    cmd += ['--n-startup', str(args.n_startup)]
                if args.espaco_completo:
                    cmd += ['--espaco-completo']
                if i > 0:
                    # Trabalhadores extras: constant_liar e semente PRÓPRIA, senão todos sorteariam a mesma
                    # sequência. `zlib.crc32` e não `hash()`: o hash de str é aleatorizado por processo.
                    cmd += ['--paralelo', '--seed-sampler',
                            str(200_000 * i + zlib.crc32(nome.encode('utf-8')) % 100_000)]
                f = open(os.path.join(args.saida, f'_{nome}_w{i}.log'), 'a', encoding='utf-8')
                vivos.append((f'{nome}#{i}', subprocess.Popen(cmd, cwd=RAIZ, env=dict(os.environ, PYTHONUTF8='1'),
                                                              stdout=f, stderr=subprocess.STDOUT), f))
                log(f'  lançado {nome} trabalhador {i} (pid {vivos[-1][1].pid})')
                time.sleep(5)                   # evita dois processos criando o mesmo study ao mesmo tempo
        falhas = []
        while vivos:
            time.sleep(60)
            for nome, p, f in [v for v in vivos if v[1].poll() is not None]:
                vivos.remove((nome, p, f)); f.close()
                log(f'  {nome} terminou (exit {p.returncode})')
                if p.returncode != 0:
                    falhas.append(nome)
        if falhas:
            raise SystemExit(f'trabalhadores falharam: {falhas}')
        for nome, _ in estudos:
            flag = os.path.join(DIR_HPO, nome, '_DONE.flag')
            if not os.path.exists(flag):
                raise SystemExit(f'{nome} sem _DONE.flag: orçamento não foi consumido')

        # ── 2. reavaliação do top-3 de cada estudo, 5 sementes, mesma rotina do método ──
        executar('reavaliacao', PY + [os.path.join(RAIZ, 'hpo', 'reavaliar_topk.py'),
                                      '--study', *[n for n, _ in estudos], '--k', '3',
                                      '--saida', args.saida_reav, '--orcamento', str(args.orcamento), '--retomar'],
                 log, os.path.join(args.saida_reav, '_DONE.flag'))

        # ── 3. a tabela 2x2 ──
        comparar(args, log)
        P.escrever_flag(args.saida, '_DONE.flag', 'controle do zero: HPO + reavaliação + comparação')
        log('controle do zero concluído')
        return 0
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code in (0, None):
            raise
        import traceback
        traceback.print_exc()
        P.escrever_flag(args.saida, '_FAILED.flag', traceback.format_exc().splitlines()[-1])
        return 1


if __name__ == '__main__':
    sys.exit(main())
