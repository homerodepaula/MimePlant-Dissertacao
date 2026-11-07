"""Roda `hpo/hpo_rl.py` com um amostrador que ele não conhece (GP-BO ou Sobol/QMC), SEM alterar o repositório.

Por que um invólucro. O `hpo_rl.py` aceita só `--sampler {tpe,random}`. Acrescentar um terceiro exigiria editar um arquivo
versionado do código da defesa. Aqui o `hpo_rl.main()` é chamado como está, com todo
o protocolo intacto (objetivo, poda Hyperband N/3, contabilidade em FRE, sementes, flags, retomada), e só o objeto amostrador
é substituído no momento em que ele é construído.

Como a substituição funciona, e por que é segura:
  * `hpo_rl` chama `optuna.samplers.RandomSampler(seed=...)` uma única vez, quando `--sampler` não é `tpe`. O invólucro troca
    esse símbolo por uma fábrica de UM DISPARO: a primeira chamada devolve o amostrador pedido, e qualquer chamada seguinte
    devolve o `RandomSampler` verdadeiro. Isso evita recursão infinita, porque o próprio GPSampler constrói um RandomSampler
    interno para a fase de partida.
  * `study.set_user_attr('sampler', ...)` é interceptado para gravar o nome REAL do amostrador no banco, senão a procedência
    do estudo diria "random".
  * Nada mais é tocado: mesma semente efetiva derivada pelo `hpo_rl`, mesmo podador, mesmo objetivo.

Amostradores disponíveis:
  gp   `optuna.samplers.GPSampler` — otimização bayesiana com processo gaussiano (modelo da função + incerteza).
       Família diferente do TPE (que modela densidades de bons/ruins). Sem dependência nova: usa torch e scipy.
  qmc  `optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True)` — sequência de baixa discrepância, que cobre o espaço
       de forma mais uniforme que o sorteio independente. É a amostragem usada pelo ARLBench nas paisagens de hiperparâmetros.

Uso (da raiz do repositório; os demais argumentos são os do hpo_rl.py):
  python -X utf8 <este arquivo> --sampler-extra gp --study gp_r3_a --n-fre 32 --orcamento 1000000 --n-startup 5 ...
"""
import os
import sys
import argparse

import optuna

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)                                      # este script fica em <repositório>/execucao
EXPERIMENTOS = os.path.join(os.path.dirname(REPO), 'experimentos')  # pasta irmã do repositório (a mesma convenção de hpo/*.py)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'hpo'))

RANDOM_REAL = optuna.samplers.RandomSampler
CRIAR_STUDY_REAL = optuna.create_study


def construir(nome, seed, n_startup):
    """O amostrador pedido, com a semente que o hpo_rl derivou."""
    if nome == 'gp':
        # n_startup_trials: quantos trials aleatórios antes de ajustar o GP. Usamos o mesmo valor do TPE (D-36 = 5),
        # para que os três métodos com modelo partam da mesma quantidade de exploração inicial.
        return optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup,
                                         independent_sampler=RANDOM_REAL(seed=seed + 1),
                                         deterministic_objective=False)
    if nome == 'qmc':
        # scramble=True: Sobol embaralhado, para que as 3 execuções (sementes 1001/1002/1003) sejam diferentes entre si
        # e a comparação tenha as mesmas 3 repetições dos outros otimizadores.
        return optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed=seed,
                                          independent_sampler=RANDOM_REAL(seed=seed + 1),
                                          warn_asynchronous_seeding=False)
    raise SystemExit(f'amostrador desconhecido: {nome!r} (use gp ou qmc)')


def instalar_patches(nome, n_startup):
    estado = {'usado': False, 'amostrador': None}

    def fabrica_um_disparo(seed=None, **kw):
        if estado['usado']:
            return RANDOM_REAL(seed=seed, **kw)          # usos internos (ex.: fase de partida do GP)
        estado['usado'] = True
        estado['amostrador'] = construir(nome, seed, n_startup)
        print(f'[extra] amostrador {nome} construído com semente {seed}', flush=True)
        return estado['amostrador']

    def criar_study(*a, **k):
        study = CRIAR_STUDY_REAL(*a, **k)
        set_attr_real = study.set_user_attr

        def set_user_attr(chave, valor):
            # procedência honesta: o banco tem de dizer qual amostrador de fato rodou
            set_attr_real(chave, nome if chave == 'sampler' else valor)
        study.set_user_attr = set_user_attr
        return study

    optuna.samplers.RandomSampler = fabrica_um_disparo
    optuna.create_study = criar_study


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--sampler-extra', required=True, choices=('gp', 'qmc'))
    ap.add_argument('--n-startup', type=int, default=5)      # lido aqui e repassado ao hpo_rl
    extras, resto = ap.parse_known_args()
    instalar_patches(extras.sampler_extra, extras.n_startup)

    import hpo_rl                                            # importado DEPOIS dos patches
    # o hpo_rl decide pelo ramo `else` (não-tpe) e cai na fábrica acima
    sys.argv = [os.path.join(REPO, 'hpo', 'hpo_rl.py'), '--sampler', 'random',
                '--n-startup', str(extras.n_startup)] + resto
    return hpo_rl.main()


if __name__ == '__main__':
    sys.exit(main())
