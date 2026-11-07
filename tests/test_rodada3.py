"""Testes da rodada 3 (D-33, 18/09/2026): cada rodada grava em pasta própria e todo script toma o caminho e o orçamento
padrão do protocolo. Até a rodada 2 os padrões estavam redigitados em oito scripts, e bastava esquecer um `--referencias`
para normalizar um agente da rodada nova pela âncora da antiga, em silêncio.
"""
import os
import re
import sys
import json

import pytest

import protocolo as P

sys.path.insert(0, os.path.join(P.RAIZ, 'hpo'))

SCRIPTS_DA_CADEIA = ['referencias.py', 'treinar_warmstart_il.py', 'bakeoff.py', os.path.join('hpo', 'hpo_rl.py'),
                     os.path.join('hpo', 'reavaliar_topk.py'), os.path.join('hpo', 'ablacoes.py'), os.path.join('hpo', 'analise_hpo.py'),
                     os.path.join('hpo', 'pos_hpo.py'), os.path.join('hpo', 'controle_zero.py')]


def test_pastas_da_rodada_ficam_todas_sob_a_pasta_da_rodada():
    assert P.RODADA == 'rodada3' and P.DIR_RODADA == os.path.join(P.DIR_RESULTS, 'rodada3')
    for d in (P.DIR_REFERENCIAS, P.DIR_WARMSTART, P.DIR_RL_PADRAO, P.DIR_REAVALIACAO, P.DIR_ABLACOES, P.DIR_POS_HPO,
              P.DIR_CONTROLE_ZERO, P.DIR_REAVALIACAO_ZERO):
        assert os.path.dirname(d) == P.DIR_RODADA, d
    assert os.path.dirname(P.REFERENCIAS_PADRAO) == P.DIR_REFERENCIAS
    assert all(s.endswith('_r3') for s in P.STUDIES_HPO)
    assert P.snapshot_protocolo()['rodada'] == {'nome': 'rodada3', 'dir': P.DIR_RODADA, 'orcamento_rl_passos': P.ORCAMENTO_RL_PASSOS,
                                                'studies_hpo': list(P.STUDIES_HPO)}


def test_nenhum_argumento_padrao_aponta_para_a_rodada_2_nem_redigita_o_orcamento():
    """Só as linhas de `add_argument` (docstrings e comentários podem citar a rodada 2 como história)."""
    proibidos = re.compile(r"referencias_v2|il_warmstart_v2|'bakeoff'|'hpo_reavaliacao'|'hpo_ablacoes'|'pos_hpo'|'controle_zero'"
                           r"|tpe_v1|rs_v1|250_?000")
    achados = []
    for rel in SCRIPTS_DA_CADEIA:
        for n, linha in enumerate(open(os.path.join(P.RAIZ, rel), encoding='utf-8'), start=1):
            if 'add_argument' in linha and 'default' in linha and proibidos.search(linha.split('help=')[0]):
                achados.append(f'{rel}:{n}: {linha.strip()}')
    assert achados == []


def test_orcamento_por_execucao_tem_fonte_unica():
    import inspect
    import bakeoff, hpo_rl, reavaliar_topk, ablacoes, pos_hpo, controle_zero
    for mod in (bakeoff, hpo_rl, reavaliar_topk, ablacoes, pos_hpo, controle_zero):
        assert "'--orcamento', type=int, default=P.ORCAMENTO_RL_PASSOS" in inspect.getsource(mod.main), mod.__name__
    # e o pós-HPO REPASSA o orçamento à reavaliação e às ablações (até a rodada 2 as duas caíam no próprio padrão)
    fonte = inspect.getsource(pos_hpo.main)
    assert fonte.count("'--orcamento', str(args.orcamento)") == 2


def test_braco_do_zero_registra_que_treinou_do_zero():
    """Defeito de registro da rodada 2: `ppo_zero` gravava `inicializacao = 'professor'` embora treinasse sem política de partida."""
    import inspect
    import bakeoff
    fonte = inspect.getsource(bakeoff.rodar_um)
    assert "if not info['warmstart']:" in fonte and "hp['inicializacao'] = 'zero'" in fonte


def test_reavaliacao_em_fatias_cobre_cada_execucao_uma_vez(tmp_path, monkeypatch):
    """`--fatia i/n` treina só a sua parte, não agrega e não grava flag na raiz; as n fatias juntas cobrem tudo, sem repetição."""
    import reavaliar_topk as R
    feitas = []
    monkeypatch.setattr(R, 'carregar_referencias', lambda caminho: {})
    monkeypatch.setattr(R, 'rodar_candidato', lambda nome, hp, seed, args, refs, log: feitas.append((nome, seed)))
    n = 4
    for i in range(n):
        monkeypatch.setattr(sys, 'argv', ['reavaliar_topk.py', '--padrao', 'ppo_il', 'ppo_zero', '--seeds', '42', '123', '2024', '7', '1337',
                                          '--saida', str(tmp_path), '--referencias', 'x.json', '--fatia', f'{i}/{n}'])
        assert R.main() == 0
    esperado = [(f'padrao_{b}', s) for b in ('ppo_il', 'ppo_zero') for s in (42, 123, 2024, 7, 1337)]
    assert sorted(feitas) == sorted(esperado) and len(feitas) == len(set(feitas)) == 10
    assert not os.path.exists(tmp_path / '_DONE.flag') and not os.path.exists(tmp_path / 'resumo.json')


def test_trabalhadores_da_hpo_tem_semente_propria():
    """Processos do mesmo study lançados juntos veem o banco vazio e, com a mesma semente, sorteariam a mesma sequência."""
    import inspect
    import hpo_rl
    fonte = inspect.getsource(hpo_rl.main)
    assert "'--trabalhador', type=int, default=0" in fonte
    assert 'semente_study += 200_000 * int(args.trabalhador)' in fonte


@pytest.mark.parametrize('script', ['lancar-rodada3.ps1', 'encadear-rodada3.ps1'])
def test_scripts_da_cadeia_sao_ascii_e_nao_tocam_a_rodada_2(script):
    caminho = os.path.join(os.path.dirname(os.path.dirname(P.RAIZ)), '_setup', script)
    if not os.path.exists(caminho):
        pytest.skip('_setup fica fora do repositório do código; sem ele (clone avulso) não há o que conferir')
    texto = open(caminho, 'rb').read()
    assert all(b < 128 for b in texto), 'PowerShell 5.1 lê o .ps1 na página de código do sistema: só ASCII'
    t = texto.decode('ascii')
    for proibido in ('referencias_v2', 'il_warmstart_v2', 'results\\bakeoff', 'tpe_v1', 'rs_v1', 'Remove-Item', 'Move-Item'):
        assert proibido not in t, proibido
    if script.startswith('encadear'):
        # a cadeia passa o orçamento e as pastas da rodada explicitamente a toda etapa que treina RL
        # (3 etapas da cadeia principal + as 2 do DEHB, que é opcional)
        assert t.count("' --orcamento ' + $Orcamento") == 5 and t.count('results\\rodada3\\referencias\\referencias.json') >= 3
        assert 'hpo_reavaliacao_dehb' in t and '--studies tpe_r3 rs_r3 --k' in t   # o DEHB fica FORA do pos_hpo que escolhe o MimePlant
        # o job_wrapper roda `cmd /c "<comando> >> log 2>&1"`: sem parênteses, em "a && b" só a saída de b iria para o log
        for linha in t.splitlines():
            if ' && ' in linha and 'Comando = ' in linha:
                comando = linha.split("Comando = ('", 1)[1].split("'); PastaSaida", 1)[0]
                assert comando.startswith('(') and comando.endswith(')'), linha[:120]
    else:
        # PastaSaida do encadeador = pasta cuja _DONE.flag só a ÚLTIMA etapa escreve: o validador toma "processo morto + flag" por
        # job concluído, e com -ComDEHB a flag do pos_hpo aparece ANTES do fim — a cadeia morta no DEHB nunca seria relançada
        assert "-PastaSaida (Join-Path $codigo 'results\\rodada3\\pos_hpo')" not in t and '-PastaSaida $pastaFinal' in t
        assert "if ($ComDEHB) { Join-Path $R3 'hpo_reavaliacao_dehb' } else { Join-Path $R3 'pos_hpo' }" in t
