"""Avaliador FALSO do DEHB, para testar o coordenador sem GPU: cumpre o contrato de `hpo_dehb.avaliar` (trava, resultado
atômico com impressão digital, códigos de saída) com uma função sintética no lugar do treino. Não importa torch.

Comportamento controlado pela variável de ambiente DEHB_FALSO (JSON):
  dorme_s                  segundos de "treino"
  falha_config_ids         ids que falham de forma determinística (saída 3 + falha.json)
  transitoria_uma_vez_ids  ids que morrem na primeira tentativa (saída 4) e funcionam na segunda
  transitoria_sempre_ids   ids que morrem sempre (saída 4)
  aptidao_nan_ids          ids cujo resultado sai não finito (o coordenador tem de tratar como falha)
  sem_resultado_ids        ids que saem com 0 sem gravar resultado
  contador                 arquivo onde cada execução acrescenta uma linha (para contar lançamentos)
  fracao_dos_passos        o "treino" que dá CERTO relata int(fidelidade x fração) passos em vez do nominal (FRE real != nominal)
"""
import os
import sys
import json
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(AQUI), 'hpo'))
sys.path.insert(0, os.path.dirname(AQUI))
import hpo_dehb as H  # noqa: E402


def main(dir_trabalho):
    ctl = json.loads(os.environ.get('DEHB_FALSO', '{}'))
    trava = H.Trava(os.path.join(dir_trabalho, '_EXECUTANDO.lock'))
    if not trava.tentar():
        return H.SAIDA_FALHA_TRANSITORIA
    try:
        tr = json.load(open(os.path.join(dir_trabalho, 'trabalho.json'), encoding='utf-8'))
        if ctl.get('contador'):
            with open(ctl['contador'], 'a', encoding='utf-8') as f:
                f.write(f"{tr['id']}\n")
        print(f"avaliador falso: trabalho {tr['id']} β validação", flush=True)   # o log tem de aguentar UTF-8
        inicio = H.agora_iso()
        time.sleep(float(ctl.get('dorme_s', 0.0)))
        passos = int(tr['fidelidade'])

        def falha(codigo, motivo, passos_gastos):
            H.escrever_json_atomico(os.path.join(dir_trabalho, 'falha.json'),
                                    {'impressao': tr['impressao'], 'id': tr['id'], 'motivo': motivo, 'detalhe': 'falso', 'passos': passos_gastos,
                                     'inicio': inicio, 'fim': H.agora_iso(), 'tempo_s': 0.0, 'hash_git': 'falso'})
            return codigo

        if tr['id'] in ctl.get('falha_config_ids', []):
            return falha(H.SAIDA_FALHA_DA_CONFIGURACAO, 'excecao_no_treino', passos // 2)
        marca = os.path.join(dir_trabalho, '_ja_falhou')
        if tr['id'] in ctl.get('transitoria_sempre_ids', []) or (tr['id'] in ctl.get('transitoria_uma_vez_ids', []) and not os.path.exists(marca)):
            open(marca, 'w').close()
            return falha(H.SAIDA_FALHA_TRANSITORIA, 'gpu_ou_sistema', passos // 4)
        if tr['id'] in ctl.get('sem_resultado_ids', []):
            return 0
        valor = -float(sum((v - 0.3) ** 2 for v in tr['vetor']))          # maximizado; o ótimo é o vetor 0,3
        if tr['id'] in ctl.get('aptidao_nan_ids', []):
            valor = float('nan')
        if 'fracao_dos_passos' in ctl:                                    # treinou menos (ou mais) que o nominal: só no caminho de sucesso
            passos = int(passos * float(ctl['fracao_dos_passos']))
        resultado = {'impressao': tr['impressao'], 'id': tr['id'], 'aptidao': -valor, 'objetivo': valor if tr['fidelidade_maxima'] else None,
                     'indicador': valor, 'passos': passos, 'fre': passos / tr['passos_max'], 'inicio': inicio, 'fim': H.agora_iso(),
                     'tempo_s': 0.0, 'curva': [{'passos': passos, 'proxy_norm': valor}], 'hash_git': 'falso'}
        if valor != valor:                                                # NaN não passa pelo escritor atômico (allow_nan=False): grava à mão
            with open(os.path.join(dir_trabalho, 'resultado.json'), 'w', encoding='utf-8') as f:
                json.dump(resultado, f)
        else:
            H.escrever_json_atomico(os.path.join(dir_trabalho, 'resultado.json'), resultado)
        return 0
    finally:
        trava.soltar()


if __name__ == '__main__':
    sys.exit(main(os.path.abspath(sys.argv[1])))
