# experimental_analysis — análise experimental reprodutível

Análise estatística dos resultados em `results/` (ILS, GRASP, BRKGA e PSO; 15 instâncias × 100 sementes),
fundamentada apenas em:

| Sigla | Fonte |
|---|---|
| [D] | Derrac, García, Molina, Herrera (2011). *A practical tutorial on the use of nonparametric statistical tests…* Swarm Evol. Comput. 1:3–18 |
| [E] | Eiben, Smit (2011). *Parameter tuning for configuring and analyzing evolutionary algorithms.* Swarm Evol. Comput. 1:19–31 |
| [B] | Beiranvand, Hare, Lucet (2017). *Best practices for comparing optimization algorithms.* arXiv:1709.08242 |
| [BB] | Bartz-Beielstein, Chiarandini, Paquete, Preuss (eds.) (2010). *Experimental Methods for the Analysis of Optimization Algorithms.* Springer |
| [S] | Slides PCO212 – 7. Projeto e Análise de Experimentos (Frinhani, 2026) |

Nenhum arquivo existente do repositório é alterado; os scripts só leem `results/`.

## Como executar

```bash
cd experimental_analysis
python build_dataset.py --results ../results --out outputs   # passo 1: consolida os resultados
python analysis.py --out outputs                              # passo 2: testes, tabelas e figuras
python tests/test_ea_stats.py                                 # verificação dos testes estatísticos
```

Dependências: `numpy`, `pandas`, `scipy`, `matplotlib` (já em `requirements.txt`). Tempo total < 30 s.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `ea_stats.py` | Friedman/Iman–Davenport, Friedman Aligned, Quade, post-hoc 1×N (Bonferroni, Holm, Hochberg, Hommel, Holland, Finner, Li), N×N (Nemenyi, Holm, Shaffer, Bergmann–Hommel), estimação de contraste, sinal e Wilcoxon [D] |
| `tests/test_ea_stats.py` | Confere a implementação com os exemplos numéricos de [D] (Tabelas 8, 11, 12–16, 20) |
| `build_dataset.py` | Gera `outputs/data/`: `runs.csv` (uma linha por execução), `instances.csv` (BKS), `effort.csv`, `trajectories.csv.gz` (ILS/GRASP), `best_allocations.csv`, `data_quality.txt` |
| `analysis.py` | Gera `outputs/tables/*.csv` e `*.tex` e `outputs/figures/*.pdf/.png` |
| `latex/secao_experimentos.tex` | Seções *Projeto Experimental*, *Resultados* e *Ameaças à Validade* para o artigo |
| `latex/referencias_experimentos.bib` | Referências dos materiais metodológicos |

## O que cada etapa da análise faz e por quê

| Etapa | Procedimento | Fonte |
|---|---|---|
| Medida de qualidade | RPD = 100·(BKS − f)/BKS; BKS = melhor valor entre todas as execuções (não há ótimo conhecido) | [S 9], [B 4.3] |
| Estatísticas descritivas | média, mediana, dp, IQR, mín., máx. por instância | [S 12], [E 4.1] |
| Comparação entre instâncias | um valor (RPD médio) por método–instância; Friedman + Iman–Davenport, Aligned, Quade; 1×N com controle; N×N Shaffer/Bergmann; contraste de medianas; n = 15 ≥ 2k = 8 | [D 4–6] |
| Sensibilidade | repetição com mediana por instância e com uma observação por localidade (n = 5 < 2k, apenas indicativo) | [E 4.1], [D 6.1] |
| Comparação por instância | Kruskal–Wallis; Mann–Whitney (amostras independentes: as sementes não dão ponto de partida comum), Holm sobre 90 testes; r_rb; IC bootstrap da diferença de medianas | [S 12, 21–23], [B 3] |
| Modelo misto | método fixo, instância aleatória, interação aleatória; F contra a interação; Tukey; diagnóstico dos resíduos | [BB 10.2.2, 10.1] |
| Robustez | dp, amplitude, taxa de sucesso (sementes); falibilidade e aplicabilidade (instâncias) | [E 4.2] |
| Custo fixo ILS × GRASP | melhor valor após E avaliações, a partir das trajetórias | [B 4.3–4.4], [S 11] |
| Tempo até o alvo | TTT com P_i = (i − 0,5)/n; perfis de desempenho (log2 τ) | [S 35–38], [B 5] |
| Esforço | avaliações (medidas / derivadas do código) e tempo reportado | [B 4.1], [S 10] |

Notas de implementação conferidas contra [D]: (i) o erro-padrão do Aligned Ranks que reproduz a Tabela 14 é √(k(kn+1)/6), não √(k(n+1)/6) como impresso na Eq. 13; (ii) a média do algoritmo C na Tabela 8 é 1,75 (impresso 1,250); (iii) os APVs de Bergmann são mantidos monótonos, como na Tabela 20.

## Problemas encontrados nos dados (ver `outputs/data/data_quality.txt`)

1. `pso/av_paulista/objectives.txt` acumula blocos de outras localidades (caminho de saída fixo `local = "av_paulista"` em `pso.py` e `brkga.py`); só o primeiro bloco de cada perfil é usado. Três reexecuções de Av. Paulista reproduziram os mesmos valores.
2. O BRKGA salvou como “melhor alocação” a execução da primeira semente, não a melhor das 100.
3. Em 3 instâncias a melhor alocação do PSO soma 97 ou 99 POIs (o PSO aceita ≤ B); BRKGA e PSO só geram Δx ∈ {0,1}.
4. Esforço desigual: ILS ≈ 29 mil avaliações, GRASP ≈ 9,6 mil (para por estagnação), BRKGA 51,1 mil, PSO ≥ 30 mil (torneio não contabilizado).
5. Tempos de BRKGA e PSO só existem como média por instância, medidos em sessões diferentes.
