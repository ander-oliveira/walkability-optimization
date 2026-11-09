"""
Step 2 - statistical analysis, tables and figures.

Reads outputs/data/*.csv (from build_dataset.py) and writes
outputs/tables/*.csv|.tex and outputs/figures/*.pdf|.png.

Each block names the source that prescribes it:
  [D]  Derrac et al. (2011) Swarm Evol. Comput. 1:3-18
  [E]  Eiben & Smit (2011) Swarm Evol. Comput. 1:19-31
  [B]  Beiranvand, Hare & Lucet (2017) arXiv:1709.08242 / Optim. Eng.
  [BB] Bartz-Beielstein et al. (eds.) (2010) Experimental Methods for the
       Analysis of Optimization Algorithms, Springer (chapter.section)
  [S]  PCO212 slides, "7 - Projeto e Analise de Experimentos" (slide no.)

Usage:  python analysis.py --out outputs [--boot 2000]
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import ea_stats as es

ALGS = ["ILS", "GRASP", "BRKGA", "PSO"]
COLORS = {"ILS": "#2a78d6", "GRASP": "#eb6834", "BRKGA": "#1baf7a", "PSO": "#eda100"}
MARKERS = {"ILS": "o", "GRASP": "s", "BRKGA": "^", "PSO": "D"}
LOC_PT = {"av_paulista": "Av. Paulista", "itajuba_centro": "Itajubá", "metro_ana_rosa": "Ana Rosa",
          "milao_italia": "Milão", "shinjuku_toquio": "Shinjuku"}
PROF_PT = {"athlete": "atleta", "average_adult": "adulto", "elderly": "idoso"}
LOCS = list(LOC_PT)
PROFS = ["athlete", "average_adult", "elderly"]
ALPHA = 0.05

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200})


# ---------------------------------------------------------------- helpers
def fmt(x, d=2):
    if pd.isna(x):
        return "--"
    return f"{x:.{d}f}".replace(".", ",")


def fmt_int(x):
    return f"{x:,.0f}".replace(",", ".")


def fmt_p(p):
    if pd.isna(p):
        return "--"
    if p < 1e-16:
        return "$<10^{-16}$"
    if p < 1e-4:
        m, e = f"{p:.1e}".split("e")
        return f"${m.replace('.', ',')}\\times10^{{{int(e)}}}$"
    return fmt(p, 4)


def inst_label(inst):
    l, p = inst.split("/")
    return f"{LOC_PT[l]} / {PROF_PT[p]}"


def save_tex(df, path, caption, label, colfmt=None):
    cols = list(df.columns)
    colfmt = colfmt or ("l" * 1 + "r" * (len(cols) - 1))
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", f"\\begin{{tabular}}{{@{{}}{colfmt}@{{}}}}", "\\toprule",
             " & ".join(cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v) for v in r.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def savefig(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"))
    plt.close(fig)


def instance_order(runs):
    return [f"{l}/{p}" for l in LOCS for p in PROFS]


# ---------------------------------------------------------------- 1. descriptive
def descriptive(runs, T):
    """[S 12] mean, median, sd, min, max; [E 4.1] median when variance is large."""
    g = runs.groupby(["instance", "algorithm"])
    d = g["best"].agg(mean="mean", sd="std", median="median",
                      q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
                      min="min", max="max").reset_index()
    d["iqr"] = d["q3"] - d["q1"]
    d = d.merge(g["rpd"].agg(rpd_mean="mean", rpd_median="median").reset_index(), on=["instance", "algorithm"])
    d.to_csv(os.path.join(T, "descriptive.csv"), index=False)
    rows = []
    for inst in instance_order(runs):
        x = d[d.instance == inst].set_index("algorithm")
        best_mean = x["mean"].max()
        r = [inst_label(inst)]
        for a in ALGS:
            s = f"{fmt(x.loc[a, 'mean'])} ({fmt(x.loc[a, 'sd'])})"
            if np.isclose(x.loc[a, "mean"], best_mean):
                s = "\\textbf{" + s + "}"
            r.append(s)
        r.append(fmt(runs.loc[runs.instance == inst, "bks"].iloc[0]))
        rows.append(r)
    save_tex(pd.DataFrame(rows, columns=["Instância"] + [f"{a} média (dp)" for a in ALGS] + ["BKS"]),
             os.path.join(T, "tab_descriptive.tex"),
             "Valor final de $\\overline{\\IQC}_f$ por instância: média (desvio padrão) sobre 100 execuções "
             "independentes; BKS é a melhor solução conhecida entre todas as execuções de todos os métodos.",
             "tab:descriptive")
    return d


# ---------------------------------------------------------------- 2. multi-problem [D]
def multiproblem(runs, T, F, stat="mean", tag="", instances_level="instance"):
    """[D 4-5] Friedman/Iman-Davenport, Aligned, Quade + post-hoc; [D 4.4] contrast."""
    key = instances_level
    agg = runs.groupby([key, "algorithm"])["rpd"].agg(stat).unstack()[ALGS]
    X = agg.values                      # RPD: lower is better
    n, k = X.shape
    res = {"friedman": es.friedman_test(X), "aligned": es.friedman_aligned_test(X), "quade": es.quade_test(X)}
    omni = []
    for name, r in res.items():
        row = {"test": name, "statistic": r["statistic"], "p_value": r["p_value"], "n": n, "k": k}
        if name == "friedman":
            row.update({"iman_davenport": r["iman_davenport"], "p_iman_davenport": r["p_iman_davenport"]})
        for a, v in zip(ALGS, r["avg_ranks"]):
            row["rank_" + a] = v
        omni.append(row)
    omni = pd.DataFrame(omni)
    omni.to_csv(os.path.join(T, f"omnibus{tag}.csv"), index=False)

    # 1 x N with control = best Friedman rank [D 4.3]
    ctrl = int(np.argmin(res["friedman"]["avg_ranks"]))
    post = []
    for name, r in res.items():
        others = [j for j in range(k) if j != ctrl]
        ps = [es.unadjusted_p(r["avg_ranks"], j, ctrl, name, n)[0] for j in others]
        zs = [es.unadjusted_p(r["avg_ranks"], j, ctrl, name, n)[1] for j in others]
        adj = es.adjust_1xN(ps)
        for idx, j in enumerate(others):
            row = {"test": name, "control": ALGS[ctrl], "versus": ALGS[j], "z": zs[idx]}
            for m in ["unadjusted", "bonferroni", "holm", "hochberg", "hommel", "holland", "finner", "li"]:
                row[m] = adj[m][idx]
            post.append(row)
    post = pd.DataFrame(post)
    post.to_csv(os.path.join(T, f"posthoc_1xN{tag}.csv"), index=False)

    # N x N on Friedman ranks [D 5]
    fr = res["friedman"]["avg_ranks"]
    pp = {(i, j): es.unadjusted_p(fr, i, j, "friedman", n)[0] for i, j in combinations(range(k), 2)}
    adj = es.adjust_NxN(pp, k)
    nxn = pd.DataFrame([{"hypothesis": f"{ALGS[i]} vs {ALGS[j]}", "better": ALGS[i] if fr[i] < fr[j] else ALGS[j],
                         **{m: adj[m][(i, j)] for m in adj}} for (i, j) in sorted(pp, key=pp.get)])
    nxn.to_csv(os.path.join(T, f"posthoc_NxN{tag}.csv"), index=False)

    # contrast estimation on RPD [D 4.4]
    E = pd.DataFrame(es.contrast_estimation(X), index=ALGS, columns=ALGS)
    E.to_csv(os.path.join(T, f"contrast{tag}.csv"))

    # pairwise Wilcoxon + sign test over instances [D 3]; Holm over the 6 pairs
    pw = []
    for i, j in combinations(range(k), 2):
        w = es.wilcoxon_signed_rank(X[:, i], X[:, j])
        s = es.sign_test(X[:, i], X[:, j])
        pw.append({"A": ALGS[i], "B": ALGS[j], "A_wins": s["wins"], "ties": s["ties"], "A_losses": s["losses"],
                   "p_sign": s["p_value"], "R_plus_A": w["R_plus"], "R_minus_A": w["R_minus"], "p_wilcoxon": w["p_value"]})
    pw = pd.DataFrame(pw)
    pw["p_wilcoxon_holm"] = es.adjust_1xN(pw["p_wilcoxon"].values)["holm"]
    pw.to_csv(os.path.join(T, f"pairwise_instances{tag}.csv"), index=False)
    return {"agg": agg, "res": res, "omni": omni, "post": post, "nxn": nxn, "E": E, "pw": pw, "ctrl": ALGS[ctrl], "n": n}


def tex_multiproblem(mp, T):
    o = mp["omni"]
    rows = []
    for _, r in o.iterrows():
        name = {"friedman": "Friedman", "aligned": "Friedman Aligned", "quade": "Quade"}[r["test"]]
        rows.append([name] + [fmt(r["rank_" + a]) for a in ALGS] + [fmt(r["statistic"], 3), fmt_p(r["p_value"])])
        if r["test"] == "friedman":
            rows.append(["Iman--Davenport", "", "", "", "", fmt(r["iman_davenport"], 3), fmt_p(r["p_iman_davenport"])])
    save_tex(pd.DataFrame(rows, columns=["Teste"] + ALGS + ["Estatística", "$p$"]),
             os.path.join(T, "tab_omnibus.tex"),
             f"Rankings médios e testes omnibus sobre o RPD médio por instância ($n={mp['n']}$ instâncias, $k=4$ métodos); "
             "menor ranking indica melhor desempenho.", "tab:omnibus")
    p = mp["post"]
    rows = []
    for _, r in p.iterrows():
        name = {"friedman": "Friedman", "aligned": "Aligned", "quade": "Quade"}[r["test"]]
        rows.append([name, f"{r['control']} vs {r['versus']}", fmt(r["z"], 3)] +
                    [fmt_p(r[m]) for m in ["unadjusted", "holm", "hochberg", "finner", "li"]])
    save_tex(pd.DataFrame(rows, columns=["Ranking", "Comparação", "$z$", "$p$ não ajust.", "Holm", "Hochberg", "Finner", "Li"]),
             os.path.join(T, "tab_posthoc_1xN.tex"),
             f"Comparações $1\\times N$ com o método de menor ranking ({mp['ctrl']}) como controle: "
             "valores de $p$ ajustados.", "tab:posthoc1xn", colfmt="llrrrrrr")
    n = mp["nxn"]
    rows = [[r["hypothesis"], r["better"]] + [fmt_p(r[m]) for m in ["unadjusted", "nemenyi", "holm", "shaffer", "bergmann"]]
            for _, r in n.iterrows()]
    save_tex(pd.DataFrame(rows, columns=["Hipótese", "Melhor ranking", "$p$ não ajust.", "Nemenyi", "Holm", "Shaffer", "Bergmann"]),
             os.path.join(T, "tab_posthoc_NxN.tex"),
             "Comparações $N\\times N$ sobre os rankings de Friedman: valores de $p$ ajustados.",
             "tab:posthocnxn", colfmt="llrrrrr")
    E = mp["E"]
    rows = [[a] + [fmt(E.loc[a, b], 3) for b in ALGS] for a in ALGS]
    save_tex(pd.DataFrame(rows, columns=["Método"] + ALGS), os.path.join(T, "tab_contrast.tex"),
             "Estimação de contraste baseada em medianas sobre o RPD (pontos percentuais). "
             "Valor negativo na linha indica RPD menor (melhor) que o método da coluna.", "tab:contrast")
    w = mp["pw"]
    rows = [[f"{r['A']} vs {r['B']}", f"{r['A_wins']}/{r['ties']}/{r['A_losses']}", fmt_p(r["p_sign"]),
             f"{fmt(r['R_plus_A'],1)} / {fmt(r['R_minus_A'],1)}", fmt_p(r["p_wilcoxon"]), fmt_p(r["p_wilcoxon_holm"])]
            for _, r in w.iterrows()]
    save_tex(pd.DataFrame(rows, columns=["Par (A vs B)", "V/E/D de A", "$p$ sinal", "$R^+$ / $R^-$", "$p$ Wilcoxon", "$p$ Holm"]),
             os.path.join(T, "tab_pairwise_instances.tex"),
             "Comparações pareadas sobre as 15 instâncias (RPD médio): vitórias/empates/derrotas, teste do sinal "
             "e teste de Wilcoxon com ajuste de Holm.", "tab:pairwise", colfmt="lrrrrr")


# ---------------------------------------------------------------- 3. single-problem
def rank_biserial(a, b):
    """Effect size for Mann-Whitney; > 0 means a tends to be larger than b."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def single_problem(runs, T, n_boot, rng):
    """Per instance: Kruskal-Wallis then pairwise Mann-Whitney (independent samples) [S 12, 21];
    effect size and bootstrap CI [S 20, 22]; Holm across all pairwise tests [D 4.3]."""
    kw, pw = [], []
    for inst in instance_order(runs):
        x = {a: runs.loc[(runs.instance == inst) & (runs.algorithm == a), "best"].values for a in ALGS}
        h = stats.kruskal(*x.values())
        kw.append({"instance": inst, "H": h.statistic, "p_value": h.pvalue})
        for a, b in combinations(ALGS, 2):
            p = stats.mannwhitneyu(x[a], x[b], alternative="two-sided").pvalue
            ia = rng.integers(0, len(x[a]), (n_boot, len(x[a])))
            ib = rng.integers(0, len(x[b]), (n_boot, len(x[b])))
            md = np.median(x[a][ia], axis=1) - np.median(x[b][ib], axis=1)
            pw.append({"instance": inst, "A": a, "B": b, "median_A": np.median(x[a]), "median_B": np.median(x[b]),
                       "median_diff": np.median(x[a]) - np.median(x[b]),
                       "ci_low": np.percentile(md, 2.5), "ci_high": np.percentile(md, 97.5),
                       "r_rb": rank_biserial(x[a], x[b]), "p_value": p})
    kw = pd.DataFrame(kw)
    kw["p_holm"] = es.adjust_1xN(kw["p_value"].values)["holm"]
    pw = pd.DataFrame(pw)
    pw["p_holm"] = es.adjust_1xN(pw["p_value"].values)["holm"]
    kw.to_csv(os.path.join(T, "kruskal_per_instance.csv"), index=False)
    pw.to_csv(os.path.join(T, "mannwhitney_per_instance.csv"), index=False)
    # summary "<, =, >" per pair [S 27]
    summ = []
    for a, b in combinations(ALGS, 2):
        s = pw[(pw.A == a) & (pw.B == b)]
        sig = s.p_holm < ALPHA
        summ.append({"A": a, "B": b, "A_better": int(((s.median_diff > 0) & sig).sum()),
                     "no_difference": int((~sig).sum()), "B_better": int(((s.median_diff < 0) & sig).sum()),
                     "median_abs_r_rb": float(np.median(np.abs(s.r_rb)))})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(T, "single_problem_summary.csv"), index=False)
    rows = [[f"{r['A']} vs {r['B']}", r["A_better"], r["no_difference"], r["B_better"], fmt(r["median_abs_r_rb"])]
            for _, r in summ.iterrows()]
    save_tex(pd.DataFrame(rows, columns=["Par (A vs B)", "A melhor", "Sem diferença", "B melhor", "Mediana de $|r_{rb}|$"]),
             os.path.join(T, "tab_single_problem_summary.tex"),
             "Número de instâncias (de 15) em que cada método foi significativamente melhor "
             "(Mann--Whitney bilateral, Holm sobre os 90 testes, $\\alpha=0{,}05$).", "tab:singleproblem")
    return kw, pw, summ


# ---------------------------------------------------------------- 4. mixed model [BB 10.2.2]
def mixed_model(runs, T, F):
    """Balanced mixed-effects ANOVA: algorithm fixed, instance random, runs replicated.
    Expected mean squares and F ratios of [BB Table 10.1]; Tukey intervals [BB 10.2.2]."""
    order = instance_order(runs)
    Y = np.stack([[runs.loc[(runs.instance == i) & (runs.algorithm == a), "rpd"].values for a in ALGS] for i in order])
    q, h, r = Y.shape
    gm = Y.mean()
    mi = Y.mean(axis=(1, 2)); mj = Y.mean(axis=(0, 2)); mij = Y.mean(axis=2)
    ss = {"algorithm": q * r * np.sum((mj - gm) ** 2), "instance": h * r * np.sum((mi - gm) ** 2),
          "interaction": r * np.sum((mij - mi[:, None] - mj[None, :] + gm) ** 2),
          "error": np.sum((Y - mij[:, :, None]) ** 2)}
    df = {"algorithm": h - 1, "instance": q - 1, "interaction": (h - 1) * (q - 1), "error": h * q * (r - 1)}
    ms = {k: ss[k] / df[k] for k in ss}
    F_ = {"algorithm": ms["algorithm"] / ms["interaction"], "instance": ms["instance"] / ms["interaction"],
          "interaction": ms["interaction"] / ms["error"]}
    den = {"algorithm": "interaction", "instance": "interaction", "interaction": "error"}
    rows = []
    for k_ in ["algorithm", "instance", "interaction", "error"]:
        row = {"source": k_, "df": df[k_], "SS": ss[k_], "MS": ms[k_]}
        if k_ in F_:
            row["F"] = F_[k_]; row["p_value"] = stats.f.sf(F_[k_], df[k_], df[den[k_]])
        rows.append(row)
    # the same table with every factor fixed (classical ANOVA), to show [BB 10.5]
    rows.append({"source": "algorithm (all factors fixed: F=MS/MSE)", "df": df["algorithm"],
                 "F": ms["algorithm"] / ms["error"],
                 "p_value": stats.f.sf(ms["algorithm"] / ms["error"], df["algorithm"], df["error"])})
    anova = pd.DataFrame(rows)
    vc = {"sigma2_error": ms["error"], "sigma2_interaction": max(0.0, (ms["interaction"] - ms["error"]) / r),
          "sigma2_instance": max(0.0, (ms["instance"] - ms["interaction"]) / (h * r))}
    anova.to_csv(os.path.join(T, "mixed_anova.csv"), index=False)
    pd.Series(vc).to_csv(os.path.join(T, "mixed_variance_components.csv"))
    sD = np.sqrt(2 * ms["interaction"] / (q * r))
    Tq = stats.studentized_range.ppf(1 - ALPHA, h, (h - 1) * (q - 1)) / np.sqrt(2)
    tk = []
    for i, j in combinations(range(h), 2):
        D = mj[i] - mj[j]
        tk.append({"pair": f"{ALGS[i]} - {ALGS[j]}", "diff_rpd": D, "low": D - Tq * sD, "high": D + Tq * sD,
                   "significant": abs(D) > Tq * sD})
    tk = pd.DataFrame(tk)
    tk.to_csv(os.path.join(T, "mixed_tukey.csv"), index=False)
    resid = (Y - mij[:, :, None]).ravel()
    diag = {"residual_skewness": stats.skew(resid), "residual_excess_kurtosis": stats.kurtosis(resid)}
    pd.Series(diag).to_csv(os.path.join(T, "mixed_diagnostics.csv"))

    rows = []
    names = {"algorithm": "Método (fixo)", "instance": "Instância (aleatório)", "interaction": "Método $\\times$ instância",
             "error": "Erro (execuções)"}
    for _, x in anova.iloc[:4].iterrows():
        rows.append([names[x["source"]], int(x["df"]), fmt(x["MS"], 3), fmt(x.get("F"), 2) if not pd.isna(x.get("F")) else "--",
                     fmt_p(x.get("p_value")) if not pd.isna(x.get("p_value")) else "--"])
    save_tex(pd.DataFrame(rows, columns=["Fonte", "gl", "QM", "$F$", "$p$"]), os.path.join(T, "tab_mixed_anova.tex"),
             "ANOVA de efeitos mistos sobre o RPD (método fixo, instância aleatória, 100 réplicas); "
             "o efeito de método é testado contra a interação método $\\times$ instância.", "tab:mixedanova", colfmt="lrrrr")

    # paired comparison plot [BB Fig 10.5]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for j, a in enumerate(ALGS):
        ax.errorbar(mj[j], j, xerr=Tq * sD / 2, fmt=MARKERS[a], color=COLORS[a], ms=6, capsize=3, lw=1.5)
    ax.set_yticks(range(h)); ax.set_yticklabels(ALGS); ax.invert_yaxis()
    ax.set_xlabel("RPD médio (%) ± T·s[D]/2 (Tukey, 95%)")
    savefig(fig, F, "mixed_tukey")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sres = np.sort((resid - resid.mean()) / resid.std())
    th = stats.norm.ppf((np.arange(1, len(sres) + 1) - 0.5) / len(sres))
    ax.plot(th, sres, ".", ms=1.5, color="#555555")
    ax.plot([-4, 4], [-4, 4], color="#2a78d6", lw=1)
    ax.set_xlabel("Quantis teóricos N(0,1)"); ax.set_ylabel("Resíduos padronizados")
    savefig(fig, F, "mixed_qq")
    return anova, vc, tk, diag


# ---------------------------------------------------------------- 5. robustness [E 4.2]
def robustness(runs, T, thresholds=(1.0, 2.0, 5.0)):
    rows = []
    for (inst, a), x in runs.groupby(["instance", "algorithm"]):
        row = {"instance": inst, "algorithm": a, "sd": x["best"].std(),
               "iqr": x["best"].quantile(.75) - x["best"].quantile(.25),
               "instability_max_minus_min": x["best"].max() - x["best"].min()}
        for t in thresholds:
            row[f"success_rate_rpd_le_{t:g}"] = float((x["rpd"] <= t).mean())
        rows.append(row)
    seed_rob = pd.DataFrame(rows)
    seed_rob.to_csv(os.path.join(T, "robustness_seeds.csv"), index=False)
    m = runs.groupby(["algorithm", "instance"])["rpd"].mean().unstack()
    prob = pd.DataFrame({"fallibility_max_minus_min_mean_rpd": m.max(axis=1) - m.min(axis=1),
                         "worst_instance_mean_rpd": m.max(axis=1), "best_instance_mean_rpd": m.min(axis=1),
                         **{f"applicability_instances_mean_rpd_le_{t:g}": (m <= t).sum(axis=1) for t in thresholds}})
    prob = prob.loc[ALGS]
    prob.to_csv(os.path.join(T, "robustness_instances.csv"))
    agg = seed_rob.groupby("algorithm").agg(
        sd_median=("sd", "median"), instability_median=("instability_max_minus_min", "median"),
        **{f"SR{t:g}": (f"success_rate_rpd_le_{t:g}", "mean") for t in thresholds}).loc[ALGS]
    rows = []
    for a in ALGS:
        rows.append([a, fmt(agg.loc[a, "sd_median"], 3), fmt(agg.loc[a, "instability_median"], 2)] +
                    [fmt(100 * agg.loc[a, f"SR{t:g}"], 1) for t in thresholds] +
                    [fmt(prob.loc[a, "fallibility_max_minus_min_mean_rpd"], 2)] +
                    [str(int(prob.loc[a, f"applicability_instances_mean_rpd_le_{t:g}"])) for t in thresholds[1:]])
    cols = (["Método", "dp", "Amplitude"] + [f"TS$_{{{t:g}}}$ (\\%)" for t in thresholds] +
            ["Falib."] + [f"Inst.$_{{{t:g}}}$" for t in thresholds[1:]])
    save_tex(pd.DataFrame(rows, columns=cols), os.path.join(T, "tab_robustness.tex"),
             "Robustez às sementes: medianas, entre instâncias, do desvio padrão (dp) e da amplitude (máx.$-$mín.) de "
             "$\\overline{\\IQC}_f$; TS$_t$ = taxa de sucesso, \\% das 1.500 execuções com RPD $\\le t\\%$. Robustez às "
             "instâncias: Falib. = amplitude do RPD médio entre instâncias (p.p.); Inst.$_t$ = número de instâncias com RPD "
             "médio $\\le t\\%$.", "tab:robustness")
    return seed_rob, prob


# ---------------------------------------------------------------- 6. figures (quality)
def fig_ecdf(runs, F):
    """[E 4.1] ECDF of solution quality; [B 5] accuracy-profile reading."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for a in ALGS:
        x = np.sort(runs.loc[runs.algorithm == a, "rpd"].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLORS[a], lw=1.6, label=a)
        i = int(0.5 * len(x))
        ax.plot(x[i], 0.5, MARKERS[a], color=COLORS[a], ms=4)
    ax.set_xlabel("RPD em relação à BKS (%)"); ax.set_ylabel("Fração das 1.500 execuções")
    ax.set_xlim(left=0); ax.legend(loc="lower right")
    savefig(fig, F, "ecdf_rpd")


def fig_box(runs, F):
    """[S 32] boxplots per instance."""
    fig, axes = plt.subplots(3, 5, figsize=(7.1, 4.6), sharey=False)
    for c, l in enumerate(LOCS):
        for r_, p in enumerate(PROFS):
            ax = axes[r_, c]
            data = [runs.loc[(runs.location == l) & (runs.profile == p) & (runs.algorithm == a), "rpd"].values for a in ALGS]
            bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                            flierprops={"markersize": 1.5, "markeredgecolor": "#888888"},
                            medianprops={"color": "#222222", "lw": 1})
            for patch, a in zip(bp["boxes"], ALGS):
                patch.set_facecolor(COLORS[a]); patch.set_edgecolor("white"); patch.set_alpha(0.9)
            ax.set_xticks(range(1, 5)); ax.set_xticklabels(ALGS, rotation=90, fontsize=6)
            if r_ == 0:
                ax.set_title(LOC_PT[l], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{PROF_PT[p]}\nRPD (%)")
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, F, "box_rpd_instances")


def fig_heatmap(runs, F):
    """[S 28] heatmap of mean RPD (sequential, one hue)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    m = runs.groupby(["instance", "algorithm"])["rpd"].mean().unstack().loc[instance_order(runs), ALGS]
    fig, ax = plt.subplots(figsize=(3.4, 4.0))
    im = ax.imshow(m.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(ALGS)
    ax.set_yticks(range(len(m))); ax.set_yticklabels([inst_label(i) for i in m.index], fontsize=6.5)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, fmt(v, 1), ha="center", va="center", fontsize=6,
                    color="white" if v > np.nanmax(m.values) * 0.45 else "#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.05); cb.set_label("RPD médio (%)")
    savefig(fig, F, "heatmap_rpd")


def fig_ranks_cd(mp, F):
    """Average Friedman ranks with Bonferroni-Dunn critical difference [D 6.2]."""
    R = mp["res"]["friedman"]["avg_ranks"]; n = mp["n"]; k = len(R)
    qa = stats.norm.ppf(1 - ALPHA / (2 * (k - 1)))
    cd = qa * np.sqrt(k * (k + 1) / (6 * n))
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    order = np.argsort(R)
    for pos, j in enumerate(order):
        ax.plot(R[j], pos, MARKERS[ALGS[j]], color=COLORS[ALGS[j]], ms=7)
        ax.text(R[j] + 0.15, pos, f"{ALGS[j]} ({fmt(R[j])})", va="center", fontsize=7)
    best = R[order[0]]
    ax.axvspan(best, best + cd, color="#e6e6e6", zorder=0)
    ax.set_yticks([]); ax.set_xlim(0.8, k + 0.6); ax.invert_yaxis()
    ax.set_xlabel(f"Ranking médio de Friedman (faixa cinza: DC de Bonferroni–Dunn = {fmt(cd)})")
    savefig(fig, F, "friedman_ranks_cd")
    return cd


def fig_interaction(runs, F):
    """[S 31a] interaction plot profile x algorithm."""
    m = runs.groupby(["profile", "algorithm"])["rpd"].mean().unstack().loc[PROFS, ALGS]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for a in ALGS:
        ax.plot(range(3), m[a].values, marker=MARKERS[a], color=COLORS[a], lw=2, ms=6, label=a)
    ax.set_xticks(range(3)); ax.set_xticklabels([PROF_PT[p] for p in PROFS])
    ax.set_ylabel("RPD médio (%)"); ax.set_xlabel("Perfil de pedestre"); ax.legend(ncol=4, loc="upper center",
                                                                                 bbox_to_anchor=(0.5, 1.18))
    savefig(fig, F, "interaction_profile")


def fig_tradeoff(runs, effort, F):
    """[S 31b] trade-off between quality and reported mean time per instance."""
    q = runs.groupby(["algorithm", "location", "profile"])["rpd"].mean().reset_index()
    e = effort[["algorithm", "location", "profile", "mean_runtime_s_reported"]]
    d = q.merge(e, on=["algorithm", "location", "profile"])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for a in ALGS:
        x = d[d.algorithm == a]
        ax.scatter(x["mean_runtime_s_reported"], x["rpd"], marker=MARKERS[a], s=22, color=COLORS[a],
                   edgecolor="white", linewidth=0.6, label=a)
    ax.set_xscale("log"); ax.set_xlabel("Tempo médio por execução reportado (s, escala log)")
    ax.set_xticks([10, 20, 50, 100]); ax.set_xticklabels(["10", "20", "50", "100"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("RPD médio (%)"); ax.legend(loc="upper right")
    savefig(fig, F, "tradeoff_quality_time")


# ---------------------------------------------------------------- 7. anytime ILS vs GRASP
def best_at(traj, budgets, col="eval_count"):
    """Best-so-far of each run at each budget; NaN before the first record.
    A run that had already stopped keeps its final value (flag 'stopped')."""
    out = []
    for key, t in traj.groupby(["algorithm", "location", "profile", "seed"], sort=False):
        e = t[col].values; b = np.maximum.accumulate(t["best_sum_iqc"].values)
        for B in budgets:
            idx = np.searchsorted(e, B, side="right") - 1
            out.append((*key, B, b[idx] if idx >= 0 else np.nan, bool(e[-1] < B)))
    return pd.DataFrame(out, columns=["algorithm", "location", "profile", "seed", "budget", "best", "stopped"])


def anytime(runs, traj, T, F):
    """Fixed-cost comparison of ILS and GRASP at equal numbers of evaluations
    [B 4.3 fixed-cost; B 4.4 'same stopping conditions'; S 11 equal budgets]."""
    traj = traj.sort_values(["algorithm", "location", "profile", "seed", "eval_count"])
    gstop = runs.loc[runs.algorithm == "GRASP", "evaluations"].min()
    budgets = [500, 1000, 2000, 3000, 4000, 8000, 15000, 30000]
    ba = best_at(traj, budgets)
    ba["instance"] = ba["location"] + "/" + ba["profile"]
    bks = runs.groupby("instance")["bks"].first()
    ba["rpd"] = 100 * (ba["instance"].map(bks) - ba["best"]) / ba["instance"].map(bks)
    ba.to_csv(os.path.join(T, "anytime_best_at_budget.csv.gz"), index=False, compression="gzip")
    rows = []
    for B in budgets:
        x = ba[ba.budget == B]
        m = x.groupby(["instance", "algorithm"])["rpd"].mean().unstack()[["ILS", "GRASP"]]
        w = es.wilcoxon_signed_rank(m["ILS"].values, m["GRASP"].values)
        s = es.sign_test(m["ILS"].values, m["GRASP"].values)
        better = {"ILS": 0, "GRASP": 0, "none": 0}
        ps = []
        for inst, g in x.groupby("instance"):
            a_ = g.loc[g.algorithm == "ILS", "best"].dropna().values; b_ = g.loc[g.algorithm == "GRASP", "best"].dropna().values
            ps.append((inst, stats.mannwhitneyu(a_, b_).pvalue, np.median(a_) - np.median(b_)))
        adj = es.adjust_1xN([p for _, p, _ in ps])["holm"]
        for (inst, p, d), pa in zip(ps, adj):
            better["none" if pa >= ALPHA else ("ILS" if d > 0 else "GRASP")] += 1
        rows.append({"budget_evals": B, "grasp_runs_already_stopped_pct": 100 * x.loc[x.algorithm == "GRASP", "stopped"].mean(),
                     "ILS_mean_rpd": m["ILS"].mean(), "GRASP_mean_rpd": m["GRASP"].mean(),
                     "instances_ILS_better": better["ILS"], "instances_GRASP_better": better["GRASP"],
                     "instances_no_diff": better["none"], "ILS_instance_wins": s["wins"],
                     "p_wilcoxon_instances": w["p_value"]})
    fb = pd.DataFrame(rows)
    fb.to_csv(os.path.join(T, "fixed_budget_ils_grasp.csv"), index=False)
    trows = [[f"{int(r['budget_evals']):,}".replace(",", "."), fmt(r["grasp_runs_already_stopped_pct"], 0),
              fmt(r["ILS_mean_rpd"]), fmt(r["GRASP_mean_rpd"]),
              f"{int(r['instances_ILS_better'])}/{int(r['instances_no_diff'])}/{int(r['instances_GRASP_better'])}",
              fmt_p(r["p_wilcoxon_instances"])] for _, r in fb.iterrows()]
    save_tex(pd.DataFrame(trows, columns=["Avaliações", "GRASP já parado (\\%)", "RPD ILS", "RPD GRASP",
                                          "ILS/sem dif./GRASP", "$p$ Wilcoxon (15 inst.)"]),
             os.path.join(T, "tab_fixed_budget.tex"),
             "ILS e GRASP comparados com o mesmo número de avaliações da função objetivo (melhor valor até o orçamento). "
             f"Até {int(gstop):,} avaliações todas as execuções do GRASP estão ativas; acima disso, as já encerradas "
             "mantêm o valor final.".replace(",", "."), "tab:fixedbudget")

    # convergence plot, median and IQR [B 5 convergence plot; S 30]
    grid = np.unique(np.concatenate([np.linspace(0, 4000, 41), np.linspace(4000, 30000, 53)])).astype(int)[1:]
    cg = best_at(traj, grid)
    cg["instance"] = cg["location"] + "/" + cg["profile"]
    fig, axes = plt.subplots(3, 5, figsize=(7.1, 4.6))
    for c, l in enumerate(LOCS):
        for r_, p in enumerate(PROFS):
            ax = axes[r_, c]
            for a in ["ILS", "GRASP"]:
                g = cg[(cg.location == l) & (cg.profile == p) & (cg.algorithm == a)]
                s = g.groupby("budget")["best"].quantile([.25, .5, .75]).unstack()
                ax.plot(s.index, s[0.5], color=COLORS[a], lw=1.4, label=a)
                ax.fill_between(s.index, s[0.25], s[0.75], color=COLORS[a], alpha=0.2, lw=0)
            ax.axvline(gstop, color="#999999", lw=0.6, ls="--")
            ax.tick_params(labelsize=6)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
            if r_ == 0: ax.set_title(LOC_PT[l], fontsize=8)
            if c == 0: ax.set_ylabel(f"{PROF_PT[p]}\n" + r"melhor $\overline{IQC}_f$")
            if r_ == 2: ax.set_xlabel("avaliações", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    savefig(fig, F, "convergence_ils_grasp")
    return fb


def ttt_points(times, reached, n_total):
    """TTT plot points [S 36]: sorted times t_1 <= ... and P_i = (i - 0.5) / n.
    n counts every run, so runs that ended without reaching the target keep the
    curve below 1. Kaplan-Meier [BB 7.2.3] is not used: it needs censoring
    independent of the event, and here censoring is the algorithm's own stop."""
    t = np.sort(np.asarray(times)[np.asarray(reached, bool)])
    return t, (np.arange(1, len(t) + 1) - 0.5) / n_total


def ttt_and_profiles(runs, traj, T, F):
    """TTT plots [S 35-36; Aiex, Resende & Ribeiro];
    performance profiles [S 37-38; B 5]. 'Time' = objective evaluations [B 4.1]."""
    traj = traj.sort_values(["algorithm", "location", "profile", "seed", "eval_count"])
    # Target per instance: the lower of the two algorithms' first-quartile final
    # values, so that at least 75 % of the runs of each algorithm reach it and
    # the median time-to-target is defined for both.
    final = runs[runs.algorithm.isin(["ILS", "GRASP"])].groupby(["instance", "algorithm"])["best"].quantile(0.25).unstack()
    targets = {"q25": final.min(axis=1)}
    recs = []
    for (a, l, p, s), t in traj.groupby(["algorithm", "location", "profile", "seed"], sort=False):
        inst = f"{l}/{p}"
        b = np.maximum.accumulate(t["best_sum_iqc"].values)
        for name, tg in targets.items():
            hit = np.nonzero(b >= tg[inst] - 1e-12)[0]
            if len(hit):
                recs.append((a, inst, s, name, t["eval_count"].values[hit[0]], t["elapsed_s"].values[hit[0]], True))
            else:
                recs.append((a, inst, s, name, t["eval_count"].values[-1], t["elapsed_s"].values[-1], False))
    tt = pd.DataFrame(recs, columns=["algorithm", "instance", "seed", "target", "evals", "seconds", "reached"])
    tt.to_csv(os.path.join(T, "time_to_target.csv"), index=False)
    pd.DataFrame(targets).to_csv(os.path.join(T, "ttt_targets.csv"))

    # TTT plots for three contrasting instances, easy target
    show = ["itajuba_centro/average_adult", "av_paulista/elderly", "milao_italia/athlete"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    for ax, inst in zip(axes, show):
        for a in ["ILS", "GRASP"]:
            x = tt[(tt.instance == inst) & (tt.algorithm == a) & (tt.target == "q25")]
            xs, ys = ttt_points(x["evals"].values, x["reached"].values, len(x))
            ax.plot(xs, ys, MARKERS[a], ms=2.5, color=COLORS[a], label=f"{a} ({int(x['reached'].sum())}/{len(x)} atingem)")
        ax.set_title(inst_label(inst), fontsize=8); ax.set_xlabel("avaliações até o alvo"); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("probabilidade acumulada")
    fig.tight_layout()
    savefig(fig, F, "ttt_ils_grasp")

    # performance profiles on the median evaluations-to-target over all runs
    # (inf when fewer than 50 % of the runs reach the target = failed test)
    prof_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    for ax, (name, col) in zip(axes, [("q25", "evals"), ("q25", "seconds")]):
        tps = {}
        for a in ["ILS", "GRASP"]:
            v = []
            for inst in instance_order(runs):
                x = tt[(tt.instance == inst) & (tt.algorithm == a) & (tt.target == name)]
                e = np.where(x["reached"].values, x[col].values, np.inf)
                v.append(float(np.median(e)))
            tps[a] = np.array(v, float)
        best = np.minimum(tps["ILS"], tps["GRASP"])
        taus = np.logspace(0, np.log2(64), 300, base=2)
        for a in ["ILS", "GRASP"]:
            with np.errstate(invalid="ignore"):
                ratio = np.where(np.isinf(best), np.inf, tps[a] / best)
            rho = [(ratio <= tau).mean() for tau in taus]
            ax.step(np.log2(taus), rho, where="post", color=COLORS[a], lw=1.6, label=a)
            for inst, t_, rt in zip(instance_order(runs), tps[a], ratio):
                prof_rows.append({"target": name, "measure": col, "algorithm": a, "instance": inst,
                                  "median_to_target": t_, "ratio": rt})
        ax.set_xlabel(r"$\log_2(\tau)$"); ax.set_ylim(0, 1.02)
        ax.set_title("medida: avaliações até o alvo" if col == "evals" else "medida: segundos até o alvo", fontsize=8)
        ax.legend(loc="lower right")
    axes[0].set_ylabel(r"$\rho_s(\tau)$")
    fig.tight_layout()
    savefig(fig, F, "performance_profile_ils_grasp")
    pd.DataFrame(prof_rows).to_csv(os.path.join(T, "performance_profile_data.csv"), index=False)
    return tt


# ---------------------------------------------------------------- 8. effort
def effort_table(runs, effort, T):
    """[B 4.1] evaluations and running time; [S 10] specify time type."""
    e = effort.groupby("algorithm").agg(evals_mean=("evaluations_mean", "mean"), evals_min=("evaluations_min", "min"),
                                        evals_max=("evaluations_max", "max"), time_mean=("mean_runtime_s_reported", "mean"),
                                        time_min=("mean_runtime_s_reported", "min"), time_max=("mean_runtime_s_reported", "max"),
                                        source=("evaluations_source", "first")).loc[ALGS]
    e.to_csv(os.path.join(T, "effort_summary.csv"))
    rows = []
    for a in ALGS:
        r = e.loc[a]
        if a in ("ILS", "GRASP"):
            ev = f"{fmt_int(r.evals_mean)} [{fmt_int(r.evals_min)}; {fmt_int(r.evals_max)}]"
        elif a == "BRKGA":
            ev = "51.100 (fixo)"
        else:
            ev = "$\\geq$ 30.050"
        rows.append([a, ev, f"{fmt(r.time_mean,1)} [{fmt(r.time_min,1)}; {fmt(r.time_max,1)}]",
                     {"ILS": "30.000 aval., 500 it., 80 sem melhora", "GRASP": "idem (encerrou por estagnação)",
                      "BRKGA": "600 gerações", "PSO": "600 iterações"}[a]])
    save_tex(pd.DataFrame(rows, columns=["Método", "Avaliações por execução", "Tempo médio por execução (s)", "Critério de parada"]),
             os.path.join(T, "tab_effort.tex"),
             "Esforço computacional por execução. ILS e GRASP: avaliações medidas (média [mín.; máx.]); BRKGA: derivado do código "
             "($100+600\\times85$); PSO: limite inferior ($50+600\\times50$), sem contar as avaliações do torneio, não registradas. "
             "Tempos: média por instância reportada pelos próprios scripts, em condições de execução distintas.",
             "tab:effort", colfmt="llll")
    return e


# ---------------------------------------------------------------- 9. solution structure
def structure_table(D, T):
    al = pd.read_csv(os.path.join(D, "best_allocations.csv"))
    g = al.groupby("algorithm").agg(hex_mean=("hexagons_used", "mean"), qmax_mean=("q_max", "mean"),
                                    hhi_mean=("hhi", "mean"), max_cell=("max_per_cell", "max"),
                                    pois_min=("total_pois", "min"), n_below_B=("total_pois", lambda x: int((x < 100).sum())),
                                    stored_best=("stored_is_best_run", "sum")).loc[ALGS]
    g.to_csv(os.path.join(T, "structure_summary.csv"))
    rows = [[a, fmt(r.hex_mean, 1), fmt(r.qmax_mean, 1), fmt(r.hhi_mean, 3), int(r.max_cell),
             f"{int(r.pois_min)} ({int(r.n_below_B)} inst.)", f"{int(r.stored_best)}/15"] for a, r in g.iterrows()]
    save_tex(pd.DataFrame(rows, columns=["Método", "Hexágonos", "$q_{\\max}$", "HHI", "Máx. por célula",
                                         "Menor $\\sum\\Delta x$", "Alocação salva = melhor execução"]),
             os.path.join(T, "tab_structure.tex"),
             "Estrutura das alocações salvas como melhor solução (médias sobre as 15 instâncias). "
             "Máx. por célula = maior $\\Delta x_{i,j}$; BRKGA e PSO só geram $\\Delta x_{i,j}\\in\\{0,1\\}$.",
             "tab:structure")
    return g


# ---------------------------------------------------------------- 10. instances [S 05]
def instances_table(D, T):
    ins = pd.read_csv(os.path.join(D, "instances.csv"))
    ins["order"] = ins["instance"].map({k: i for i, k in enumerate(f"{l}/{p}" for l in LOCS for p in PROFS)})
    ins = ins.sort_values("order")
    rows = [[LOC_PT[r.location], PROF_PT[r.profile], r.n_hex, fmt_int(r.n_impacts), fmt(r.baseline), fmt(r.bks),
             r.bks_algorithm] for r in ins.itertuples()]
    save_tex(pd.DataFrame(rows, columns=["Localidade", "Perfil", "Hexágonos", "Pares origem--destino",
                                         "$\\overline{\\IQC}_b$", "BKS", "BKS obtida por"]),
             os.path.join(T, "tab_instances.tex"),
             "Instâncias: número de hexágonos H3 elegíveis, número de pares origem--destino da matriz de impacto "
             "(identificador \\texttt{impacts} do conjunto de dados), "
             "valor da linha de base e melhor solução conhecida (BKS) entre todas as execuções.", "tab:instances",
             colfmt="llrrrrl")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    D = os.path.join(a.out, "data"); T = os.path.join(a.out, "tables"); F = os.path.join(a.out, "figures")
    os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    runs = pd.read_csv(os.path.join(D, "runs.csv"))
    effort = pd.read_csv(os.path.join(D, "effort.csv"))
    traj = pd.read_csv(os.path.join(D, "trajectories.csv.gz"))

    descriptive(runs, T)
    mp = multiproblem(runs, T, F, "mean")
    tex_multiproblem(mp, T)
    multiproblem(runs, T, F, "median", tag="_median")                       # sensitivity: median [E 4.1]
    multiproblem(runs, T, F, "mean", tag="_by_location", instances_level="location")  # n = 5 < 2k: see [D 6.1]
    cd = fig_ranks_cd(mp, F)
    single_problem(runs, T, a.boot, rng)
    mixed_model(runs, T, F)
    robustness(runs, T)
    fig_ecdf(runs, F); fig_box(runs, F); fig_heatmap(runs, F); fig_interaction(runs, F)
    fig_tradeoff(runs, effort, F)
    effort_table(runs, effort, T)
    anytime(runs, traj, T, F)
    ttt_and_profiles(runs, traj, T, F)
    structure_table(D, T)
    instances_table(D, T)
    print("done; Bonferroni-Dunn CD =", round(cd, 3))


if __name__ == "__main__":
    main()
