"""
Step 1 - consolidate the raw experiment outputs into tidy tables.

Reads the four result trees produced by the project (results/ils, results/grasp,
results/brkga, results/pso) and writes:

  outputs/data/runs.csv           one row per (algorithm, instance, seed)
  outputs/data/instances.csv      one row per instance (baseline, size, BKS)
  outputs/data/effort.csv         declared / measured computational effort
  outputs/data/trajectories.csv.gz anytime data (ILS and GRASP only)
  outputs/data/data_quality.txt   checks performed while reading

Following Beiranvand et al. (2017, Sec. 4.2) every run is stored as a
separate record so that any aggregate can be recomputed later.

Usage:  python build_dataset.py --results ../results --out outputs
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re

import numpy as np
import pandas as pd

LOCATIONS = ["av_paulista", "itajuba_centro", "metro_ana_rosa", "milao_italia", "shinjuku_toquio"]
PROFILES = ["athlete", "average_adult", "elderly"]
ALGORITHMS = ["ILS", "GRASP", "BRKGA", "PSO"]


def _run_dir(root, algo, loc, prof):
    """Return the run_* directory of an instance (latest when repeated)."""
    dirs = sorted(glob.glob(os.path.join(root, algo, loc, prof, "dataset_*", "run_*")))
    return dirs


def read_ils_grasp(root, algo_key, algo_name, log):
    rows, traj = [], []
    cfgs = {}
    for loc in LOCATIONS:
        for prof in PROFILES:
            dirs = _run_dir(root, algo_key, loc, prof)
            # Prefer runs made with binary allocation (config field added with the
            # binary version of ILS/GRASP); older runs have no such field.
            def _is_binary(run_dir):
                c = json.load(open(os.path.join(run_dir, "config", "experiment_config.json")))
                return bool(c.get("runtime_config", {}).get("binary_allocation", False))
            binary_dirs = [x for x in dirs if _is_binary(x)]
            chosen = binary_dirs if binary_dirs else dirs
            if len(dirs) != 1 or not binary_dirs:
                log.append(f"[{algo_name}] {loc}/{prof}: {len(dirs)} run directories, {len(binary_dirs)} binary; "
                           f"using {os.path.basename(chosen[-1])} ({'binary' if binary_dirs else 'INTEGER, old version'})")
            d = chosen[-1]
            cfg = json.load(open(os.path.join(d, "config", "experiment_config.json")))
            cfgs[(loc, prof)] = cfg
            ds = os.path.basename(os.path.dirname(d))
            m = re.match(r"dataset_hex(\d+)_impacts(\d+)", ds)
            summ = pd.read_csv(os.path.join(d, "summary", "runs_summary.csv"))
            seeds_cfg = [int(s) for s in cfg["seeds"]]
            if sorted(summ["seed"].astype(int)) != sorted(seeds_cfg):
                log.append(f"[{algo_name}] {loc}/{prof}: seeds in summary differ from config")
            for _, r in summ.iterrows():
                rows.append({
                    "algorithm": algo_name, "location": loc, "profile": prof,
                    "seed": int(r["seed"]),
                    "best": float(r["best_objective_value"]),
                    "initial": float(r["initial_objective_value"]),
                    "evaluations": float(r["evaluations"]),
                    "runtime_s": float(r["runtime_seconds"]),
                    "stopping_reason": r["stopping_reason"],
                    "baseline": float(cfg["baseline_iqc_total"]),
                    "n_hex": int(m.group(1)), "n_impacts": int(m.group(2)),
                })
            for sd in sorted(glob.glob(os.path.join(d, "seed_runs", "seed_*"))):
                t = pd.read_csv(os.path.join(sd, "trajectory.csv"),
                                usecols=["seed", "iter", "eval_count", "elapsed_s", "best_sum_iqc"])
                t.insert(0, "profile", prof); t.insert(0, "location", loc); t.insert(0, "algorithm", algo_name)
                traj.append(t)
                # consistency: last best in trajectory == best in summary
                b = summ.loc[summ["seed"] == int(t["seed"].iloc[0]), "best_objective_value"].iloc[0]
                if abs(t["best_sum_iqc"].max() - b) > 1e-9:
                    log.append(f"[{algo_name}] {loc}/{prof} seed {t['seed'].iloc[0]}: trajectory max "
                               f"{t['best_sum_iqc'].max()} != summary {b}")
    return rows, traj, cfgs


def read_brkga_pso(root, algo_key, algo_name, seeds, baselines, log):
    rows, eff = [], []
    for loc in LOCATIONS:
        f = os.path.join(root, algo_key, loc, "objectives.txt")
        lines = [l for l in open(f, encoding="utf-8").read().splitlines() if l.strip()]
        rc = pd.read_csv(os.path.join(root, algo_key, loc, "runs.csv"), skipinitialspace=True)
        rc.columns = [c.strip() for c in rc.columns]
        seen = {}
        for i, line in enumerate(lines):
            prof, vals = line.split(":", 1)
            prof = prof.strip()
            vals = ast.literal_eval(vals.strip())
            if prof in seen:
                same = np.allclose(seen[prof], vals)
                where = "identical to the first block (re-execution reproduced the values)"
                if not same:
                    where = "different values"
                    for other in LOCATIONS:
                        if other == loc:
                            continue
                        fo = os.path.join(root, algo_key, other, "objectives.txt")
                        for lo in open(fo, encoding="utf-8").read().splitlines():
                            if lo.startswith(prof + ":") and np.allclose(ast.literal_eval(lo.split(":", 1)[1].strip()), vals):
                                where = f"equal to {other}/{prof} (appended here by the hard-coded output path)"
                log.append(f"[{algo_name}] {loc}/objectives.txt block #{i} ({prof}): {where}; ignored")
                continue
            seen[prof] = vals
            if len(vals) != len(seeds):
                log.append(f"[{algo_name}] {loc}/{prof}: {len(vals)} values for {len(seeds)} seeds")
            mean_time = float(rc.iloc[i]["mean time (s)"])
            for s, v in zip(seeds, vals):
                rows.append({"algorithm": algo_name, "location": loc, "profile": prof, "seed": int(s),
                             "best": float(v), "initial": np.nan, "evaluations": np.nan,
                             "runtime_s": np.nan, "stopping_reason": "max_generations",
                             "baseline": baselines[(loc, prof)][0],
                             "n_hex": baselines[(loc, prof)][1], "n_impacts": baselines[(loc, prof)][2]})
            eff.append({"algorithm": algo_name, "location": loc, "profile": prof,
                        "mean_runtime_s_reported": mean_time})
        for prof in PROFILES:
            if prof not in seen:
                log.append(f"[{algo_name}] {loc}/{prof}: missing")
    return rows, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = os.path.join(a.out, "data")
    os.makedirs(out, exist_ok=True)
    log = []

    r_ils, t_ils, c_ils = read_ils_grasp(a.results, "ils", "ILS", log)
    r_gr, t_gr, c_gr = read_ils_grasp(a.results, "grasp", "GRASP", log)
    seeds = [int(s) for s in next(iter(c_ils.values()))["seeds"]]
    base = {k: (v["baseline_iqc_total"], None, None) for k, v in c_ils.items()}
    dfi = pd.DataFrame(r_ils)
    base = {(l, p): (g["baseline"].iloc[0], int(g["n_hex"].iloc[0]), int(g["n_impacts"].iloc[0]))
            for (l, p), g in dfi.groupby(["location", "profile"])}
    for k in c_gr:
        if abs(c_gr[k]["baseline_iqc_total"] - c_ils[k]["baseline_iqc_total"]) > 1e-9:
            log.append(f"baseline differs between ILS and GRASP for {k}")
        if [int(s) for s in c_gr[k]["seeds"]] != seeds:
            log.append(f"GRASP seeds differ for {k}")
    r_br, e_br = read_brkga_pso(a.results, "brkga", "BRKGA", seeds, base, log)
    r_ps, e_ps = read_brkga_pso(a.results, "pso", "PSO", seeds, base, log)

    runs = pd.DataFrame(r_ils + r_gr + r_br + r_ps)
    runs["instance"] = runs["location"] + "/" + runs["profile"]
    runs["gain_pct"] = 100 * (runs["best"] - runs["baseline"]) / runs["baseline"]
    # Best known solution per instance over every run of every algorithm
    # (Beiranvand et al. 2017, Sec. 4.3 "No known solution available";
    #  PCO212 slide 09: BKS, RPD).
    bks = runs.groupby("instance")["best"].max()
    runs["bks"] = runs["instance"].map(bks)
    runs["rpd"] = 100 * (runs["bks"] - runs["best"]) / runs["bks"]        # >= 0, lower is better
    runs["rpd_gain"] = 100 * (runs["bks"] - runs["best"]) / (runs["bks"] - runs["baseline"])
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)

    inst = (runs.groupby(["instance", "location", "profile"])
            .agg(baseline=("baseline", "first"), n_hex=("n_hex", "first"),
                 n_impacts=("n_impacts", "first"), bks=("bks", "first")).reset_index())
    who = runs.loc[runs.groupby("instance")["best"].idxmax(), ["instance", "algorithm", "seed"]]
    inst = inst.merge(who.rename(columns={"algorithm": "bks_algorithm", "seed": "bks_seed"}), on="instance")
    inst.to_csv(os.path.join(out, "instances.csv"), index=False)

    # Effort (Beiranvand et al. 2017, Sec. 4.1; Eiben & Smit 2011, Sec. 4.1)
    eff = []
    for name in ["ILS", "GRASP"]:
        g = runs[runs.algorithm == name].groupby(["location", "profile"])
        for (l, p), x in g:
            eff.append({"algorithm": name, "location": l, "profile": p,
                        "evaluations_mean": x["evaluations"].mean(),
                        "evaluations_min": x["evaluations"].min(),
                        "evaluations_max": x["evaluations"].max(),
                        "evaluations_source": "measured (run_metrics)",
                        "mean_runtime_s_reported": x["runtime_s"].mean()})
    for e in e_br:
        e.update({"evaluations_mean": 100 + 600 * (100 - 15), "evaluations_min": 51100,
                  "evaluations_max": 51100,
                  "evaluations_source": "derived from code: 100 initial + 600 generations x 85 offspring"})
        eff.append(e)
    for e in e_ps:
        e.update({"evaluations_mean": np.nan, "evaluations_min": 50 + 600 * 50, "evaluations_max": np.nan,
                  "evaluations_source": "lower bound from code: 50 initial + 600 x 50 moves; "
                                        "k-tournament evaluations not logged"})
        eff.append(e)
    pd.DataFrame(eff).to_csv(os.path.join(out, "effort.csv"), index=False)

    # Stored "global best" allocation of each algorithm and instance: structure
    # and feasibility with respect to sum(DeltaX) = B = 100 (paper, Eq. budget).
    alloc = []
    for key, name in [("ils", "ILS"), ("grasp", "GRASP"), ("brkga", "BRKGA"), ("pso", "PSO")]:
        for loc in LOCATIONS:
            for prof in PROFILES:
                d = _run_dir(a.results, key, loc, prof)[-1]
                f = os.path.join(d, "summary", "global_best_allocation.csv")
                js = [os.path.join(d, "summary", x) for x in ("method_result.json", "matrix_result.json")]
                j = json.load(open([x for x in js if os.path.isfile(x)][0]))
                v = j.get("global_best_objective_value", j.get("returned_objective_value"))
                al = pd.read_csv(f)
                q = al.groupby("h3_id")["quantity"].sum()
                best_run = runs.loc[(runs.algorithm == name) & (runs.location == loc) & (runs.profile == prof), "best"].max()
                alloc.append({"algorithm": name, "location": loc, "profile": prof, "stored_objective": v,
                              "best_run_objective": best_run, "stored_is_best_run": bool(abs(v - best_run) < 1e-6),
                              "total_pois": int(al["quantity"].sum()), "max_per_cell": int(al["quantity"].max()),
                              "hexagons_used": int((q > 0).sum()), "q_max": int(q.max()),
                              "hhi": float(((q / q.sum()) ** 2).sum())})
    alloc = pd.DataFrame(alloc)
    alloc.to_csv(os.path.join(out, "best_allocations.csv"), index=False)
    for _, r in alloc.iterrows():
        if r["total_pois"] != 100:
            log.append(f"[{r['algorithm']}] {r['location']}/{r['profile']}: stored best allocation has "
                       f"{r['total_pois']} POIs (budget B = 100)")
        if not r["stored_is_best_run"]:
            log.append(f"[{r['algorithm']}] {r['location']}/{r['profile']}: stored allocation "
                       f"({r['stored_objective']:.4f}) is not the best run ({r['best_run_objective']:.4f})")

    traj = pd.concat(t_ils + t_gr, ignore_index=True)
    traj.to_csv(os.path.join(out, "trajectories.csv.gz"), index=False, compression="gzip")

    counts = runs.groupby(["algorithm"]).size().to_dict()
    log.insert(0, f"runs per algorithm: {counts}")
    log.insert(1, f"instances: {runs['instance'].nunique()}, seeds: {len(seeds)}")
    open(os.path.join(out, "data_quality.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
