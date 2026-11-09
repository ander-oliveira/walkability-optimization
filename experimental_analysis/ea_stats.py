"""
Nonparametric procedures for multi-problem comparison of algorithms.

Every procedure here follows, formula by formula, the tutorial

    Derrac, J., Garcia, S., Molina, D., Herrera, F. (2011).
    A practical tutorial on the use of nonparametric statistical tests as a
    methodology for comparing evolutionary and swarm intelligence algorithms.
    Swarm and Evolutionary Computation 1, 3-18.

Section numbers in the docstrings refer to that paper. The module is
validated against the paper's own case study (Tables 1, 11, 13-17, 19, 20)
in ``tests/test_ea_stats.py``.

Convention: ``data`` is an (n problems x k algorithms) array where a LOWER
value is BETTER (rank 1 = best), as in the paper. For a maximisation
problem pass ``-data`` or use a minimisation measure (e.g. RPD to the
best-known solution).
"""
from __future__ import annotations

from itertools import combinations
from math import comb, sqrt
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# Rankings (Sections 4.2)
# --------------------------------------------------------------------------
def friedman_ranks(data: np.ndarray) -> np.ndarray:
    """Rank algorithms inside each problem, 1 = best, average ranks on ties."""
    return np.apply_along_axis(stats.rankdata, 1, np.asarray(data, float))


def friedman_test(data: np.ndarray) -> Dict[str, object]:
    """Friedman test and Iman-Davenport extension (Sec. 4.2, Eqs. 2-3)."""
    data = np.asarray(data, float)
    n, k = data.shape
    return friedman_from_ranks(friedman_ranks(data).mean(axis=0), n)


def friedman_from_ranks(R: Sequence[float], n: int) -> Dict[str, object]:
    """Friedman / Iman-Davenport statistics from average ranks (Eqs. 2-3)."""
    R = np.asarray(R, float)
    k = len(R)
    chi2 = 12.0 * n / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = stats.chi2.sf(chi2, k - 1)
    f_id = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p_id = stats.f.sf(f_id, k - 1, (k - 1) * (n - 1))
    return {"avg_ranks": R, "statistic": chi2, "p_value": p_chi2,
            "iman_davenport": f_id, "p_iman_davenport": p_id, "n": n, "k": k}


def aligned_ranks(data: np.ndarray) -> np.ndarray:
    """Aligned observations ranked from 1 to k*n (Sec. 4.2)."""
    data = np.asarray(data, float)
    aligned = data - data.mean(axis=1, keepdims=True)
    return stats.rankdata(aligned.ravel()).reshape(data.shape)


def friedman_aligned_test(data: np.ndarray) -> Dict[str, object]:
    """Friedman Aligned Ranks test (Sec. 4.2, Eq. 4)."""
    data = np.asarray(data, float)
    n, k = data.shape
    ar = aligned_ranks(data)
    Rj = ar.sum(axis=0)          # rank totals per algorithm
    Ri = ar.sum(axis=1)          # rank totals per problem
    kn = k * n
    num = (k - 1) * (np.sum(Rj ** 2) - (k * n ** 2 / 4.0) * (kn + 1) ** 2)
    den = (kn * (kn + 1) * (2 * kn + 1) / 6.0) - np.sum(Ri ** 2) / k
    stat = num / den
    return {"avg_ranks": ar.mean(axis=0), "statistic": stat,
            "p_value": stats.chi2.sf(stat, k - 1), "n": n, "k": k}


def quade_test(data: np.ndarray) -> Dict[str, object]:
    """Quade test (Sec. 4.2, Eqs. 5-11)."""
    data = np.asarray(data, float)
    n, k = data.shape
    r = friedman_ranks(data)
    ranges = data.max(axis=1) - data.min(axis=1)
    Q = stats.rankdata(ranges)
    S = Q[:, None] * (r - (k + 1) / 2.0)
    Sj = S.sum(axis=0)
    W = (Q[:, None] * r).sum(axis=0)
    T = W / (n * (n + 1) / 2.0)
    A = n * (n + 1) * (2 * n + 1) * k * (k + 1) * (k - 1) / 72.0
    B = np.sum(Sj ** 2) / n
    if np.isclose(A, B):
        stat, p = np.inf, (1.0 / np.math.factorial(k)) ** (n - 1)
    else:
        stat = (n - 1) * B / (A - B)
        p = stats.f.sf(stat, k - 1, (k - 1) * (n - 1))
    return {"avg_ranks": T, "statistic": stat, "p_value": p, "n": n, "k": k}


# --------------------------------------------------------------------------
# z statistics for post-hoc comparisons (Sec. 4.3, Eqs. 12-14)
# --------------------------------------------------------------------------
def _se(test: str, n: int, k: int) -> float:
    if test == "friedman":
        return sqrt(k * (k + 1) / (6.0 * n))
    if test == "aligned":
        # Eq. 13 is printed as sqrt(k(n+1)/6); the values in Table 14 are only
        # reproduced with sqrt(k(kn+1)/6) (aligned ranks run from 1 to kn).
        return sqrt(k * (k * n + 1) / 6.0)
    if test == "quade":
        return sqrt(k * (k + 1) * (2 * n + 1) * (k - 1) / (18.0 * n * (n + 1)))
    raise ValueError(test)


def unadjusted_p(avg_ranks: Sequence[float], i: int, j: int, test: str, n: int) -> float:
    k = len(avg_ranks)
    z = (avg_ranks[i] - avg_ranks[j]) / _se(test, n, k)
    return float(2 * stats.norm.sf(abs(z))), float(z)


# --------------------------------------------------------------------------
# Adjusted p-values, 1 x N (Sec. 4.3). Input p-values in any order.
# --------------------------------------------------------------------------
def _order(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="mergesort")
    return p, o


def adjust_1xN(p: Sequence[float]) -> Dict[str, np.ndarray]:
    """Bonferroni-Dunn, Holm, Hochberg, Hommel, Holland, Finner and Li APVs."""
    p, o = _order(p)
    m = len(p)
    ps = p[o]
    out = {}
    out["bonferroni"] = np.minimum(m * ps, 1.0)
    out["holm"] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps), 1.0)
    hoch = np.minimum.accumulate(((m - np.arange(m)) * ps)[::-1])[::-1]
    out["hochberg"] = np.minimum(hoch, 1.0)
    out["holland"] = np.minimum(np.maximum.accumulate(1 - (1 - ps) ** (m - np.arange(m))), 1.0)
    out["finner"] = np.minimum(np.maximum.accumulate(1 - (1 - ps) ** (m / (np.arange(m) + 1.0))), 1.0)
    out["li"] = ps / (ps + 1 - ps[-1])
    out["hommel"] = _hommel(ps)
    res = {}
    for key, v in out.items():
        back = np.empty(m)
        back[o] = v
        res[key] = back
    res["unadjusted"] = p
    return res


def _hommel(p: np.ndarray) -> np.ndarray:
    """Hommel adjusted p-values (Fig. 1 of the tutorial); p sorted ascending."""
    n = len(p)
    if n == 1:
        return np.minimum(p, 1.0)
    i = np.arange(1, n + 1)
    q = np.full(n, np.min(n * p / i))
    pa = q.copy()
    for m in range(n - 1, 1, -1):
        i1 = np.arange(0, n - m + 1)
        i2 = np.arange(n - m + 1, n)
        q1 = np.min(m * p[i2] / np.arange(2, m + 1))
        q[i1] = np.minimum(m * p[i1], q1)
        q[i2] = q[n - m]
        pa = np.maximum(pa, q)
    return np.minimum(np.maximum(pa, p), 1.0)


# --------------------------------------------------------------------------
# N x N (Sec. 5): Nemenyi, Holm, Shaffer static, Bergmann-Hommel
# --------------------------------------------------------------------------
def _shaffer_S(k: int, memo={}) -> set:
    """Set of possible numbers of true hypotheses with k algorithms (Eq. 18)."""
    if k in memo:
        return memo[k]
    if k <= 1:
        memo[k] = {0}
        return memo[k]
    s = set()
    for j in range(1, k + 1):
        for x in _shaffer_S(k - j):
            s.add(comb(j, 2) + x)
    memo[k] = s
    return s


def _set_partitions(items: List[int]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for idx in range(len(part)):
            yield part[:idx] + [[first] + part[idx]] + part[idx + 1:]
        yield [[first]] + part


def adjust_NxN(p_pairs: Dict[tuple, float], k: int) -> Dict[str, Dict[tuple, float]]:
    """APVs for all pairwise hypotheses. Keys are (i, j) with i < j."""
    keys = sorted(p_pairs, key=lambda t: p_pairs[t])
    ps = np.array([p_pairs[t] for t in keys])
    m = len(ps)
    nem = np.minimum(m * ps, 1.0)
    holm = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps), 1.0)
    S = sorted(_shaffer_S(k))
    t = []
    for i in range(1, m + 1):              # i-1 hypotheses already false
        t.append(max(x for x in S if x <= m - i + 1))
    shaffer = np.minimum(np.maximum.accumulate(np.array(t) * ps), 1.0)
    # Bergmann-Hommel: exhaustive sets = sets of within-block pairs of a partition
    pos = {key: idx for idx, key in enumerate(keys)}
    exhaustive = []
    for part in _set_partitions(list(range(k))):
        I = [pos[(min(a, b), max(a, b))] for blk in part for a, b in combinations(blk, 2)]
        if I:
            exhaustive.append(I)
    berg = np.zeros(m)
    for idx in range(m):
        v = 0.0
        for I in exhaustive:
            if idx in I:
                v = max(v, len(I) * min(ps[I]))
        berg[idx] = min(v, 1.0)
    berg = np.maximum.accumulate(berg)     # APVs kept monotone, as in Table 20
    out = {}
    for name, arr in [("unadjusted", ps), ("nemenyi", nem), ("holm", holm),
                      ("shaffer", shaffer), ("bergmann", berg)]:
        out[name] = {key: float(arr[pos[key]]) for key in keys}
    return out


# --------------------------------------------------------------------------
# Contrast estimation based on medians (Sec. 4.4, Eqs. 16-17)
# --------------------------------------------------------------------------
def contrast_estimation(data: np.ndarray) -> np.ndarray:
    """Matrix E with E[u, v] = m_u - m_v, estimator of M_u - M_v."""
    data = np.asarray(data, float)
    n, k = data.shape
    Z = np.zeros((k, k))
    for u in range(k):
        for v in range(k):
            if u != v:
                Z[u, v] = np.median(data[:, u] - data[:, v])
    m = Z.mean(axis=1)
    return m[:, None] - m[None, :]


# --------------------------------------------------------------------------
# Pairwise (Sec. 3)
# --------------------------------------------------------------------------
def sign_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Wins of a over b (lower is better); ties split evenly (Sec. 3.1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wins = int(np.sum(a < b)); losses = int(np.sum(a > b)); ties = int(np.sum(a == b))
    n = len(a)
    w = wins + ties // 2
    p = stats.binomtest(w, n - (ties % 2), 0.5).pvalue
    return {"wins": wins, "ties": ties, "losses": losses, "p_value": float(p)}


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-ranks test with R+ / R- as in Sec. 3.2 (Eq. 1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a                       # positive -> a better (lower)
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum() + 0.5 * r[d == 0].sum()
    rm = r[d < 0].sum() + 0.5 * r[d == 0].sum()
    if np.all(d == 0):
        p = 1.0
    else:
        p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
    return {"R_plus": float(rp), "R_minus": float(rm), "p_value": float(p)}
