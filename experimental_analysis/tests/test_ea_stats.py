"""Validate ea_stats against the worked case study of Derrac et al. (2011).

Run:  python -m pytest tests/test_ea_stats.py   (or)   python tests/test_ea_stats.py
The reference numbers are copied from Tables 11, 13, 14, 15, 16, 17, 19, 20.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ea_stats as es  # noqa: E402

DF = pd.read_csv(os.path.join(HERE, "derrac_table1.csv")).set_index("function")
ALGS = list(DF.columns)
X = DF.values


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


# Table 7 (Example 4): rows P1..P4, columns A..D
TOY = np.array([[2.711, 3.147, 2.515, 2.612],
                [7.832, 9.828, 7.832, 7.921],
                [0.012, 0.532, 0.122, 0.005],
                [3.431, 4.111, 3.401, 3.401]])
# Average ranks printed in Table 11 (the printed Table 1 is rounded, so ranks
# recomputed from it differ slightly; the post-hoc tables depend only on these).
T11_FRIEDMAN = [7, 4.84, 6.28, 5.5, 4.64, 5.4, 4, 3.5, 3.84]


def test_example4_rankings():
    # Table 8 prints 1.250 for C, but its own column (1, 1.5, 3, 1.5) averages
    # to 1.75 and the four averages must sum to k(k+1)/2 = 10: a typo in the paper.
    np.testing.assert_allclose(es.friedman_test(TOY)["avg_ranks"], [2.375, 4, 1.75, 1.875])
    np.testing.assert_allclose(es.friedman_aligned_test(TOY)["avg_ranks"], [7.625, 14.5, 5.5, 6.375])
    np.testing.assert_allclose(es.quade_test(TOY)["avg_ranks"], [2.3, 4, 1.55, 2.15], atol=1e-9)


def test_table11_friedman_statistic():
    fr = es.friedman_from_ranks(T11_FRIEDMAN, 25)
    close(fr["statistic"], 35.99733, 1e-4)
    close(fr["p_value"], 0.000018, 1e-6)
    close(fr["iman_davenport"], 5.267817, 1e-4)


def test_tables13_15_z_values():
    # Table 12 ranks for IPOP-CMA-ES, CHC, SS-BLX, SaDE (n = 25)
    n = 25
    p, z = es.unadjusted_p([2.48, 3.12, 2.44, 1.96], 1, 3, "friedman", n)
    close(z, 3.176791, 1e-5); close(p, 0.001489, 1e-6)
    p, z = es.unadjusted_p([51.96, 65.92, 48.52, 35.6], 1, 3, "aligned", n)
    close(z, 3.694997, 1e-5); close(p, 0.000220, 1e-6)
    p, z = es.unadjusted_p([2.3785, 3.4185, 2.48, 1.7231], 1, 3, "quade", n)
    close(z, 3.315129, 1e-4); close(p, 0.000916, 1e-6)


def test_table16_friedman_apvs():
    fr = T11_FRIEDMAN
    ctrl = ALGS.index("DE-Exp")
    others = [j for j in range(len(ALGS)) if j != ctrl]
    p = [es.unadjusted_p(fr, j, ctrl, "friedman", X.shape[0])[0] for j in others]
    adj = es.adjust_1xN(p)
    order = np.argsort(p)
    ref = {  # PSO, CHC, SSGA, SS-Arit, IPOP, SS-BLX, DE-Bin, SaDE (sorted by p)
        "unadjusted": [0.000006, 0.000332, 0.009823, 0.014171, 0.083642, 0.141093, 0.518605, 0.660706],
        "bonferroni": [0.000050, 0.002656, 0.078586, 0.113371, 0.669139, 1.0, 1.0, 1.0],
        "holm": [0.000050, 0.002324, 0.058940, 0.070857, 0.334569, 0.423278, 1.0, 1.0],
        "hochberg": [0.000050, 0.002324, 0.058940, 0.070857, 0.334569, 0.423278, 0.660706, 0.660706],
        "hommel": [0.000050, 0.002324, 0.049116, 0.070857, 0.282186, 0.423278, 0.660706, 0.660706],
        "holland": [0.000050, 0.002322, 0.057511, 0.068877, 0.294885, 0.366366, 0.768259, 0.768259],
        "finner": [0.000050, 0.001327, 0.025981, 0.028142, 0.130431, 0.183552, 0.566345, 0.660706],
        "li": [0.000018, 0.000978, 0.028137, 0.040093, 0.197766, 0.293707, 0.604506, 0.660706],
    }
    for key, vals in ref.items():
        np.testing.assert_allclose(adj[key][order], vals, atol=2e-5, err_msg=key)


def test_contrast_estimation_properties():
    # The printed Table 1 does not reproduce Table 11 ranks exactly (see above),
    # so Table 19 cannot be matched number by number from it. The procedure of
    # Sec. 4.4 is checked on its defining case instead: when the differences
    # between algorithms are the same on every problem (its stated assumption),
    # the estimator must return exactly those differences.
    rng = np.random.default_rng(0)
    shift = np.array([0.0, 1.5, -2.0, 4.0])
    data = rng.normal(size=(15, 1)) * 10 + shift[None, :]
    E = es.contrast_estimation(data)
    np.testing.assert_allclose(E, shift[:, None] - shift[None, :], atol=1e-12)
    E = es.contrast_estimation(X)
    np.testing.assert_allclose(E, -E.T, atol=1e-12)
    # Same qualitative reading as Table 19: SaDE has negative estimators
    # against every other algorithm, CHC positive ones.
    assert np.all(np.delete(E[ALGS.index("SaDE")], ALGS.index("SaDE")) < 0)
    assert np.all(np.delete(E[ALGS.index("CHC")], ALGS.index("CHC")) > 0)


def test_table20_NxN():
    fr = T11_FRIEDMAN
    k, n = len(ALGS), X.shape[0]
    pp = {(i, j): es.unadjusted_p(fr, i, j, "friedman", n)[0] for i in range(k) for j in range(i + 1, k)}
    adj = es.adjust_NxN(pp, k)
    def key(a, b):
        i, j = ALGS.index(a), ALGS.index(b)
        return (min(i, j), max(i, j))
    rows = [("PSO", "DE-Exp", 0.000224, 0.000224, 0.000224, 0.000224),
            ("PSO", "SaDE", 0.001624, 0.001579, 0.001263, 0.001263),
            ("PSO", "DE-Bin", 0.00387, 0.003655, 0.00301, 0.002365),
            ("CHC", "SaDE", 0.058772, 0.052242, 0.045712, 0.034284),
            ("SSGA", "DE-Exp", 0.353638, 0.275052, 0.275052, 0.216112),
            ("SS-Arit", "SaDE", 1.0, 1.0, 0.968322, 0.621874)]
    for a, b, nem, holm, sha, berg in rows:
        kk = key(a, b)
        close(adj["nemenyi"][kk], nem, 2e-4)
        close(adj["holm"][kk], holm, 2e-4)
        close(adj["shaffer"][kk], sha, 2e-4)
        close(adj["bergmann"][kk], berg, 2e-4)


def test_wilcoxon_rank_sums():
    # Table 6 (R+ = 261, R- = 64 for SaDE vs PSO) was computed on unrounded
    # data; on the printed (rounded) Table 1 several F8 entries tie, so only the
    # identity R+ + R- = n(n+1)/2 (Example 2) and the direction are checked.
    w = es.wilcoxon_signed_rank(DF["SaDE"].values, DF["PSO"].values)
    assert w["R_plus"] + w["R_minus"] == 25 * 26 / 2
    assert w["R_plus"] > w["R_minus"] and w["p_value"] < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
