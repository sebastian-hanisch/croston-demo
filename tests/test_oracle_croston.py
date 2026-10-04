"""Unabhängige Orakel für die Croston-Demo (ergänzt die festen Fälle aus test_methods.py).

1. Croston, SBA, TSB und einfache Glättung auf zufälligen Instanzen gegen eine andere Rechenweise: ereignisweise (nur Bedarfstage zählen, Abstand aus den Tagesnummern) bzw. geschlossene Formen
   der exponentiellen Gewichte - inklusive Artikel ohne Bedarf im Anfangsfenster und mit Bedarf an jedem Tag.
2. `evaluate` (Wiederbeschaffungszeit-Summen, Skala, Fehler, MAE/RMSSE/Verzerrung) gegen Brute-Force-Schleifen; `choose_alpha` gegen eine Schleife.
3. Orakel des Portfolios: der Erwartungswert der Menge max(1, rint(X)) (X log-normal) gegen Monte Carlo und eine direkte Summe - die Rundung und das Mindestmaß 1 heben ihn über den Parameter m
   (bei Mittel 1 und CV 1,5 um 40 %); die Orakel-Rate muss das berücksichtigen (früher: p * m, bis 29 % zu niedrig).
"""

import math

import numpy as np
import pytest

import cr_constants as C
import cr_evaluation as E
import cr_methods as M
import cr_scenario as S


def _reference(y, alpha, start, kind, beta=None):
    T = len(y)
    first = y[:start]
    occ0 = np.nonzero(first > 0)[0]
    z0 = first[occ0].mean() if len(occ0) else 1.0
    p0 = start / len(occ0) if len(occ0) else float(start)
    last0 = occ0[-1] if len(occ0) else -1
    rate = np.empty(T + 1)
    if kind in ("croston", "sba"):
        z, p, last, cur = z0, p0, last0, z0 / p0
        rate[:start + 1] = cur
        for t in range(start, T):
            if y[t] > 0:
                z, p, last = z + alpha * (y[t] - z), p + alpha * ((t - last) - p), t           # Abstand = Tage seit dem letzten Bedarf, aus den Tagesnummern
                cur = z / p
            rate[t + 1] = cur
        return rate * (1.0 - alpha / 2.0) if kind == "sba" else rate
    if kind == "tsb":
        b = alpha if beta is None else beta
        d0 = 1.0 / p0
        ind = (y > 0).astype(float)
        z = z0
        rate[:start + 1] = d0 * z0
        for t in range(start, T):
            if y[t] > 0:
                z += alpha * (y[t] - z)
            d = (1 - b) ** (t - start + 1) * d0 + b * sum((1 - b) ** (t - s) * ind[s] for s in range(start, t + 1))      # geschlossene Form der Wahrscheinlichkeits-Glättung
            rate[t + 1] = d * z
        return rate
    l0 = first.mean()                                                                                      # ses
    rate[:start + 1] = l0
    for t in range(start, T):
        rate[t + 1] = (1 - alpha) ** (t - start + 1) * l0 + alpha * sum((1 - alpha) ** (t - s) * y[s] for s in range(start, t + 1))
    return rate


def test_methods_equal_an_eventwise_and_closed_form_reference_on_random_instances():
    rng = np.random.default_rng(77)
    for it in range(120):
        n, T, start = int(rng.integers(1, 5)), int(rng.integers(30, 100)), int(rng.integers(3, 25))
        pr = float(rng.choice([0.0, 0.02, 0.1, 0.4, 1.0]))
        y = np.where(rng.random((n, T)) < pr, np.floor(rng.lognormal(1, 0.8, (n, T))) + 1, 0.0)
        if it % 7 == 0:
            y[0, :start] = 0.0                                                                             # kein Bedarf im Anfangsfenster
        alpha = float(rng.uniform(0.01, 0.6))
        beta = float(rng.uniform(0.01, 0.6)) if it % 3 == 0 else None
        cro, sba = M.croston_rates(y, alpha, start=start)
        got = {"croston": cro, "sba": sba, "tsb": M.tsb_rates(y, alpha, beta=beta, start=start), "ses": M.ses_rates(y, alpha, start=start)}
        for i in range(n):
            for kind, rates in got.items():
                ref = _reference(y[i], alpha, start, kind, beta if kind == "tsb" else None)
                assert np.allclose(rates[i], ref, rtol=1e-9, atol=1e-9), (it, i, kind)


def test_evaluate_and_choose_alpha_equal_brute_force():
    rng = np.random.default_rng(78)
    for it in range(12):
        port = S.generate(n_items=int(rng.integers(3, 8)), seed=int(rng.integers(0, 1000)), p_mean=float(rng.uniform(0.03, 0.4)))
        h, alpha = int(rng.integers(1, 29)), float(rng.choice([0.05, 0.1, 0.3]))
        rates, _ = M.all_rates(port.y, alpha)
        org, actual, errors, scale, summary = E.evaluate(port, rates, h)
        y = port.y
        n, T = y.shape
        origins = list(range(C.FIRST_TEST, T - h + 1))
        assert list(org) == origins
        act = np.array([[y[i, t:t + h].sum() for t in origins] for i in range(n)])
        sc = np.array([max(np.sqrt(np.mean([(y[i, t:t + h].sum() - y[i, t - h:t].sum()) ** 2 for t in range(h, C.FIRST_TEST - h + 1)])), C.SCALE_FLOOR) for i in range(n)])
        assert np.array_equal(act, actual) and np.allclose(sc, scale)
        for m in ("ses", "croston", "tsb", "zero", "mean"):
            err = np.array([[h * rates[m][i, t] - act[i, j] for j, t in enumerate(origins)] for i in range(n)])
            s = summary[m]
            assert np.allclose(err, errors[m])
            assert s["mae"] == pytest.approx(np.abs(err).mean()) and s["me_pct"] == pytest.approx(100 * err.mean() / act.mean())
            assert s["rmsse"] == pytest.approx(np.sqrt(np.mean(np.mean(err ** 2, axis=1) / sc ** 2)))
    y = S.generate(n_items=6, seed=11).y
    grid = M.grid_rates(y)
    for meth in ("ses", "sba", "tsb"):
        _, alphas = M.choose_alpha(grid, y, meth)
        for i in range(6):
            mse = {a: np.mean((grid[a][meth][i, C.INIT_DAYS:C.FIRST_TEST] - y[i, C.INIT_DAYS:C.FIRST_TEST]) ** 2) for a in sorted(grid)}
            assert alphas[i] == min(sorted(mse), key=lambda a: (mse[a], a))


def _direct_mean_size(m, cv):
    """E[max(1, rint(X))] als direkte Summe k * P(Bin k) mit der log-normalen Verteilungsfunktion (math.erf), ohne Näherung des Rests."""
    sigma = math.sqrt(math.log(1 + cv ** 2))
    mu = math.log(m) - 0.5 * sigma ** 2

    def cdf(x):
        return 0.5 * (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2)))) if x > 0 else 0.0

    top = int(m * math.exp(10 * sigma)) + 10
    return cdf(1.5) + sum(k * (cdf(k + 0.5) - cdf(k - 0.5)) for k in range(2, top))


@pytest.mark.parametrize("m,cv", [(1.0, 0.1), (1.0, 0.6), (1.0, 1.5), (2.3, 1.0), (5.0, 0.6), (12.0, 1.5), (40.0, 0.6)])
def test_mean_size_equals_a_direct_sum(m, cv):
    assert S.mean_size(np.array([m]), cv)[0] == pytest.approx(_direct_mean_size(m, cv), rel=1e-6)


def test_oracle_rate_is_the_true_expected_demand_even_for_small_lumpy_sizes():
    """Realisierte Gesamtnachfrage gegen die Orakel-Rate: bei Mittel 1 und CV 1,5 liegt p * m um 29 % unter der Wirklichkeit, das Orakel muss darauf kommen."""
    port = S.generate(n_items=400, p_mean=0.3, size_mean=1.0, cv=1.5, drift=0.0, seed=1)
    assert port.y.sum() / port.mu.sum() == pytest.approx(1.0, abs=0.02)
    port = S.generate(n_items=400, p_mean=0.3, size_mean=2.0, cv=1.0, drift=0.0, seed=1)
    assert port.y.sum() / port.mu.sum() == pytest.approx(1.0, abs=0.02)
