"""Die Verfahren: Anfangswerte und jeder Schritt von Hand, unabhängige Schleife je Artikel, statsmodels für die einfache Glättung, Eigenschaften (Croston bleibt zwischen Bedarfen stehen, TSB fällt), Wahl von alpha."""

import numpy as np
import pytest

import cr_constants as C
import cr_methods as M
import cr_scenario as S


def _init(z, p, q, d=None, l=0.0):
    return {"z": np.array([z], dtype=float), "p": np.array([p], dtype=float), "q": np.array([q], dtype=float), "d": np.array([1.0 / p if d is None else d]), "l": np.array([l], dtype=float)}


def test_init_state_by_hand():
    y = np.array([[0, 0, 3, 0, 5, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 2]], dtype=float)
    st = M.init_state(y, 6)
    assert st["z"].tolist() == pytest.approx([4.0, 1.0, 3.0]) and st["p"].tolist() == pytest.approx([3.0, 6.0, 3.0]) and st["q"].tolist() == [2.0, 7.0, 1.0]
    assert st["d"].tolist() == pytest.approx([1 / 3, 1 / 6, 1 / 3]) and st["l"].tolist() == pytest.approx([8 / 6, 0.0, 1.0])


def test_ses_by_hand():
    y = np.array([[0.0, 4.0, 0.0, 2.0]])
    out = M.ses_rates(y, 0.5, start=0, init=_init(1, 1, 1, l=1.0))
    assert out[0].tolist() == pytest.approx([1.0, 0.5, 2.25, 1.125, 1.5625])


def test_croston_and_sba_by_hand():
    y = np.array([[0.0, 3.0, 0.0, 0.0, 1.0]])
    cro, sba = M.croston_rates(y, 0.5, start=0, init=_init(2, 2, 1))
    # t0: kein Bedarf, q = 2. t1: Bedarf 3, Abstand q = 2: z = 2,5, p = 2, q = 1. t2, t3: q = 2, 3. t4: Bedarf 1, Abstand 3: z = 1,75, p = 2,5.
    assert cro[0].tolist() == pytest.approx([1.0, 1.0, 1.25, 1.25, 1.25, 0.7])
    assert sba[0].tolist() == pytest.approx([0.75 * v for v in [1.0, 1.0, 1.25, 1.25, 1.25, 0.7]])


def test_tsb_by_hand():
    y = np.array([[0.0, 3.0, 0.0]])
    out = M.tsb_rates(y, 0.5, start=0, init=_init(2, 2, 1, d=0.5))
    # d: 0,25 -> 0,625 -> 0,3125; z: 2, 2,5, 2,5
    assert out[0].tolist() == pytest.approx([1.0, 0.25 * 2, 0.625 * 2.5, 0.3125 * 2.5])
    other = M.tsb_rates(y, 0.5, beta=0.1, start=0, init=_init(2, 2, 1, d=0.5))
    assert other[0, 1] == pytest.approx(0.45 * 2)


def test_simple_methods_by_hand():
    y = np.array([[0.0, 4.0, 2.0]])
    r = M.simple_rates(y)
    assert r["naive"][0].tolist() == [0.0, 0.0, 4.0, 2.0] and r["zero"][0].tolist() == [0.0] * 4 and r["mean"][0].tolist() == pytest.approx([0.0, 0.0, 2.0, 2.0])


# --- unabhängige Schleife je Artikel -----------------------------------------------------------------------------------------------------------


def _loop_croston(y, alpha, start, variant):
    first = y[:start]
    nz = [v for v in first if v > 0]
    z = float(np.mean(nz)) if nz else 1.0
    p = start / len(nz) if nz else float(start)
    d = 1.0 / p
    last = max([i for i, v in enumerate(first) if v > 0], default=-1)
    q = float(start - last)
    out = [(1 - alpha / 2 if variant == "sba" else 1.0) * z / p] * (start + 1)
    if variant == "tsb":
        out = [d * z] * (start + 1)
    for t in range(start, len(y)):
        if variant == "tsb":
            d += alpha * ((1.0 if y[t] > 0 else 0.0) - d)
            if y[t] > 0:
                z += alpha * (y[t] - z)
            out.append(d * z)
        else:
            if y[t] > 0:
                z += alpha * (y[t] - z)
                p += alpha * (q - p)
                q = 1.0
            else:
                q += 1.0
            out.append(z / p if variant == "croston" else (1 - alpha / 2) * z / p)
    return np.array(out)


def test_vectorised_methods_equal_an_independent_loop():
    port = S.generate(n_items=12, seed=4)
    y = port.y
    cro, sba = M.croston_rates(y, 0.15)
    tsb = M.tsb_rates(y, 0.15)
    ses = M.ses_rates(y, 0.15)
    for i in range(12):
        assert cro[i] == pytest.approx(_loop_croston(y[i], 0.15, C.INIT_DAYS, "croston"))
        assert sba[i] == pytest.approx(_loop_croston(y[i], 0.15, C.INIT_DAYS, "sba"))
        assert tsb[i] == pytest.approx(_loop_croston(y[i], 0.15, C.INIT_DAYS, "tsb"))
        l = y[i, :C.INIT_DAYS].mean()
        ref = [l] * (C.INIT_DAYS + 1)
        for t in range(C.INIT_DAYS, y.shape[1]):
            l += 0.15 * (y[i, t] - l)
            ref.append(l)
        assert ses[i] == pytest.approx(ref)


def test_ses_matches_statsmodels_for_fixed_alpha_and_initial_level():
    sm = pytest.importorskip("statsmodels.tsa.holtwinters")
    y = S.generate(n_items=3, seed=2).y
    for i in range(3):
        res = sm.SimpleExpSmoothing(y[i], initialization_method="known", initial_level=float(y[i, :C.INIT_DAYS].mean())).fit(smoothing_level=0.2, optimized=False)
        mine = M.ses_rates(y[i:i + 1], 0.2, start=0, init=_init(1, 1, 1, l=float(y[i, :C.INIT_DAYS].mean())))[0]
        assert mine[:-1] == pytest.approx(np.asarray(res.fittedvalues), abs=1e-9)


# --- Eigenschaften ----------------------------------------------------------------------------------------------------------------------------


def test_croston_stays_constant_between_demands_and_tsb_decays():
    y = np.zeros((1, 400))
    y[0, ::20] = 5.0
    cro, _ = M.croston_rates(y, 0.1)
    tsb = M.tsb_rates(y, 0.1)
    quiet = np.arange(C.INIT_DAYS + 1, 400)
    inside = [t for t in quiet if y[0, t - 1] == 0 and y[0, t] == 0]
    assert all(abs(cro[0, t + 1] - cro[0, t]) < 1e-12 for t in inside)
    assert all(tsb[0, t + 1] < tsb[0, t] for t in inside)
    dead = np.zeros((1, 400))
    dead[0, :100:10] = 5.0
    assert M.croston_rates(dead, 0.1)[0][0, -1] == pytest.approx(5.0 / 18.0) and M.tsb_rates(dead, 0.1)[0, -1] < 0.05


def test_sba_is_croston_times_one_minus_half_alpha():
    y = S.generate(n_items=5, seed=1).y
    cro, sba = M.croston_rates(y, 0.3)
    assert sba == pytest.approx(0.85 * cro)


def test_rates_are_available_for_every_origin_and_use_only_the_past():
    y = S.generate(n_items=5, seed=1).y
    y2 = y.copy()
    y2[:, 900:] += 7
    for f in (lambda v: M.ses_rates(v, 0.1), lambda v: M.croston_rates(v, 0.1)[0], lambda v: M.tsb_rates(v, 0.1)):
        a, b = f(y), f(y2)
        assert a.shape == (5, C.N_DAYS + 1) and np.array_equal(a[:, :901], b[:, :901]) and not np.array_equal(a[:, 902:], b[:, 902:])


def test_grid_and_fixed_rates_agree():
    y = S.generate(n_items=4, seed=1).y
    grid = M.grid_rates(y, (0.05, 0.2))
    fixed = M.fixed_rates(y, 0.2)
    assert set(grid) == {0.05, 0.2} and all(np.array_equal(grid[0.2][m], fixed[m]) for m in fixed)


def test_one_step_mse_by_hand():
    rates = np.array([[9.0, 1.0, 2.0, 3.0, 9.0]])
    y = np.array([[0.0, 0.0, 4.0, 0.0, 0.0]])
    assert M.one_step_mse(rates, y, 1, 4)[0] == pytest.approx(((1 - 0) ** 2 + (2 - 4) ** 2 + (3 - 0) ** 2) / 3)


def test_choose_alpha_picks_the_smallest_in_sample_error():
    y = S.generate(n_items=30, seed=3, drift=0.0).y
    grid = M.grid_rates(y)
    rates, alphas = M.choose_alpha(grid, y, "ses")
    for i in (0, 7, 19):
        mse = {a: M.one_step_mse(grid[a]["ses"], y)[i] for a in grid}
        assert alphas[i] == min(mse, key=mse.get) and rates[i] == pytest.approx(grid[alphas[i]]["ses"][i])
    assert set(alphas) <= set(C.ALPHA_GRID) and np.mean(alphas) < 0.1


def test_all_rates_contains_every_method():
    y = S.generate(n_items=6, seed=1).y
    rates, chosen = M.all_rates(y, 0.1)
    assert set(rates) == {m for m in C.METHODS} and set(chosen) == {"ses_opt", "sba_opt", "tsb_opt"} and all(r.shape == (6, C.N_DAYS + 1) for r in rates.values())
    assert np.array_equal(rates["ses"], M.ses_rates(y, 0.1)) and np.array_equal(M.all_rates(y, 0.1234)[0]["croston"], M.croston_rates(y, 0.1234)[0])
