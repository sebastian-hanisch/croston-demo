"""Das Portfolio, die Kennzahlen von Hand, die Auswertung und die vier Experimente."""

import numpy as np
import pytest

import cr_constants as C
import cr_evaluation as E
import cr_scenario as S


def test_generate_is_reproducible_shaped_and_integer():
    a, b, c = S.generate(n_items=20, seed=4), S.generate(n_items=20, seed=4), S.generate(n_items=20, seed=5)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.y.shape == (20, C.N_DAYS) and a.mu.shape == a.y.shape and a.n == 20
    assert np.all(a.y >= 0) and np.all(a.y == np.round(a.y)) and a.y[a.y > 0].min() == 1.0


def test_occurrence_rate_and_mean_size_follow_the_parameters():
    port = S.generate(n_items=300, p_mean=0.2, size_mean=6.0, cv=0.3, drift=0.0, seed=1)
    share = (port.y > 0).mean(axis=1)
    assert np.abs(share - port.p).mean() < 0.02
    nz = np.where(port.y > 0, port.y, np.nan)
    assert np.nanmean(np.nanmean(nz, axis=1) / port.m) == pytest.approx(1.0, abs=0.05)
    assert port.mu == pytest.approx(port.p[:, None] * port.m[:, None] * np.ones((1, C.N_DAYS)), rel=0.01)                  # E[Menge] liegt wegen Rundung und Mindestmaß 1 knapp über m (hier unter 1 %; exakt: tests/test_oracle_croston.py)


def test_drift_changes_the_expected_rate_and_zero_drift_keeps_it_constant():
    still = S.generate(n_items=30, drift=0.0, seed=2)
    moving = S.generate(n_items=30, drift=0.04, seed=2)
    assert np.allclose(still.mu, still.mu[:, :1]) and np.std(np.log(moving.mu[:, -1] / moving.mu[:, 0])) > 0.5


def test_obsolete_items_die_out():
    port = S.generate(n_items=200, obs_share=0.5, drift=0.0, seed=3)
    o = port.obsolete
    assert 0.35 < o.mean() < 0.65 and np.all(port.t_dead[~o] == -1) and np.all(port.t_dead[o] >= C.OBS_WINDOW[0])
    late = port.y[o][:, -100:].sum(axis=1)
    early = port.y[o][:, :300].sum(axis=1)
    assert late.mean() < 0.05 * early.mean() and np.all(port.mu[~o] == port.mu[~o][:, :1])


def test_lead_sums_and_scale_by_hand():
    y = np.array([[0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0]])
    assert E.lead_sums(y, 3, np.array([0, 1, 7])).tolist() == [[1.0, 3.0, 4.0]]
    sc = E.scale_of(y, 3, first=10)
    now = [y[0, t:t + 3].sum() for t in range(3, 8)]
    prev = [y[0, t - 3:t].sum() for t in range(3, 8)]
    assert sc[0] == pytest.approx(max(np.sqrt(np.mean((np.array(now) - np.array(prev)) ** 2)), C.SCALE_FLOOR))
    assert E.scale_of(np.zeros((1, 20)), 2, first=20)[0] == C.SCALE_FLOOR


def test_metrics_by_hand():
    err = np.array([[1.0, -1.0, 2.0], [0.0, 3.0, -3.0]])
    actual = np.array([[4.0, 4.0, 4.0], [2.0, 2.0, 2.0]])
    m = E.metrics(err, actual, np.array([1.0, 2.0]))
    assert m["mae"] == pytest.approx(10 / 6) and m["rmse"] == pytest.approx(np.sqrt(24 / 6)) and m["me"] == pytest.approx(2 / 6) and m["me_pct"] == pytest.approx(100 * (2 / 6) / 3.0)
    assert m["rmsse"] == pytest.approx(np.sqrt(np.mean([2.0 / 1.0, 6.0 / 4.0]))) and m["mse_item"].tolist() == pytest.approx([2.0, 6.0])


def test_item_stats_by_hand():
    y = np.zeros((2, C.FIRST_TEST + 5))
    y[0, [10, 20, 30, 40]] = [2, 4, 2, 4]
    adi, cv2 = E.item_stats(y)
    assert adi[0] == pytest.approx(C.FIRST_TEST / 4) and cv2[0] == pytest.approx((1.0 / 3.0) ** 2) and np.isinf(adi[1]) and cv2[1] == 0.0


def test_analyse_is_consistent_and_the_oracle_is_best():
    s = E.Settings(n_items=60, seed=2)
    a = E.analyse(s)
    assert set(a.summary) == set(C.METHODS) | {"oracle"} and len(a.origins) == C.N_DAYS - s.horizon + 1 - C.FIRST_TEST and a.actual.shape == (60, len(a.origins))
    assert a.summary["oracle"]["rmsse"] == min(v["rmsse"] for v in a.summary.values()) and a.best in C.METHODS and a.best_mae in C.METHODS
    assert a.summary["zero"]["me_pct"] == pytest.approx(-100.0) and a.errors["zero"] == pytest.approx(-a.actual)
    for m in ("ses", "croston"):
        assert a.errors[m] + a.actual == pytest.approx(s.horizon * a.rates[m][:, a.origins])
    assert a.errors["oracle"] + a.actual == pytest.approx(np.stack([a.port.mu[:, t:t + s.horizon].sum(axis=1) for t in a.origins], axis=1))


def test_caches_are_shared_across_alpha_and_horizon():
    E._grid.cache_clear()
    E.analyse(E.Settings(n_items=40, seed=9))
    n = E._grid.cache_info().misses
    E.analyse(E.Settings(n_items=40, seed=9, alpha=0.2, horizon=14))
    assert E._grid.cache_info().misses == n


def test_experiments_return_consistent_rows():
    me = E.metric_experiment(horizons=(1, 7), seeds=(0, 1), base=E.Settings(n_items=60))
    assert [r["horizon"] for r in me] == [1, 7] and me[0]["methods"]["zero"]["mae"] < me[0]["methods"]["ses"]["mae"] and me[1]["methods"]["zero"]["mae"] > me[1]["methods"]["oracle"]["mae"]
    bi = E.bias_experiment(alphas=(0.1, 0.3), p_means=(0.1,), seeds=(0, 1), base=E.Settings(n_items=80))
    assert [r["alpha"] for r in bi] == [0.1, 0.3] and bi[1]["croston"] > bi[0]["croston"] > 0 and bi[1]["sba"] < bi[1]["croston"]
    al = E.alpha_experiment(alphas=(0.05, 0.3), seeds=(0, 1), base=E.Settings(n_items=60))
    assert al["fixed"][("ses", 0.3)][0] > al["fixed"][("ses", 0.05)][0] and set(al["opt"]) == {"ses_opt", "sba_opt", "tsb_opt"} and al["ref"]["oracle"] < al["ref"]["mean"]
    ob = E.obsolescence_experiment(seeds=(0, 1), base=E.Settings(n_items=100))
    assert ob["n_dead"] > 10 and ob["ratio"]["croston"] > 5 * ob["ratio"]["tsb"] and ob["dead"]["croston"] > ob["dead"]["tsb"]
