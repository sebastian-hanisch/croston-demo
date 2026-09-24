"""Jede Zahl aus README und PRESET_HELP als Test. Portfolios und Verfahren sind deterministisch (kein Training, feste Seeds); die Bänder sind trotzdem großzügiger als die Rundung, damit andere numpy-Versionen nicht stören."""

import numpy as np
import pytest

import cr_constants as C
import cr_evaluation as E
import cr_presets as P

STD = "Standardfall: 200 Artikel, p = 0,15"


def _preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["n_items"], p["p_mean"], p["size_mean"], p["cv"], p["drift"], p["obs"], p["horizon"], p["alpha"], p["seed"]))


def _r(a):
    return {k: v["rmsse"] for k, v in a.summary.items()}


def test_standard_preset():
    a = _preset(STD)
    r = _r(a)
    assert r["tsb_opt"] == pytest.approx(0.756, abs=0.01) and r["sba"] == pytest.approx(0.770, abs=0.01) and r["croston"] == pytest.approx(0.773, abs=0.01) and r["mean"] == pytest.approx(0.791, abs=0.01) and r["ses"] == pytest.approx(0.856, abs=0.015)
    assert r["naive"] == pytest.approx(2.04, abs=0.05) and r["zero"] == pytest.approx(1.24, abs=0.03) and r["oracle"] == pytest.approx(0.733, abs=0.01) and a.best == "tsb_opt"
    assert a.summary["croston"]["me_pct"] == pytest.approx(2.8, abs=0.6) and a.summary["sba"]["me_pct"] == pytest.approx(-2.3, abs=0.6) and float(np.mean(a.port.y[:, C.FIRST_TEST:] == 0)) == pytest.approx(0.78, abs=0.01)


def test_sporadic_preset_and_the_mae_trap():
    a = _preset("Sehr sporadisch (p = 0,03)")
    r = _r(a)
    assert r["croston"] == pytest.approx(0.793, abs=0.01) and r["sba"] == pytest.approx(0.793, abs=0.01) and r["ses"] == pytest.approx(0.907, abs=0.015) and float(np.mean(a.port.y[:, C.FIRST_TEST:] == 0)) == pytest.approx(0.953, abs=0.01)
    assert a.best_mae == "zero" and a.summary["zero"]["mae"] == pytest.approx(1.85, abs=0.05) and a.summary["oracle"]["mae"] == pytest.approx(2.23, abs=0.05) and a.summary["zero"]["mae"] < a.summary["oracle"]["mae"]


def test_lumpy_preset():
    r = _r(_preset("Klumpige Mengen (CV 1,5)"))
    assert r["croston"] == pytest.approx(0.879, abs=0.015) and r["sba"] == pytest.approx(0.875, abs=0.015) and r["ses"] == pytest.approx(0.982, abs=0.02) and r["tsb_opt"] == pytest.approx(0.852, abs=0.015) and r["oracle"] == pytest.approx(0.839, abs=0.01)


def test_obsolescence_preset():
    a = _preset("Auslaufende Artikel (30 %)")
    r = _r(a)
    assert int(a.port.obsolete.sum()) == 64 and a.summary["croston"]["me_pct"] == pytest.approx(16.5, abs=2) and a.summary["sba"]["me_pct"] == pytest.approx(10.7, abs=2) and a.summary["ses"]["me_pct"] == pytest.approx(0.7, abs=1) and a.summary["tsb"]["me_pct"] == pytest.approx(0.5, abs=1)
    assert r["tsb_opt"] == pytest.approx(0.662, abs=0.015) and r["ses"] == pytest.approx(0.740, abs=0.015) and r["sba"] == pytest.approx(0.688, abs=0.015) and r["croston"] == pytest.approx(0.694, abs=0.015)


def test_coarse_alpha_preset():
    a = _preset("Grobes α = 0,3")
    r = _r(a)
    assert r["ses"] == pytest.approx(1.085, abs=0.03) and r["tsb"] == pytest.approx(1.026, abs=0.03) and r["croston"] == pytest.approx(0.859, abs=0.02) and r["sba"] == pytest.approx(0.827, abs=0.02) and a.summary["croston"]["me_pct"] == pytest.approx(10.4, abs=1.5)
    assert all(0.74 < r[m] < 0.78 for m in ("ses_opt", "sba_opt", "tsb_opt"))


def test_one_day_preset():
    a = _preset("Wiederbeschaffungszeit 1 Tag")
    assert a.best_mae == "zero" and a.summary["zero"]["mae"] == pytest.approx(1.21, abs=0.04) and a.summary["oracle"]["mae"] == pytest.approx(1.54, abs=0.04) and _r(a)["zero"] == pytest.approx(0.81, abs=0.02) and _r(a)["oracle"] == pytest.approx(0.717, abs=0.01)


def test_metric_experiment():
    rows = {r["horizon"]: r["methods"] for r in E.metric_experiment()}
    one, seven, long = rows[1], rows[7], rows[28]
    assert one["zero"]["mae"] == pytest.approx(1.11, abs=0.04) and one["oracle"]["mae"] == pytest.approx(1.46, abs=0.04) and one["zero"]["mae"] < one["oracle"]["mae"] and one["zero"]["rmse"] == pytest.approx(3.08, abs=0.1) and one["oracle"]["rmse"] == pytest.approx(2.63, abs=0.1)
    assert long["zero"]["mae"] == pytest.approx(31.1, abs=1.5) and long["oracle"]["mae"] == pytest.approx(9.8, abs=0.5) and seven["naive"]["rmse"] == pytest.approx(19.6, abs=1.0) and seven["oracle"]["rmse"] == pytest.approx(7.0, abs=0.3)
    assert all(rows[h]["zero"]["me_pct"] == pytest.approx(-100.0) for h in rows) and max(seven, key=lambda m: seven[m]["rmse"]) == "naive"


def test_bias_experiment():
    rows = E.bias_experiment()
    cro = max(rows, key=lambda x: x["croston"])
    sba = min(rows, key=lambda x: x["sba"])
    assert cro["croston"] == pytest.approx(17.4, abs=2) and (cro["p_mean"], cro["alpha"]) == (0.03, 0.3) and sba["sba"] == pytest.approx(-7.2, abs=1.5) and (sba["p_mean"], sba["alpha"]) == (0.3, 0.3)
    assert max(max(abs(x["ses"]), abs(x["tsb"])) for x in rows) < 1.0 and all(x["croston"] > 0 for x in rows if x["alpha"] >= 0.1)


def test_alpha_experiment():
    r = E.alpha_experiment()
    f = {k: v[0] for k, v in r["fixed"].items()}
    assert f[("ses", 0.02)] == pytest.approx(0.756, abs=0.02) and f[("ses", 0.2)] == pytest.approx(0.966, abs=0.03) and f[("ses", 0.3)] == pytest.approx(1.08, abs=0.04) and f[("sba", 0.2)] == pytest.approx(0.786, abs=0.02) and f[("croston", 0.05)] == pytest.approx(0.750, abs=0.02)
    assert (f[("ses", 0.2)] - f[("sba", 0.2)]) / f[("ses", 0.2)] == pytest.approx(0.19, abs=0.03)
    assert f[("ses", 0.05)] == pytest.approx(0.789, abs=0.02) and f[("ses", 0.1)] == pytest.approx(0.849, abs=0.025)
    o = {m: v[0] for m, v in r["opt"].items()}
    assert o["ses_opt"] == pytest.approx(0.750, abs=0.015) and o["sba_opt"] == pytest.approx(0.756, abs=0.015) and o["tsb_opt"] == pytest.approx(0.747, abs=0.015) and max(o.values()) - min(o.values()) < 0.02
    assert r["chosen"]["ses_opt"] == pytest.approx(0.012, abs=0.006) and r["chosen"]["sba_opt"] == pytest.approx(0.05, abs=0.02) and r["ref"]["mean"] == pytest.approx(0.793, abs=0.02) and r["ref"]["oracle"] == pytest.approx(0.725, abs=0.01)


def test_obsolescence_experiment():
    r = E.obsolescence_experiment()
    assert r["n_dead"] == pytest.approx(62, abs=6) and r["dead"]["ses"] == pytest.approx(3.0, abs=0.3) and r["dead"]["croston"] == pytest.approx(4.07, abs=0.4) and r["dead"]["sba"] == pytest.approx(3.9, abs=0.4) and r["dead"]["tsb"] == pytest.approx(2.9, abs=0.3) and r["dead"]["mean"] == pytest.approx(7.0, abs=0.5)
    assert r["live"]["ses"] == pytest.approx(7.9, abs=0.4) and r["live"]["croston"] == pytest.approx(7.1, abs=0.4) and r["live"]["sba"] == pytest.approx(7.05, abs=0.4) and r["live"]["tsb"] == pytest.approx(7.7, abs=0.4) and r["live"]["mean"] == pytest.approx(6.8, abs=0.4)
    assert r["ratio"]["croston"] == pytest.approx(53, abs=15) and r["ratio"]["sba"] == pytest.approx(51, abs=15) and r["ratio"]["ses"] == pytest.approx(1.13, abs=0.3) and r["ratio"]["tsb"] == pytest.approx(1.08, abs=0.3) and r["ratio"]["mean"] == pytest.approx(98, abs=25)
    assert r["dead"]["tsb"] < r["dead"]["croston"] and r["live"]["croston"] < r["live"]["tsb"] < r["live"]["ses"] and r["live"]["mean"] < r["live"]["croston"]
