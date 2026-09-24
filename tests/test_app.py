"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Artikel-Regler, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cr_constants as C
import cr_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 5 and len(at.get("plotly_chart")) == 2 and len(at.info) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_null_forecast_warning_appears_for_sporadic_demand():
    at = _run(p_slider=0.03)
    _ok(at)
    assert any("gewinnt hier die Null-Prognose" in w.value for w in at.warning)


def test_item_slider_survives_a_smaller_portfolio_and_obsolete_items_show():
    at = _run(item_slider=190, obs_slider=0.5)
    _ok(at)
    at.slider(key="items_slider").set_value(50).run()
    _ok(at)
    assert at.session_state["item_slider"] <= 49


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Portfolio generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["p"] = "0.977"
    at.query_params["alpha"] = "9"
    at.query_params["horizon"] = "0"
    at.query_params["items"] = "77"
    at.query_params["cv"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["p_slider"] == C.P_MAX and at.session_state["alpha_slider"] == C.ALPHA_MAX and at.session_state["horizon_slider"] == C.HORIZON_MIN and at.session_state["items_slider"] == 100 and at.session_state["cv_slider"] == C.DEFAULT_CV


@pytest.mark.parametrize("kw", [dict(p_slider=C.P_MIN, size_slider=1.0), dict(p_slider=C.P_MAX, size_slider=C.SIZE_MAX, cv_slider=C.CV_MAX), dict(drift_slider=C.DRIFT_MAX, obs_slider=C.OBS_MAX), dict(horizon_slider=C.HORIZON_MAX, alpha_slider=C.ALPHA_MAX),
                                dict(horizon_slider=1, alpha_slider=C.ALPHA_MIN), dict(items_slider=C.ITEMS_MIN, drift_slider=0.0, obs_slider=0.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))


def test_metric_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run(items_slider=50)
    next(b for b in at.button if b.key == "metric_start").click().run()
    _ok(at)
    assert at.session_state["metric_on"] and any("Null-Prognose** einen MAE" in w.value for w in at.warning)


def test_bias_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "BIAS_ALPHAS", (0.1, 0.3))
    monkeypatch.setattr(C, "BIAS_P", (0.1, 0.3))
    at = _run(items_slider=50)
    next(b for b in at.button if b.key == "bias_start").click().run()
    _ok(at)
    assert at.session_state["bias_on"] and any("überschätzt" in w.value for w in at.warning)


def test_alpha_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "ALPHA_LEVELS", (0.05, 0.1, 0.2, 0.3))
    at = _run(items_slider=50)
    next(b for b in at.button if b.key == "alpha_start").click().run()
    _ok(at)
    assert at.session_state["alpha_on"] and any("Vorsprung von" in w.value for w in at.warning)


def test_obsolescence_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run(items_slider=100)
    next(b for b in at.button if b.key == "obs_start").click().run()
    _ok(at)
    assert at.session_state["obs_on"] and any("blind für das Auslaufen" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
