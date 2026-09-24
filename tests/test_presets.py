"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import cr_constants as C
import cr_evaluation as E
import cr_presets as P


def _settings(p):
    return E.Settings(p["n_items"], p["p_mean"], p["size_mean"], p["cv"], p["drift"], p["obs"], p["horizon"], p["alpha"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_items", "items_slider"), ("p_mean", "p_slider"), ("size_mean", "size_slider"), ("cv", "cv_slider"), ("drift", "drift_slider"), ("obs", "obs_slider"), ("alpha", "alpha_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: 200 Artikel, p = 0,15"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("p_slider") == (C.P_MIN, C.P_MAX) and P.bounds("alpha_slider") == (C.ALPHA_MIN, C.ALPHA_MAX)
    assert set(P.STEPS) == {"items_slider", "p_slider", "size_slider", "cv_slider", "drift_slider", "obs_slider", "alpha_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
