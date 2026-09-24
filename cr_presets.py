"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. ari_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cr_constants as C


def _bool(value):
    v = str(value).strip().lower()
    if v in ("1", "true", "ja", "yes"):
        return True
    if v in ("0", "false", "nein", "no"):
        return False
    raise ValueError(value)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "items_slider": SettingSpec("items", int, C.DEFAULT_ITEMS, C.ITEMS_MIN, C.ITEMS_MAX),
    "p_slider": SettingSpec("p", float, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "size_slider": SettingSpec("size", float, C.DEFAULT_SIZE, C.SIZE_MIN, C.SIZE_MAX),
    "cv_slider": SettingSpec("cv", float, C.DEFAULT_CV, C.CV_MIN, C.CV_MAX),
    "drift_slider": SettingSpec("drift", float, C.DEFAULT_DRIFT, C.DRIFT_MIN, C.DRIFT_MAX),
    "obs_slider": SettingSpec("obs", float, C.DEFAULT_OBS, C.OBS_MIN, C.OBS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_items": "items_slider", "p_mean": "p_slider", "size_mean": "size_slider", "cv": "cv_slider", "drift": "drift_slider", "obs": "obs_slider", "horizon": "horizon_slider", "alpha": "alpha_slider", "seed": "seed_input"}
STEPS = {"items_slider": C.ITEMS_STEP, "p_slider": C.P_STEP, "size_slider": C.SIZE_STEP, "cv_slider": C.CV_STEP, "drift_slider": C.DRIFT_STEP, "obs_slider": C.OBS_STEP, "alpha_slider": C.ALPHA_STEP}


def _p(**kw):
    base = {"n_items": C.DEFAULT_ITEMS, "p_mean": C.DEFAULT_P, "size_mean": C.DEFAULT_SIZE, "cv": C.DEFAULT_CV, "drift": C.DEFAULT_DRIFT, "obs": C.DEFAULT_OBS, "horizon": C.DEFAULT_HORIZON, "alpha": C.DEFAULT_ALPHA, "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: 200 Artikel, p = 0,15": _p(),
    "Sehr sporadisch (p = 0,03)": _p(p_mean=0.03),
    "Klumpige Mengen (CV 1,5)": _p(cv=1.5),
    "Auslaufende Artikel (30 %)": _p(obs=0.3),
    "Grobes α = 0,3": _p(alpha=0.3),
    "Wiederbeschaffungszeit 1 Tag": _p(horizon=1),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: 200 Artikel, p = 0,15": "Seed 3, 200 Artikel, 78 % Nulltage im Testjahr, Wiederbeschaffungszeit 7 Tage: RMSSE TSB (α gewählt) 0,756, SBA (α = 0,1) 0,770, Croston 0,773, Mittel der Historie 0,791, einfache Glättung 0,856, naiv 2,04, Null 1,24; Orakel 0,733. "
                                           "Verzerrung: Croston +2,8 %, SBA −2,3 %.",
    "Sehr sporadisch (p = 0,03)": "Seed 3, 95 % Nulltage: nach RMSSE liegen Croston und SBA (0,793) vor der einfachen Glättung (0,907); nach dem MAE dagegen gewinnt die Null-Prognose (1,85 Stück gegen 2,23 beim Orakel) - die falsche Kennzahl.",
    "Klumpige Mengen (CV 1,5)": "Seed 3: RMSSE Croston 0,879, SBA 0,875, einfache Glättung 0,982, TSB (α gewählt) 0,852; Orakel 0,839. Klumpige Mengen heben das Niveau aller Fehler, ändern die Rangfolge aber kaum.",
    "Auslaufende Artikel (30 %)": "Seed 3, 64 auslaufende Artikel: Croston überschätzt um +16,5 %, SBA um +10,7 %, einfache Glättung (+0,7 %) und TSB (+0,5 %) nicht; RMSSE TSB (α gewählt) 0,662, einfache Glättung 0,740, SBA 0,688, Croston 0,694.",
    "Grobes α = 0,3": "Seed 3, α = 0,3: die einfache Glättung fällt auf RMSSE 1,085, TSB auf 1,026, Croston auf 0,859 (Verzerrung +10,4 %), SBA auf 0,827; mit je Artikel gewähltem α sind es wieder 0,756 bis 0,764.",
    "Wiederbeschaffungszeit 1 Tag": "Seed 3, ein Tag: die Null-Prognose hat den kleinsten MAE (1,21 Stück gegen 1,54 beim Orakel), aber einen RMSSE von 0,81 gegen 0,717 beim Orakel und 100 % Verzerrung nach unten.",
}
