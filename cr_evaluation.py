"""Auswertung: Prognoseverfahren für sporadische Nachfrage im Rolling-Origin-Vergleich auf einem Portfolio und vier Experimente (Kennzahl, Verzerrung, Glättungsparameter, Auslaufen).

Geprüft wird die Nachfrage in der Wiederbeschaffungszeit: am Ursprung t gegen die Summe der Tage t..t+h-1 (die Prognose ist h mal die Rate). Kennzahlen (gepoolt über alle Artikel und Ursprünge): MAE und RMSE in Stück, die Verzerrung
(mittlerer Fehler in Prozent der mittleren Nachfrage) und der skalierte RMSSE: der mittlere quadrierte Fehler je Artikel geteilt durch das Quadrat des Fehlers der Prognose "die letzte Wiederbeschaffungszeit wiederholt sich" auf den Trainingstagen, gemittelt über die Artikel, Wurzel."""

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import cr_constants as C
import cr_methods as M
import cr_scenario as S


@dataclass(frozen=True)
class Settings:
    n_items: int = C.DEFAULT_ITEMS
    p_mean: float = C.DEFAULT_P
    size_mean: float = C.DEFAULT_SIZE
    cv: float = C.DEFAULT_CV
    drift: float = C.DEFAULT_DRIFT
    obs: float = C.DEFAULT_OBS
    horizon: int = C.DEFAULT_HORIZON
    alpha: float = C.DEFAULT_ALPHA
    seed: int = 3

    @property
    def portfolio_key(self):
        return (self.n_items, self.p_mean, self.size_mean, self.cv, self.drift, self.obs, self.seed)


@dataclass
class Analysis:
    settings: Settings
    port: S.Portfolio
    rates: dict                # Verfahren -> (n, T+1)
    chosen: dict               # Optimierungsvariante -> alpha je Artikel
    origins: np.ndarray
    actual: np.ndarray         # (n, n_origins) Nachfrage in der Wiederbeschaffungszeit
    errors: dict               # Verfahren -> (n, n_origins) Prognose minus Ist
    scale: np.ndarray          # (n,) Skala der skalierten Fehler
    summary: dict

    @property
    def best(self):
        return min((m for m in self.summary if m != "oracle"), key=lambda m: self.summary[m]["rmsse"])

    @property
    def best_mae(self):
        return min((m for m in self.summary if m != "oracle"), key=lambda m: self.summary[m]["mae"])


@lru_cache(maxsize=64)
def _portfolio(key):
    n, p, size, cv, drift, obs, seed = key
    return S.generate(n, p, size, cv, drift, obs, seed)


@lru_cache(maxsize=32)
def _grid(key):
    return M.grid_rates(_portfolio(key).y)


@lru_cache(maxsize=128)
def _rates(key, alpha):
    port = _portfolio(key)
    return M.all_rates(port.y, alpha, _grid(key))


def lead_sums(y, h, org):
    cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)], axis=1)
    return cs[:, org + h] - cs[:, org]


def scale_of(y, h, first=C.FIRST_TEST):
    """Skala je Artikel: quadratischer Mittelfehler der Prognose 'die vorige Wiederbeschaffungszeit wiederholt sich' auf den Trainingstagen (Wurzel), mindestens SCALE_FLOOR."""
    cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)], axis=1)
    t = np.arange(h, first - h + 1)
    now, prev = cs[:, t + h] - cs[:, t], cs[:, t] - cs[:, t - h]
    return np.maximum(np.sqrt(np.mean((now - prev) ** 2, axis=1)), C.SCALE_FLOOR)


def metrics(err, actual, scale):
    mse_item = np.mean(err ** 2, axis=1)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(err ** 2))), "me": float(err.mean()), "me_pct": float(100 * err.mean() / actual.mean()), "rmsse": float(np.sqrt(np.mean(mse_item / scale ** 2))), "mse_item": mse_item}


def evaluate(port, rates, h, first=C.FIRST_TEST):
    n, T = port.y.shape
    org = np.arange(first, T - h + 1)
    actual = lead_sums(port.y, h, org)
    scale = scale_of(port.y, h, first)
    errors = {m: h * r[:, org] - actual for m, r in rates.items()}
    cm = np.concatenate([np.zeros((n, 1)), np.cumsum(port.mu, axis=1)], axis=1)
    errors["oracle"] = (cm[:, org + h] - cm[:, org]) - actual
    return org, actual, errors, scale, {m: metrics(e, actual, scale) for m, e in errors.items()}


def analyse(s):
    port = _portfolio(s.portfolio_key)
    rates, chosen = _rates(s.portfolio_key, s.alpha)
    org, actual, errors, scale, summary = evaluate(port, rates, s.horizon)
    return Analysis(s, port, rates, chosen, org, actual, errors, scale, summary)


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


def item_stats(y, first=C.FIRST_TEST):
    """Durchschnittlicher Abstand zwischen Bedarfstagen (ADI) und quadrierter Variationskoeffizient der Mengen (CV²) je Artikel auf den Trainingstagen."""
    tr = y[:, :first]
    cnt = (tr > 0).sum(axis=1)
    adi = np.where(cnt > 0, first / np.maximum(cnt, 1), np.inf)
    vals = np.where(tr > 0, tr, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)
        cv2 = np.where(cnt > 1, (std / mean) ** 2, 0.0)
    return adi, cv2


# --- Experiment 1: Kennzahl ---------------------------------------------------------------------------------------------------------------------


def metric_experiment(horizons=None, seeds=None, base=None):
    """Alle Verfahren für kurze und lange Wiederbeschaffungszeiten: MAE, RMSE und Verzerrung (Mittel über die Seeds)."""
    horizons = C.METRIC_HORIZONS if horizons is None else horizons
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for h in horizons:
        acc = {}
        for sd in seeds:
            a = analyse(_replace(base, horizon=h, seed=sd))
            for m, v in a.summary.items():
                acc.setdefault(m, []).append((v["mae"], v["rmse"], v["me_pct"], v["rmsse"]))
        res = {m: dict(zip(("mae", "rmse", "me_pct", "rmsse"), np.mean(v, axis=0))) for m, v in acc.items()}
        rows.append({"horizon": h, "n_seeds": len(seeds), "methods": {m: {k: float(x) for k, x in d.items()} for m, d in res.items()}})
    return rows


# --- Experiment 2: Verzerrung ---------------------------------------------------------------------------------------------------------------------


def bias_experiment(alphas=None, p_means=None, seeds=None, base=None):
    """Verzerrung (mittlerer Fehler in Prozent der Nachfrage) von einfacher Glättung, Croston, SBA und TSB für verschiedene alpha und Bedarfshäufigkeiten (stabile Nachfrage, Drift 0)."""
    alphas = C.BIAS_ALPHAS if alphas is None else alphas
    p_means = C.BIAS_P if p_means is None else p_means
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = _replace(Settings() if base is None else base, drift=0.0, obs=0.0)
    rows = []
    for p in p_means:
        for al in alphas:
            acc = {m: [] for m in ("ses", "croston", "sba", "tsb")}
            for sd in seeds:
                a = analyse(_replace(base, p_mean=p, alpha=al, seed=sd))
                for m in acc:
                    acc[m].append(a.summary[m]["me_pct"])
            rows.append({"p_mean": p, "alpha": al, "n_seeds": len(seeds), **{m: _mean_se(v)[0] for m, v in acc.items()}})
    return rows


# --- Experiment 3: Glättungsparameter --------------------------------------------------------------------------------------------------------------


def alpha_experiment(alphas=None, seeds=None, base=None):
    """RMSSE von einfacher Glättung, Croston, SBA und TSB bei festem alpha und mit je Artikel gewähltem alpha; dazu Mittel der Historie und Orakel."""
    alphas = C.ALPHA_LEVELS if alphas is None else alphas
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    fixed = {(m, al): [] for m in ("ses", "croston", "sba", "tsb") for al in alphas}
    opt = {m: [] for m in ("ses_opt", "sba_opt", "tsb_opt")}
    ref = {"mean": [], "oracle": [], "naive": []}
    chosen = {m: [] for m in opt}
    for sd in seeds:
        for al in alphas:
            a = analyse(_replace(base, alpha=al, seed=sd))
            for m in ("ses", "croston", "sba", "tsb"):
                fixed[(m, al)].append(a.summary[m]["rmsse"])
        for m in opt:
            opt[m].append(a.summary[m]["rmsse"])
            chosen[m].append(float(np.mean(a.chosen[m])))
        for m in ref:
            ref[m].append(a.summary[m]["rmsse"])
    return {"n_seeds": len(seeds), "alphas": tuple(alphas), "fixed": {k: _mean_se(v) for k, v in fixed.items()}, "opt": {m: _mean_se(v) for m, v in opt.items()}, "ref": {m: _mean_se(v)[0] for m, v in ref.items()},
            "chosen": {m: float(np.mean(v)) for m, v in chosen.items()}}


# --- Experiment 4: Auslaufen -----------------------------------------------------------------------------------------------------------------------


def obsolescence_experiment(seeds=None, base=None, share=None):
    """Portfolio mit auslaufenden Artikeln (Anteil OBS_EXP_SHARE, stabile Nachfrage sonst): RMSE (Stück) getrennt für auslaufende und aktive Artikel, dazu die Prognose für auslaufende Artikel in den letzten 100 Ursprüngen im Verhältnis zur wahren Nachfrage."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    share = C.OBS_EXP_SHARE if share is None else share
    base = _replace(Settings() if base is None else base, drift=0.0, obs=share)
    methods = ("ses", "croston", "sba", "tsb", "mean")
    dead, live, ratio = ({m: [] for m in methods} for _ in range(3))
    n_dead = []
    for sd in seeds:
        a = analyse(_replace(base, seed=sd))
        o = a.port.obsolete
        n_dead.append(int(o.sum()))
        late = slice(-100, None)
        true_late = np.array([a.actual[o][:, late].mean()])
        for m in methods:
            e = a.errors[m]
            dead[m].append(float(np.sqrt(np.mean(e[o] ** 2))))
            live[m].append(float(np.sqrt(np.mean(e[~o] ** 2))))
            ratio[m].append(float((e[o][:, late] + a.actual[o][:, late]).mean() / max(true_late[0], 1e-9)))
    return {"n_seeds": len(seeds), "share": share, "n_dead": float(np.mean(n_dead)), "dead": {m: float(np.mean(v)) for m, v in dead.items()}, "live": {m: float(np.mean(v)) for m, v in live.items()},
            "ratio": {m: float(np.mean(v)) for m, v in ratio.items()}, "alpha": base.alpha}
