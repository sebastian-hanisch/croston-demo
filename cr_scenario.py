"""Vehikel "Lagerartikel eines Depots mit sporadischer Nachfrage": ein Portfolio aus n Artikeln, je 1 095 Tage. An einem Tag gibt es Nachfrage mit der Wahrscheinlichkeit p_i(t), die Menge ist dann ganzzahlig und mindestens 1
(log-normal mit Mittel m_i und Variationskoeffizient cv). Die Häufigkeit driftet (Irrfahrt des Logarithmus, Streuung je Tag = drift); ein Anteil der Artikel läuft aus: ab einem zufälligen Tag um den Beginn des Testjahres
fällt die Häufigkeit mit der Zeitkonstanten OBS_TAU auf null. Der Erwartungswert je Tag (Orakel) ist p_i(t) * m_i."""

from dataclasses import dataclass

import numpy as np

import cr_constants as C


@dataclass(frozen=True)
class Portfolio:
    y: np.ndarray             # (n, T) Nachfrage je Tag
    mu: np.ndarray            # (n, T) Erwartungswert je Tag (Orakel)
    p: np.ndarray             # (n,) Ausgangshäufigkeit
    m: np.ndarray             # (n,) mittlere Menge je Bedarfstag
    obsolete: np.ndarray      # (n,) True, wenn der Artikel ausläuft
    t_dead: np.ndarray        # (n,) Tag, ab dem er ausläuft (-1: nie)
    seed: int

    @property
    def n(self):
        return self.y.shape[0]


def generate(n_items=C.DEFAULT_ITEMS, p_mean=C.DEFAULT_P, size_mean=C.DEFAULT_SIZE, cv=C.DEFAULT_CV, drift=C.DEFAULT_DRIFT, obs_share=C.DEFAULT_OBS, seed=0, n_days=C.N_DAYS):
    """p_mean: mittlere Häufigkeit eines Bedarfstags; size_mean: mittlere Menge; cv: Variationskoeffizient der Mengen; drift: Streuung des Log der Häufigkeit je Tag; obs_share: Anteil auslaufender Artikel."""
    rng = np.random.default_rng(seed)
    p = np.clip(p_mean * np.exp(C.SPREAD_P * rng.normal(size=n_items)), 0.01, 0.9)
    m = np.maximum(1.0, size_mean * np.exp(C.SPREAD_SIZE * rng.normal(size=n_items)))
    t = np.arange(n_days)[None, :]
    walk = np.cumsum(drift * rng.normal(size=(n_items, n_days)), axis=1)
    obsolete = rng.random(n_items) < obs_share
    t0 = rng.integers(C.OBS_WINDOW[0], C.OBS_WINDOW[1], size=n_items)
    decay = np.where(obsolete[:, None] & (t > t0[:, None]), np.exp(-(t - t0[:, None]) / C.OBS_TAU), 1.0)
    pt = np.clip(p[:, None] * np.exp(walk) * decay, 0.0, 0.95)
    occurs = rng.random((n_items, n_days)) < pt
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    size = np.maximum(1.0, np.rint(m[:, None] * np.exp(sigma * rng.normal(size=(n_items, n_days)) - 0.5 * sigma ** 2)))
    y = np.where(occurs, size, 0.0)
    return Portfolio(y, pt * m[:, None], p, m, obsolete, np.where(obsolete, t0, -1), int(seed))
