"""Prognoseverfahren für sporadische Nachfrage, vektorisiert über alle Artikel eines Portfolios.

Jedes Verfahren liefert eine Rate (erwartete Nachfrage je Tag); die Prognose für die Wiederbeschaffungszeit ist h mal die Rate. Spalte t der Ergebnismatrix (n, T+1) ist die Rate, die mit den Tagen 0..t-1 gebildet wird.

  naive     letzter Tag
  zero      immer 0
  mean      Mittel aller bisherigen Tage
  ses       einfache exponentielle Glättung der Tagesnachfrage: l <- l + alpha (y - l)
  croston   glättet Menge z und Abstand p getrennt und nur an Bedarfstagen: z <- z + alpha (y - z), p <- p + alpha (q - p) (q = Tage seit dem letzten Bedarf); Rate z / p
  sba       Syntetos-Boylan-Näherung: Croston mal (1 - alpha / 2), gegen die Verzerrung
  tsb       Teunter-Syntetos-Babai: glättet die Bedarfswahrscheinlichkeit d in jeder Periode (d <- d + beta (I - d)) und die Menge z an Bedarfstagen; Rate d z

Die Anfangswerte stammen aus den ersten INIT_DAYS Tagen; ab dann wird gelernt (start = INIT_DAYS). Das je Artikel gewählte alpha minimiert den Ein-Schritt-Fehler (Rate gegen Tagesnachfrage) auf den Trainingstagen."""

import numpy as np

import cr_constants as C


def init_state(y, init_days=C.INIT_DAYS):
    """Anfangswerte aus den ersten init_days Tagen: mittlere Menge z, mittlerer Abstand p, Tage seit dem letzten Bedarf q, Wahrscheinlichkeit d, Niveau l."""
    first = y[:, :init_days]
    nz = first > 0
    cnt = nz.sum(axis=1)
    has = cnt > 0
    z = np.where(has, (first * nz).sum(axis=1) / np.maximum(cnt, 1), 1.0)
    p = np.where(has, init_days / np.maximum(cnt, 1), float(init_days))
    last = np.where(has, init_days - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    q = (init_days - last).astype(float)
    return {"z": z, "p": p, "q": q, "d": 1.0 / p, "l": first.mean(axis=1)}


def ses_rates(y, alpha, start=C.INIT_DAYS, init=None):
    n, T = y.shape
    init = init_state(y, start) if init is None else init
    l = init["l"].copy()
    out = np.empty((n, T + 1))
    out[:, :start + 1] = l[:, None]
    for t in range(start, T):
        l = l + alpha * (y[:, t] - l)
        out[:, t + 1] = l
    return out


def croston_rates(y, alpha, start=C.INIT_DAYS, init=None):
    """Croston und SBA in einem Durchlauf: (croston, sba)."""
    n, T = y.shape
    init = init_state(y, start) if init is None else init
    z, p, q = init["z"].copy(), init["p"].copy(), init["q"].copy()
    out = np.empty((n, T + 1))
    out[:, :start + 1] = (z / p)[:, None]
    for t in range(start, T):
        occ = y[:, t] > 0
        z = np.where(occ, z + alpha * (y[:, t] - z), z)
        p = np.where(occ, p + alpha * (q - p), p)
        q = np.where(occ, 1.0, q + 1.0)
        out[:, t + 1] = z / p
    return out, (1.0 - alpha / 2.0) * out


def tsb_rates(y, alpha, beta=None, start=C.INIT_DAYS, init=None):
    n, T = y.shape
    beta = alpha if beta is None else beta
    init = init_state(y, start) if init is None else init
    z, d = init["z"].copy(), init["d"].copy()
    out = np.empty((n, T + 1))
    out[:, :start + 1] = (d * z)[:, None]
    for t in range(start, T):
        occ = y[:, t] > 0
        d = d + beta * (occ.astype(float) - d)
        z = np.where(occ, z + alpha * (y[:, t] - z), z)
        out[:, t + 1] = d * z
    return out


def simple_rates(y):
    """Verfahren ohne Parameter: naiv, null, Mittel der Historie."""
    n, T = y.shape
    naive = np.concatenate([np.zeros((n, 1)), y], axis=1)
    mean = np.concatenate([np.zeros((n, 1)), np.cumsum(y, axis=1) / np.arange(1, T + 1)[None, :]], axis=1)
    return {"naive": naive, "zero": np.zeros((n, T + 1)), "mean": mean}


def fixed_rates(y, alpha):
    """ses, croston, sba und tsb mit demselben alpha."""
    cro, sba = croston_rates(y, alpha)
    return {"ses": ses_rates(y, alpha), "croston": cro, "sba": sba, "tsb": tsb_rates(y, alpha)}


def grid_rates(y, grid=C.ALPHA_GRID):
    """Alle Verfahren für alle alpha des Gitters: {alpha: {Verfahren: Rate}}."""
    return {a: fixed_rates(y, a) for a in grid}


def one_step_mse(rates, y, start=C.INIT_DAYS, end=C.FIRST_TEST):
    """Ein-Schritt-Fehler je Artikel: die vor dem Tag gebildete Rate gegen die Tagesnachfrage, Tage start bis end-1."""
    return np.mean((rates[:, start:end] - y[:, start:end]) ** 2, axis=1)


def choose_alpha(grid, y, method, start=C.INIT_DAYS, end=C.FIRST_TEST):
    """Je Artikel das alpha des Gitters mit dem kleinsten Ein-Schritt-Fehler auf den Trainingstagen: (Raten (n, T+1), alpha je Artikel)."""
    alphas = sorted(grid)
    mse = np.stack([one_step_mse(grid[a][method], y, start, end) for a in alphas])
    best = np.argmin(mse, axis=0)
    stack = np.stack([grid[a][method] for a in alphas])
    rates = stack[best, np.arange(y.shape[0])]
    return rates, np.array(alphas)[best]


def all_rates(y, alpha, grid=None):
    """Alle Verfahren des Vergleichs: {Name: Rate (n, T+1)} und die gewählten alpha der Optimierungsvarianten."""
    grid = grid_rates(y) if grid is None else grid
    out = simple_rates(y)
    out.update(fixed_rates(y, alpha) if alpha not in grid else grid[alpha])
    chosen = {}
    for name, method in (("ses_opt", "ses"), ("sba_opt", "sba"), ("tsb_opt", "tsb")):
        out[name], chosen[name] = choose_alpha(grid, y, method)
    return out, chosen
