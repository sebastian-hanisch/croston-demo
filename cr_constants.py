"""Konstanten der Croston-Demo: Vehikel "Lagerartikel eines Depots mit sporadischer Nachfrage" (Stück 5 der Zeitreihen-Prognose-Linie), Verfahren, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095                                       # drei Jahre täglich
FIRST_TEST = 730                                    # Ursprünge liegen im letzten Jahr
INIT_DAYS = 180                                     # Anfangswerte der Verfahren aus den ersten 180 Tagen; danach wird gelernt
SCALE_FLOOR = 1.0                                   # kleinste Skala (Stück) im Nenner der skalierten Fehler

ITEMS_MIN, ITEMS_MAX, ITEMS_STEP, DEFAULT_ITEMS = 50, 400, 50, 200
P_MIN, P_MAX, P_STEP, DEFAULT_P = 0.02, 0.5, 0.01, 0.15                     # mittlere Wahrscheinlichkeit eines Bedarfstags
SIZE_MIN, SIZE_MAX, SIZE_STEP, DEFAULT_SIZE = 1.0, 20.0, 1.0, 5.0           # mittlere Bedarfsmenge an einem Bedarfstag
CV_MIN, CV_MAX, CV_STEP, DEFAULT_CV = 0.1, 1.5, 0.1, 0.6                    # Variationskoeffizient der Mengen
DRIFT_MIN, DRIFT_MAX, DRIFT_STEP, DEFAULT_DRIFT = 0.0, 0.05, 0.01, 0.02     # tägliche Streuung des Log der Bedarfshäufigkeit
OBS_MIN, OBS_MAX, OBS_STEP, DEFAULT_OBS = 0.0, 0.5, 0.1, 0.0                # Anteil auslaufender Artikel
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 7                        # Wiederbeschaffungszeit in Tagen
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.01, 0.5, 0.01, 0.10    # Glättungsparameter
SPREAD_P = 0.6                                      # Streuung (Log) der Bedarfshäufigkeit zwischen den Artikeln
SPREAD_SIZE = 0.4                                   # Streuung (Log) der mittleren Mengen zwischen den Artikeln
OBS_TAU = 60.0                                      # Zeitkonstante des Auslaufens in Tagen
OBS_WINDOW = (FIRST_TEST - 200, FIRST_TEST + 100)   # Tage, an denen ein auslaufender Artikel zu sterben beginnt

METHODS = ("naive", "zero", "mean", "ses", "croston", "sba", "tsb", "ses_opt", "sba_opt", "tsb_opt")
METHOD_NAMES = {
    "naive": "Naiv (letzter Tag)",
    "zero": "Null-Prognose",
    "mean": "Mittel der Historie",
    "ses": "Einfache Glättung (α)",
    "croston": "Croston (α)",
    "sba": "SBA (α)",
    "tsb": "TSB (α = β)",
    "ses_opt": "Einfache Glättung, α je Artikel gewählt",
    "sba_opt": "SBA, α je Artikel gewählt",
    "tsb_opt": "TSB, α je Artikel gewählt",
}
ALPHA_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)      # Kandidaten für das je Artikel gewählte alpha

# --- Experimente (feste Seeds) -------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(6))
METRIC_HORIZONS = (1, 7, 28)
BIAS_ALPHAS = (0.05, 0.1, 0.2, 0.3)
BIAS_P = (0.03, 0.1, 0.3)
ALPHA_LEVELS = (0.02, 0.05, 0.1, 0.2, 0.3)
OBS_EXP_SHARE = 0.3
