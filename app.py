"""Croston, SBA und TSB - Prognosen bei sporadischer Nachfrage - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: Lagerartikel eines Depots, an den meisten Tagen ohne Nachfrage. Die Glättung aus Stück 2 verzerrt hier; Croston, SBA und TSB glätten Menge und Abstand getrennt.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import cr_constants as C
import cr_evaluation as E
from cr_evaluation import Settings, alpha_experiment, analyse, bias_experiment, item_stats, metric_experiment, obsolescence_experiment
from cr_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from cr_visualization import SHORT, build_alpha, build_bars, build_bias, build_item, build_metric, build_obsolescence

st.set_page_config(page_title="Croston – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _metric(horizons, seeds):
    return metric_experiment(horizons=horizons, seeds=seeds)


@st.cache_data(show_spinner=False)
def _bias(alphas, p_means, seeds):
    return bias_experiment(alphas=alphas, p_means=p_means, seeds=seeds)


@st.cache_data(show_spinner=False)
def _alpha(alphas, seeds):
    return alpha_experiment(alphas=alphas, seeds=seeds)


@st.cache_data(show_spinner=False)
def _obs(seeds):
    return obsolescence_experiment(seeds=seeds)


st.title("🐌 Croston, SBA und TSB – Prognosen bei sporadischer Nachfrage")
st.markdown(
    """
Ein Ersatzteil, ein Spezialartikel, ein Nachzügler im Sortiment: **an den meisten Tagen wird nichts bestellt**, und wenn, dann in Klumpen. Die Demo zeigt an einem **Portfolio von Lagerartikeln eines Depots**, was Prognoseverfahren dafür leisten: die einfache Glättung aus Stück 2
und die drei Klassiker für sporadische Nachfrage - **Croston** (Menge und Abstand getrennt glätten), **SBA** (Crostons Verzerrung korrigiert) und **TSB** (die Bedarfswahrscheinlichkeit in jedem Tag fortschreiben). Geprüft wird die **Nachfrage in der Wiederbeschaffungszeit** - die Größe, aus der Bestellpunkte und
Sicherheitsbestände entstehen. Und es geht um eine Frage, die vor dem Verfahren kommt: **woran misst man überhaupt?**
"""
)
st.caption(
    "Fünftes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe (Ast für sporadische Nachfrage; Nachfolger der Exponentiellen Glättung, deren Schwäche - Nullen - es aufgreift); alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben. "
    "**Bezug zu OR:** die Prognose der Nachfrage in der Wiederbeschaffungszeit ist der Eingang jeder Bestandsrechnung; das Bestands-Stück der Linie nimmt sie auf."
)

with st.expander("So funktionieren die Verfahren", expanded=True):
    st.markdown(
        """
1. **Einfache Glättung** schreibt eine Rate fort: $\\ell \\leftarrow \\ell + \\alpha\\,(y - \\ell)$. An Nulltagen sinkt sie, an einem Bedarfstag springt sie: bei großem $\\alpha$ folgt sie jedem Klumpen.
2. **Croston** trennt zwei Fragen: *wie groß ist ein Bedarf?* ($z$) und *wie viele Tage liegen zwischen zwei Bedarfen?* ($p$). Beide werden **nur an Bedarfstagen** geglättet, die Rate ist $z/p$; zwischen zwei Bedarfen bleibt sie unverändert.
3. **SBA** (Syntetos-Boylan) multipliziert Crostons Rate mit $1 - \\alpha/2$: Croston überschätzt im Mittel, weil der Quotient zweier Schätzungen verzerrt ist.
4. **TSB** (Teunter-Syntetos-Babai) schätzt statt des Abstands die **Wahrscheinlichkeit** eines Bedarfstags, $d \\leftarrow d + \\beta\\,(I - d)$, **in jedem Tag** - auch an Nulltagen. Die Rate ist $d\\,z$ und fällt, wenn lange nichts kommt.
5. **Bewertung.** Am Ursprung $t$ wird die Nachfrage der nächsten $h$ Tage (Wiederbeschaffungszeit) prognostiziert: $h$ mal die Rate. Kennzahlen: MAE, RMSE, Verzerrung und der **RMSSE** (skalierter RMSE), dazu ein **Orakel** mit der wahren Rate.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielportfolio laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Portfolio**")
    n_items = st.slider("Zahl der Artikel", *bounds("items_slider"), key="items_slider", step=C.ITEMS_STEP, help="Wie viele Artikel das Portfolio hat; die Kennzahlen mitteln über alle.")
    p_mean = st.slider("Bedarfshäufigkeit (Anteil der Tage mit Nachfrage)", *bounds("p_slider"), key="p_slider", step=C.P_STEP, help="Mittlere Wahrscheinlichkeit, dass ein Artikel an einem Tag nachgefragt wird; die Artikel streuen um diesen Wert (0,15 heißt: im Mittel an jedem siebten Tag).")
    size_mean = st.slider("Mittlere Menge je Bedarfstag", *bounds("size_slider"), key="size_slider", step=C.SIZE_STEP, help="Mittlere Stückzahl an einem Tag mit Nachfrage; die Artikel streuen um diesen Wert.")
    cv = st.slider("Streuung der Mengen (CV)", *bounds("cv_slider"), key="cv_slider", step=C.CV_STEP, help="Variationskoeffizient der Menge an einem Bedarfstag: klein = gleichmäßige Bestellungen, groß = klumpig.")
    drift = st.slider("Instabilität der Häufigkeit", *bounds("drift_slider"), key="drift_slider", step=C.DRIFT_STEP, help="Tägliche Streuung des Logarithmus der Bedarfshäufigkeit (Irrfahrt): 0 = stabil; bei 0,02 kann sich die Häufigkeit im Jahr um etwa die Hälfte ändern.")
    obs = st.slider("Anteil auslaufender Artikel", *bounds("obs_slider"), key="obs_slider", step=C.OBS_STEP, help="Diese Artikel werden um den Beginn des Testjahres herum nicht mehr nachgefragt (die Häufigkeit fällt mit einer Zeitkonstanten von 60 Tagen auf null).")
    st.markdown("**Die Prognose**")
    horizon = st.slider("Wiederbeschaffungszeit (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Geprüft wird die Nachfrage der nächsten so vielen Tage. Bei 1 Tag zählt der einzelne Tag.")
    alpha = st.slider("Glättungsparameter α", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP, help="Für einfache Glättung, Croston, SBA und TSB (bei TSB ist β = α). Die Zeilen 'α gewählt' suchen das α je Artikel selbst aus den Trainingstagen.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das ganze Portfolio fest.")
    st.button("🎲 Neues Portfolio generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"items_slider": int(n_items), "p_slider": round(float(p_mean), 2), "size_slider": round(float(size_mean), 2), "cv_slider": round(float(cv), 2), "drift_slider": round(float(drift), 2), "obs_slider": round(float(obs), 2),
                   "horizon_slider": int(horizon), "alpha_slider": round(float(alpha), 2), "seed_input": int(seed)})

settings = Settings(int(n_items), round(float(p_mean), 2), round(float(size_mean), 2), round(float(cv), 2), round(float(drift), 2), round(float(obs), 2), int(horizon), round(float(alpha), 2), int(seed))
a = analyse(settings)
port = a.port
sm = a.summary
h = settings.horizon

# --- Das Portfolio und ein Artikel ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das Portfolio und ein Artikel")
zero_share = float(np.mean(port.y[:, C.FIRST_TEST:] == 0))
adi, cv2 = item_stats(port.y)
finite = adi[np.isfinite(adi)]
st.session_state["item_slider"] = min(port.n - 1, max(0, st.session_state.get("item_slider", 0)))
item = int(st.slider("Artikel", 0, port.n - 1, key="item_slider", help="Welcher Artikel des Portfolios in der Abbildung gezeigt wird."))
st.plotly_chart(build_item(a, item), width="stretch", key="item_chart")
st.caption(
    f"Im Testjahr haben {pct(zero_share)} aller Artikel-Tage keine Nachfrage; im Mittel liegen {de(float(np.median(finite)), 1)} Tage zwischen zwei Bedarfen (Median des ADI über die Artikel). Artikel {item}: Bedarfshäufigkeit {de(port.p[item], 2)}, mittlere Menge {de(port.m[item], 1)} Stück, "
    f"ADI {de(adi[item], 1)} Tage, CV² der Mengen {de(cv2[item], 2)}" + (f"; läuft ab Tag {int(port.t_dead[item])} aus" if port.obsolete[item] else "") + ". Die Balken sind die Nachfrage je Tag (linke Achse), die Linien die Raten der Verfahren "
    f"(rechte Achse; Croston und SBA bleiben zwischen zwei Bedarfen konstant, einfache Glättung und TSB sinken an Nulltagen)."
)

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Auswertung über das Portfolio")
best, best_mae = a.best, a.best_mae
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Bester RMSSE: {SHORT[best]}", de(sm[best]["rmsse"], 3), help="Skalierter RMSE der Nachfrage in der Wiederbeschaffungszeit; kleiner ist besser.")
c2.metric(f"Einfache Glättung (α = {de(settings.alpha, 2)})", de(sm["ses"]["rmsse"], 3), help="RMSSE der einfachen Glättung.")
c3.metric(f"SBA (α = {de(settings.alpha, 2)})", de(sm["sba"]["rmsse"], 3), help="RMSSE von SBA.")
c4.metric("Orakel (wahre Rate)", de(sm["oracle"]["rmsse"], 3), help="RMSSE der Prognose 'wahre Erwartung': das Rauschen der Nachfrage. Kein Verfahren liegt im Mittel darunter.")
c5.metric(f"Bester MAE: {SHORT[best_mae]}", de(sm[best_mae]["mae"], 2) + " Stück", help="Kleinster mittlerer absoluter Fehler; bei sporadischer Nachfrage oft das Verfahren, das am wenigsten Nachfrage vorhersagt.")
st.plotly_chart(build_bars(a), width="stretch", key="bars_chart")
rows = [{"Verfahren": ("Orakel (wahre Rate)" if m == "oracle" else (f"{C.METHOD_NAMES[m].replace('(α)', f'(α = {de(settings.alpha, 2)})').replace('(α = β)', f'(α = β = {de(settings.alpha, 2)})')}" + (f", im Mittel α = {de(float(np.mean(a.chosen[m])), 3)}" if m in a.chosen else ""))),
         "RMSSE": de(sm[m]["rmsse"], 3), "RMSE (Stück)": de(sm[m]["rmse"], 2), "MAE (Stück)": de(sm[m]["mae"], 2), "Verzerrung": f"{'+' if sm[m]['me_pct'] > 0 else ''}{de(sm[m]['me_pct'], 1)} %"} for m in sorted(sm, key=lambda m: sm[m]["rmsse"])]
st.dataframe(rows, hide_index=True)
gap_fixed = (sm["ses"]["rmsse"] - sm["sba"]["rmsse"]) / sm["ses"]["rmsse"]
gap_opt = (sm["ses_opt"]["rmsse"] - sm["sba_opt"]["rmsse"]) / sm["ses_opt"]["rmsse"]
if sm["zero"]["mae"] <= min(sm[m]["mae"] for m in sm if m not in ("zero", "oracle")):
    st.warning(f"⚠️ Nach dem **MAE** gewinnt hier die Null-Prognose ({de(sm['zero']['mae'], 2)} Stück): der MAE belohnt, was den Median trifft, und der ist bei sporadischer Nachfrage 0. Ihr RMSSE ({de(sm['zero']['rmsse'], 2)}) und ihre Verzerrung (−100 %) zeigen, dass sie nutzlos ist. Wer sporadische Nachfrage nach dem MAE bewertet, wählt das falsche Verfahren.")
else:
    st.info(f"Beste Verfahren nach RMSSE: {SHORT[best]} ({de(sm[best]['rmsse'], 3)}), Orakel {de(sm['oracle']['rmsse'], 3)}. Bei α = {de(settings.alpha, 2)} liegt SBA {pct(gap_fixed, 1)} {'vor' if gap_fixed >= 0 else 'hinter'} der einfachen Glättung; mit je Artikel gewähltem α "
            f"sind es {pct(abs(gap_opt), 1)} {'vor' if gap_opt >= 0 else 'hinter'} ihr. Der Vorsprung der Croston-Familie hängt stark daran, mit welchem α die Glättung verglichen wird (Experiment unten).")
st.caption(f"{len(a.origins)} Ursprünge im Testjahr, Wiederbeschaffungszeit {h} Tage, {port.n} Artikel. RMSSE: mittlerer quadratischer Fehler je Artikel geteilt durch das Quadrat des Fehlers der Prognose 'die letzte Wiederbeschaffungszeit wiederholt sich' auf den Trainingstagen, gemittelt über die Artikel, Wurzel. "
           "Verzerrung: mittlerer Fehler in Prozent der mittleren Nachfrage.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Woran misst man sporadische Nachfrage?")
st.caption(f"Standardportfolio; Wiederbeschaffungszeit {', '.join(str(x) for x in C.METRIC_HORIZONS)} Tage; MAE und RMSE jedes Verfahrens im Verhältnis zum Orakel. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Kennzahlen durchrechnen", key="metric_start"):
    st.session_state["metric_on"] = True
if st.session_state.get("metric_on"):
    r = _metric(C.METRIC_HORIZONS, C.EXP_SEEDS)
    st.plotly_chart(build_metric(r), width="stretch", key="metric_chart")
    r1 = r[0]["methods"]
    rl = r[-1]["methods"]
    st.warning(
        f"**Befund:** Bei einer Wiederbeschaffungszeit von {r[0]['horizon']} Tag hat die **Null-Prognose** einen MAE von {de(r1['zero']['mae'], 2)} Stück - kleiner als der des Orakels ({de(r1['oracle']['mae'], 2)}), das die wahre Rate kennt - und ihr RMSE ist {de(r1['zero']['rmse'], 2)} gegen {de(r1['oracle']['rmse'], 2)}. "
        f"Der MAE prämiert sie, weil der Median eines Tages fast immer 0 ist. Bei {r[-1]['horizon']} Tagen verliert sie auch nach MAE ({de(rl['zero']['mae'], 1)} gegen {de(rl['oracle']['mae'], 1)}). Die Verzerrung der Null-Prognose ist immer −100 %. Sporadische Nachfrage gehört nach dem **RMSE** (oder besser: nach den Kosten des Bestands) bewertet. "
        f"Die naive Prognose (letzter Tag) ist dabei das schlechteste Verfahren: RMSE {de(r[1]['methods']['naive']['rmse'], 1)} gegen {de(r[1]['methods']['oracle']['rmse'], 1)} beim Orakel ({r[1]['horizon']} Tage)."
    )

st.markdown("---")

st.subheader("🔬 Wie verzerrt ist Croston?")
st.caption(f"Stabile Nachfrage (Drift 0); Bedarfshäufigkeit {', '.join(de(x, 2) for x in C.BIAS_P)}, α {', '.join(de(x, 2) for x in C.BIAS_ALPHAS)}; Verzerrung = mittlerer Fehler in Prozent der mittleren Nachfrage. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer einige Sekunden.")
if st.button("Verzerrung durchrechnen", key="bias_start"):
    st.session_state["bias_on"] = True
if st.session_state.get("bias_on"):
    rb = _bias(C.BIAS_ALPHAS, C.BIAS_P, C.EXP_SEEDS)
    st.plotly_chart(build_bias(rb), width="stretch", key="bias_chart")
    worst = max(rb, key=lambda x: x["croston"])
    worst_sba = min(rb, key=lambda x: x["sba"])
    st.warning(
        f"**Befund:** Croston **überschätzt** die Nachfrage systematisch, und zwar umso stärker, je größer α und je seltener der Bedarf: bis zu {de(worst['croston'], 1)} % bei Häufigkeit {de(worst['p_mean'], 2)} und α = {de(worst['alpha'], 2)}. SBA gleicht das aus, überkorrigiert aber bei häufigem Bedarf: "
        f"bis {de(worst_sba['sba'], 1)} % bei Häufigkeit {de(worst_sba['p_mean'], 2)} und α = {de(worst_sba['alpha'], 2)}. Einfache Glättung und TSB sind praktisch unverzerrt (unter ±{de(max(max(abs(x['ses']), abs(x['tsb'])) for x in rb), 1)} %)."
    )

st.markdown("---")

st.subheader("🔬 Croston gegen einfache Glättung - bei welchem α?")
st.caption(f"Standardportfolio (mit Drift); RMSSE bei festem α {', '.join(de(x, 2) for x in C.ALPHA_LEVELS)} und mit je Artikel gewähltem α (gestrichelt). Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer einige Sekunden.")
if st.button("α durchrechnen", key="alpha_start"):
    st.session_state["alpha_on"] = True
if st.session_state.get("alpha_on"):
    ra = _alpha(C.ALPHA_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_alpha(ra), width="stretch", key="alpha_chart")
    mid = C.ALPHA_LEVELS[len(C.ALPHA_LEVELS) // 2 + 1] if len(C.ALPHA_LEVELS) > 3 else C.ALPHA_LEVELS[-1]
    f = ra["fixed"]
    so, bo, to = ra["opt"]["ses_opt"][0], ra["opt"]["sba_opt"][0], ra["opt"]["tsb_opt"][0]
    st.warning(
        f"**Befund:** Bei festem α = {de(mid, 2)} liegt SBA bei {de(f[('sba', mid)][0], 3)}, die einfache Glättung bei {de(f[('ses', mid)][0], 3)} - ein Vorsprung von {pct((f[('ses', mid)][0] - f[('sba', mid)][0]) / f[('ses', mid)][0], 0)}. Wählt man aber das α je Artikel aus den Trainingstagen, "
        f"erreichen einfache Glättung ({de(so, 3)}), SBA ({de(bo, 3)}) und TSB ({de(to, 3)}) praktisch dasselbe; das gewählte α ist dabei klein (einfache Glättung im Mittel {de(ra['chosen']['ses_opt'], 3)}, SBA {de(ra['chosen']['sba_opt'], 3)}). Das Mittel der Historie erreicht {de(ra['ref']['mean'], 3)}, das Orakel {de(ra['ref']['oracle'], 3)}. "
        "Der Vorsprung der Croston-Familie entsteht also großteils dadurch, dass die einfache Glättung mit einem zu großen α läuft; mit je Artikel gewähltem α gleicht er sich fast völlig an."
    )

st.markdown("---")

st.subheader("🔬 Was passiert, wenn ein Artikel ausläuft?")
st.caption(f"Portfolio mit {pct(C.OBS_EXP_SHARE)} auslaufenden Artikeln (stabile Nachfrage sonst), α = 0,10; RMSE (Stück) getrennt für auslaufende und aktive Artikel. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Auslaufen durchrechnen", key="obs_start"):
    st.session_state["obs_on"] = True
if st.session_state.get("obs_on"):
    ro = _obs(C.EXP_SEEDS)
    st.plotly_chart(build_obsolescence(ro), width="stretch", key="obs_chart")
    st.warning(
        f"**Befund:** Croston und SBA sind blind für das Auslaufen: ihre Rate bleibt bis zum nächsten Bedarf stehen; in den letzten 100 Ursprüngen prognostizieren sie für auslaufende Artikel das {de(ro['ratio']['croston'], 0)}-Fache (Croston) und das {de(ro['ratio']['sba'], 0)}-Fache (SBA) der wahren Nachfrage, "
        f"die einfache Glättung das {de(ro['ratio']['ses'], 1)}-Fache und TSB das {de(ro['ratio']['tsb'], 1)}-Fache. Der RMSE für auslaufende Artikel: Croston {de(ro['dead']['croston'], 1)}, SBA {de(ro['dead']['sba'], 1)}, einfache Glättung {de(ro['dead']['ses'], 1)}, TSB {de(ro['dead']['tsb'], 1)} Stück. "
        f"Dafür ist Croston bei aktiven, stabilen Artikeln besser ({de(ro['live']['croston'], 1)} gegen {de(ro['live']['ses'], 1)} bei einfacher Glättung und {de(ro['live']['tsb'], 1)} bei TSB). Das Mittel der Historie ist bei stabilen Artikeln am besten ({de(ro['live']['mean'], 1)}), bei auslaufenden am schlechtesten ({de(ro['dead']['mean'], 1)}). "
        "TSB fällt wie die einfache Glättung mit dem Bedarf und ist bei aktiven Artikeln besser als sie, aber schlechter als Croston."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Rate zählt** | Alle Verfahren liefern eine einzige Zahl je Tag; die Verteilung der Nachfrage in der Wiederbeschaffungszeit (für Sicherheitsbestände) kennen sie nicht. | Prognoseintervalle (Stück 7), Bestandsrechnung (Stück 10) |
| **Kein Kalender, keine Aktionen** | Ein Artikel mit Aktionen oder Saison bekommt hier eine glatte Rate; das Muster bleibt im Fehler. | Dynamische Regression (Stück 4), Boosting (Stück 6) |
| **Jeder Artikel für sich** | Wenige Bedarfe je Artikel machen jede Schätzung unsicher; ähnliche Artikel teilen ihr Wissen nicht. | Hierarchische Abstimmung (Stück 8), globale Modelle |
| **Die Kennzahl passt zum Zweck** | Der MAE prämiert die Null-Prognose; der RMSE misst die Prognose der Rate, nicht die Kosten von Fehlbeständen und Überbeständen. | Bestandskosten (Stück 10) |
| **Feste Parameter aus den Trainingstagen** | α und Anfangswerte stammen aus zwei Jahren und werden nicht nachgeführt. | – |
| **Erzeugte Portfolios, sechs Seeds** | Das Vehikel erzeugt genau die Muster (Bernoulli-Bedarf, log-normale Mengen, Drift, Auslaufen); echte Artikel sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu **Croston, SBA, TSB**, Boosting, Prognoseintervalle, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (alle Stücke der Linie sind inzwischen gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Vehikel.** Artikel $i$, Tag $t$: Bedarf mit Wahrscheinlichkeit $p_i(t) = p_i\,e^{W_i(t)}\,\delta_i(t)$ ($W$ eine Irrfahrt mit Schritt $\sigma_d$, $\delta$ das Auslaufen: $e^{-(t - t_0)/60}$ nach dem Tag $t_0$ bei auslaufenden Artikeln), Menge $\max(1, \mathrm{round}(m_i\,e^{\sigma z - \sigma^2/2}))$ mit $\sigma^2 = \ln(1 + \mathrm{CV}^2)$.
Erwartung je Tag: $\mu_i(t) \approx p_i(t)\,m_i$.

**Verfahren** (Rate $\hat r_t$ aus den Tagen $< t$): einfache Glättung $\ell_t = \ell_{t-1} + \alpha\,(y_{t-1} - \ell_{t-1})$. Croston: an Bedarfstagen $z \leftarrow z + \alpha\,(y - z)$ und $p \leftarrow p + \alpha\,(q - p)$ ($q$ = Tage seit dem letzten Bedarf einschließlich dieses Tages), sonst unverändert; $\hat r = z/p$.
SBA: $\hat r = (1 - \alpha/2)\,z/p$. TSB: in jedem Tag $d \leftarrow d + \beta\,(\mathbb 1[y > 0] - d)$, an Bedarfstagen $z \leftarrow z + \alpha\,(y - z)$; $\hat r = d\,z$. Anfangswerte aus den ersten 180 Tagen ($z$ = mittlere Menge, $p$ = mittlerer Abstand, $d = 1/p$, $\ell$ = Tagesmittel).

**Prognose der Wiederbeschaffungszeit** $h$: $\hat S_t = h\,\hat r_t$ gegen $S_t = \sum_{j=0}^{h-1} y_{t+j}$. **Kennzahlen:** $\mathrm{MAE}$, $\mathrm{RMSE}$, Verzerrung $= \overline{\hat S - S} / \bar S$ und
$\mathrm{RMSSE} = \sqrt{\tfrac1n \sum_i \mathrm{MSE}_i / s_i^2}$ mit der Skala $s_i^2$ = mittlerer quadrierter Fehler der Prognose "die vorige Wiederbeschaffungszeit wiederholt sich" auf den Trainingstagen (mindestens 1 Stück). **Orakel:** $\hat S_t = \sum_j \mu(t+j)$.
**Je Artikel gewähltes α:** aus $\{0{,}01, 0{,}02, 0{,}05, 0{,}1, 0{,}2, 0{,}3\}$ das mit dem kleinsten mittleren quadrierten Ein-Schritt-Fehler (Rate gegen Tagesnachfrage) auf den Trainingstagen 180 bis 729.

Implementiert in `cr_methods.py` (Verfahren), `cr_scenario.py` (das Portfolio), `cr_evaluation.py` (Analyse, Kennzahlen, vier Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Zeitreihen-Prognose: von Naiv bis Vortraining](https://sebastianhanisch.net/konzepte-zeitreihen-prognose.html)."
)
