"""Plotly-Abbildungen der Croston-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cr_constants as C

COLORS = {"naive": "#9e9e9e", "zero": "#212121", "mean": "#795548", "ses": "#4c78a8", "croston": "#f58518", "sba": "#e45756", "tsb": "#54a24b", "ses_opt": "#9ecae9", "sba_opt": "#f7a8a6", "tsb_opt": "#a1d99b", "oracle": "#00897b"}
SHORT = {"naive": "Naiv", "zero": "Null", "mean": "Mittel", "ses": "SES", "croston": "Croston", "sba": "SBA", "tsb": "TSB", "ses_opt": "SES (α gewählt)", "sba_opt": "SBA (α gewählt)", "tsb_opt": "TSB (α gewählt)", "oracle": "Orakel"}
ACTUAL = "#14233B"
ORACLE = "#00897b"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def build_item(a, item, first=560):
    """Ein Artikel: Nachfrage je Tag (Balken, linke Achse) und die Raten der Verfahren (Linien, rechte Achse; Rate = erwartete Nachfrage je Tag) ab Tag first."""
    port = a.port
    t = np.arange(first, port.y.shape[1])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=t, y=port.y[item, first:], name="Nachfrage je Tag", marker=dict(color="rgba(20,35,59,0.35)"), showlegend=True), secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=port.mu[item, first:], mode="lines", name="wahre Rate (Orakel)", line=dict(color=ORACLE, width=2, dash="dot")), secondary_y=True)
    for m in ("mean", "ses", "croston", "sba", "tsb"):
        fig.add_trace(go.Scatter(x=t, y=a.rates[m][item, first:port.y.shape[1]], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=2)), secondary_y=True)
    fig.add_vline(x=C.FIRST_TEST - 0.5, line=dict(color="#f58518", dash="dash"), annotation_text="Testjahr beginnt", annotation_position="top left")
    if port.obsolete[item]:
        fig.add_vline(x=int(port.t_dead[item]), line=dict(color="#e45756", dash="dot"), annotation_text="läuft aus", annotation_position="bottom right")
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Nachfrage je Tag (Stück)", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Rate (Stück je Tag)", rangemode="tozero", secondary_y=True)
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.3))


def build_bars(a):
    """RMSSE je Verfahren (skalierter Fehler der Nachfrage in der Wiederbeschaffungszeit), dazu das Orakel."""
    ms = sorted((m for m in a.summary if m != "oracle"), key=lambda m: a.summary[m]["rmsse"])
    vals = [a.summary[m]["rmsse"] for m in ms]
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in ms], y=vals, marker=dict(color=[COLORS[m] for m in ms]), text=[de(v, 2) for v in vals], textposition="outside", showlegend=False))
    fig.add_hline(y=a.summary["oracle"]["rmsse"], line=dict(color=ORACLE, dash="dot"), annotation_text="Orakel (wahre Rate)", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="RMSSE 1 = 'die letzte Wiederbeschaffungszeit wiederholt sich'", annotation_position="bottom right")
    fig.update_yaxes(title_text="RMSSE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_metric(rows):
    """Für jede Wiederbeschaffungszeit: MAE und RMSE jedes Verfahrens im Verhältnis zum Orakel (1 = so gut wie die wahre Rate)."""
    methods = [m for m in rows[0]["methods"] if m != "oracle"]
    fig = make_subplots(rows=1, cols=len(rows), shared_yaxes=False, subplot_titles=[f"Wiederbeschaffungszeit {r['horizon']} Tage" for r in rows])
    for i, r in enumerate(rows, start=1):
        o = r["methods"]["oracle"]
        mae = [min(r["methods"][m]["mae"] / o["mae"], 4.0) for m in methods]
        rmse = [min(r["methods"][m]["rmse"] / o["rmse"], 4.0) for m in methods]
        fig.add_trace(go.Bar(x=[SHORT[m] for m in methods], y=mae, name="MAE / Orakel", marker=dict(color="#f7a8a6"), showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Bar(x=[SHORT[m] for m in methods], y=rmse, name="RMSE / Orakel", marker=dict(color="#4c78a8"), showlegend=(i == 1)), row=1, col=i)
        fig.add_hline(y=1.0, line=dict(color=ORACLE, dash="dot"), row=1, col=i)
    fig.update_yaxes(title_text="Verhältnis zum Orakel (auf 4 gekappt)", rangemode="tozero", col=1)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-60)
    return _base(fig, 440).update_layout(legend=dict(orientation="h", y=-0.5))


def build_bias(rows):
    ps = sorted({r["p_mean"] for r in rows})
    fig = make_subplots(rows=1, cols=len(ps), shared_yaxes=True, subplot_titles=[f"Bedarfshäufigkeit {de(p, 2)}" for p in ps])
    for i, p in enumerate(ps, start=1):
        sub = [r for r in rows if r["p_mean"] == p]
        for m in ("ses", "croston", "sba", "tsb"):
            fig.add_trace(go.Scatter(x=[r["alpha"] for r in sub], y=[r[m] for r in sub], mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2), showlegend=(i == 1)), row=1, col=i)
        fig.add_hline(y=0.0, line=dict(color="#7f7f7f", dash="dot"), row=1, col=i)
        fig.update_xaxes(title_text="α", type="category", row=1, col=i)
    fig.update_yaxes(title_text="Verzerrung (% der mittleren Nachfrage)", col=1)
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_alpha(res):
    als = list(res["alphas"])
    fig = go.Figure()
    for m in ("ses", "croston", "sba", "tsb"):
        fig.add_trace(go.Scatter(x=[str(a) for a in als], y=[res["fixed"][(m, a)][0] for a in als], error_y=dict(type="data", array=[res["fixed"][(m, a)][1] for a in als]), mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2)))
    for m in ("ses_opt", "sba_opt", "tsb_opt"):
        fig.add_hline(y=res["opt"][m][0], line=dict(color=COLORS[m], dash="dash"), annotation_text=SHORT[m], annotation_position="top right" if m != "sba_opt" else "bottom right")
    fig.add_hline(y=res["ref"]["mean"], line=dict(color=COLORS["mean"], dash="dot"), annotation_text="Mittel der Historie", annotation_position="top left")
    fig.add_hline(y=res["ref"]["oracle"], line=dict(color=ORACLE, dash="dot"), annotation_text="Orakel", annotation_position="bottom left")
    fig.update_xaxes(title_text="Glättungsparameter α", type="category")
    fig.update_yaxes(title_text="RMSSE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 400).update_layout(legend=dict(orientation="h", y=-0.3))


def build_obsolescence(res):
    methods = ("mean", "ses", "croston", "sba", "tsb")
    labels = [SHORT[m] for m in methods]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"auslaufende Artikel ({res['n_dead']:.0f} im Portfolio)", "aktive Artikel"))
    fig.add_trace(go.Bar(x=labels, y=[res["dead"][m] for m in methods], marker=dict(color=[COLORS[m] for m in methods]), text=[de(res["dead"][m], 1) for m in methods], textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[res["live"][m] for m in methods], marker=dict(color=[COLORS[m] for m in methods]), text=[de(res["live"][m], 1) for m in methods], textposition="outside", showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="RMSE der Nachfrage in der Wiederbeschaffungszeit (Stück)", rangemode="tozero", col=1)
    fig.update_yaxes(rangemode="tozero", col=2)
    return _base(fig, 380)
