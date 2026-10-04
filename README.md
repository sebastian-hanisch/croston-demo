# 🐌 Croston, SBA und TSB – Prognosen bei sporadischer Nachfrage

**[→ Demo live ausprobieren](https://sebastianhanisch-croston-demo.streamlit.app/)**

Fünftes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Der Ast für **sporadische Nachfrage**: Nachfolger der [Exponentiellen Glättung](https://github.com/sebastian-hanisch/exponential-smoothing-demo), deren Schwäche – viele Nullen – er aufgreift.
Die sechs weiteren Stücke der Linie (Boosting, Prognoseintervalle, Hierarchische Abstimmung, Kombination, Prognose → Bestand, ein vortrainiertes Netz) sind inzwischen gebaut.

Ein Ersatzteil, ein Spezialartikel, ein Nachzügler im Sortiment: **an den meisten Tagen wird nichts bestellt**, und wenn, dann in Klumpen. Die Demo zeigt an einem **Portfolio von Lagerartikeln eines Depots** (ein neues Vehikel: bisher ging es um tägliche Aufträge), was Prognoseverfahren dafür leisten: die einfache Glättung, **Croston** (Menge und Abstand getrennt glätten), **SBA** (Crostons Verzerrung korrigiert)
und **TSB** (die Bedarfswahrscheinlichkeit in jedem Tag fortschreiben). Geprüft wird die **Nachfrage in der Wiederbeschaffungszeit** – die Größe, aus der Bestellpunkte und Sicherheitsbestände entstehen –, gegen ein **Orakel** mit der wahren Rate. Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben.

**Bezug zu OR:** die Prognose der Nachfrage in der Wiederbeschaffungszeit ist der Eingang jeder Bestandsrechnung; das Bestands-Stück der Linie nimmt sie auf.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Croston verzerrt, SBA korrigiert; die Familie schlägt die Glättung". Die Messung bestätigt den ersten Teil und relativiert den zweiten deutlich:

1. **Die erste Frage ist die Kennzahl, nicht das Verfahren.** Bei einer Wiederbeschaffungszeit von einem Tag hat die **Null-Prognose** den kleinsten MAE (1,11 Stück) – kleiner als das Orakel, das die wahre Rate kennt (1,46). Der MAE belohnt, was den Median trifft, und der ist bei sporadischer Nachfrage 0. Ihr RMSE (3,08 gegen 2,63) und ihre Verzerrung (−100 %) zeigen, dass sie nutzlos ist.
2. **Crostons Verzerrung ist real.** Bis zu +17 % Überschätzung (Bedarfshäufigkeit 0,03, α = 0,3); SBA gleicht aus, überkorrigiert aber bei häufigem Bedarf (−7 % bei Häufigkeit 0,3, α = 0,3); die einfache Glättung und TSB sind unverzerrt.
3. **Der Vorsprung der Familie hängt am α der Glättung.** Bei festem α = 0,2 liegt SBA 19 % vor der einfachen Glättung (RMSSE 0,786 gegen 0,966). Wählt man das α je Artikel aus den Trainingstagen, erreichen einfache Glättung (0,750), SBA (0,756) und TSB (0,747) praktisch dasselbe: das gewählte α der einfachen Glättung ist winzig (im Mittel 0,012). Das war nicht geplant: der Vorsprung entsteht großteils dadurch, dass die einfache Glättung mit einem zu großen α läuft.
4. **Croston und SBA sind blind für das Auslaufen.** Ihre Rate bleibt bis zum nächsten Bedarf stehen: bei auslaufenden Artikeln prognostizieren sie in den letzten 100 Ursprüngen das 53- (Croston) und 51-Fache (SBA) der wahren Nachfrage, die einfache Glättung das 1,1-Fache und TSB das 1,1-Fache. Bei aktiven, stabilen Artikeln ist Croston dafür besser (7,1 gegen 7,9 bei der einfachen Glättung).
5. **Naiv ist katastrophal.** Der letzte Tag als Prognose der nächsten sieben: RMSE 19,6 Stück gegen 7,0 beim Orakel.

## Modell

- **Das Portfolio** (`cr_scenario.py`): 1 095 Tage je Artikel (Ursprünge im letzten Jahr, ab Tag 730). Bedarf mit der Wahrscheinlichkeit $p_i(t) = p_i\,e^{W_i(t)}\,\delta_i(t)$ ($W$ eine Irrfahrt mit Schritt = Drift, $\delta$ das Auslaufen: $e^{-(t-t_0)/60}$ nach einem Tag $t_0$ um den Beginn des Testjahres); die Menge ist ganzzahlig, mindestens 1 und log-normal (Mittel $m_i$, Variationskoeffizient CV).
  Die Artikel streuen um die Regler-Mittelwerte (Bedarfshäufigkeit, Menge). Der Erwartungswert je Tag ist $p_i(t)\,m_i$ (Orakel).
- **Verfahren** (`cr_methods.py`, vektorisiert über alle Artikel): naiv, Null, Mittel der Historie, einfache Glättung $\ell \leftarrow \ell + \alpha(y - \ell)$, **Croston** ($z \leftarrow z + \alpha(y - z)$ und $p \leftarrow p + \alpha(q - p)$ nur an Bedarfstagen, Rate $z/p$), **SBA** (mal $1 - \alpha/2$), **TSB** ($d \leftarrow d + \beta(I - d)$ in jedem Tag, Rate $d\,z$; hier $\beta = \alpha$).
  Anfangswerte aus den ersten 180 Tagen. Dazu je Verfahren das **je Artikel gewählte α**: aus $\{0{,}01, 0{,}02, 0{,}05, 0{,}1, 0{,}2, 0{,}3\}$ das mit dem kleinsten Ein-Schritt-Fehler (Rate gegen Tagesnachfrage) auf den Trainingstagen.
- **Kennzahlen** (`cr_evaluation.py`): Nachfrage in der Wiederbeschaffungszeit $h$ (Summe der nächsten $h$ Tage) gegen $h$ mal die Rate; MAE und RMSE in Stück, Verzerrung (mittlerer Fehler in Prozent der mittleren Nachfrage) und der **RMSSE**: mittlerer quadrierter Fehler je Artikel geteilt durch das Quadrat des Fehlers der Prognose "die vorige Wiederbeschaffungszeit wiederholt sich" auf den Trainingstagen (mindestens 1 Stück), gemittelt über die Artikel, Wurzel.
  **Orakel:** die Summe der wahren Erwartungswerte.

## Methodik

- **Handrechnungen:** die Anfangswerte, einfache Glättung, Croston (mit den Abständen 2 und 3), SBA, TSB (auch mit eigenem β), naiv, Null und Mittel; die Kennzahlen (MAE, RMSE, Verzerrung, RMSSE) auf einer kleinen Fehlermatrix; Summen der Wiederbeschaffungszeit und die Skala; ADI und CV² auf einem Artikel.
- **Gegenprobe:** Croston, SBA, TSB und einfache Glättung des ganzen Portfolios gegen eine **unabhängige Schleife je Artikel**; die einfache Glättung außerdem gegen `statsmodels.SimpleExpSmoothing` (festes α, bekanntes Anfangsniveau) auf 1e-9.
- **Eigenschaften:** Croston bleibt zwischen zwei Bedarfen exakt stehen, TSB fällt an jedem Nulltag; SBA ist Croston mal $1 - \alpha/2$; die Raten nutzen nur die Vergangenheit; das gewählte α ist das mit dem kleinsten Trainingsfehler.
- **Das Portfolio:** die Häufigkeit der Bedarfstage folgt der Ausgangshäufigkeit, die mittlere Menge dem Regler; ohne Drift bleibt der Erwartungswert konstant, mit Drift streut er; auslaufende Artikel haben am Ende praktisch keine Nachfrage mehr; das Orakel ist nach allen Kennzahlen das beste Verfahren.
- **Statistik:** sechs feste Seeds mit je 200 Artikeln (Fehlerbalken = Standardfehler). Die Verfahren sind deterministisch (kein Training außer der Wahl von α), die Zahlen daher exakt reproduzierbar.
- **Literatur** (nicht nachgebaut): Croston 1972; Syntetos/Boylan 2005 (SBA); Teunter/Syntetos/Babai 2011 (TSB); Hyndman/Koehler 2006 (skalierte Fehler).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Woran misst man?** (Wiederbeschaffungszeit 1 / 7 / 28 Tage, 6 Seeds) | Ein Tag: **Null-Prognose** MAE 1,11 Stück gegen 1,46 beim Orakel (RMSE 3,08 gegen 2,63, Verzerrung −100 %). Sieben Tage: naiv RMSE 19,6 gegen 7,0 beim Orakel (das schlechteste Verfahren). 28 Tage: Null MAE 31,1 gegen 9,8 – dann verliert sie auch nach MAE. | `test_metric_experiment` |
| **Verzerrung** (Häufigkeit 0,03 / 0,1 / 0,3; α 0,05 bis 0,3; stabile Nachfrage) | Croston überschätzt: bis **+17,4 %** (Häufigkeit 0,03, α = 0,3), bei α ≥ 0,1 immer; SBA bis **−7,2 %** (Häufigkeit 0,3, α = 0,3); einfache Glättung und TSB unter ±1 %. | `test_bias_experiment` |
| **Croston gegen einfache Glättung** (RMSSE, α 0,02 / 0,05 / 0,1 / 0,2 / 0,3) | Einfache Glättung 0,756 / 0,789 / 0,849 / **0,966** / 1,08; SBA bei α = 0,2: **0,786** (19 % besser); Croston bei α = 0,05: 0,750. **Mit je Artikel gewähltem α:** einfache Glättung 0,750, SBA 0,756, TSB 0,747 (gewähltes α im Mittel 0,012 / 0,050 / 0,013). Mittel der Historie 0,793, Orakel 0,725. | `test_alpha_experiment` |
| **Auslaufende Artikel** (30 %, α = 0,1, stabil sonst) | RMSE (Stück) für auslaufende Artikel: TSB **2,9**, einfache Glättung 3,0, SBA 3,9, Croston **4,1**, Mittel 7,0; für aktive Artikel: Mittel 6,8, SBA 7,05, Croston 7,1, TSB 7,7, einfache Glättung 7,9. Prognose für auslaufende Artikel im Verhältnis zur wahren Nachfrage (letzte 100 Ursprünge): Croston 53, SBA 51, einfache Glättung 1,1, TSB 1,1, Mittel 98. | `test_obsolescence_experiment` |
| Standardfall (Preset, Seed 3, α = 0,1, 7 Tage) | RMSSE: TSB (α gewählt) **0,756**, SBA 0,770, Croston 0,773, Mittel 0,791, einfache Glättung 0,856, Null 1,24, naiv 2,04; Orakel 0,733. Verzerrung: Croston +2,8 %, SBA −2,3 %. 78 % Nulltage im Testjahr. | `test_standard_preset` |
| Sehr sporadisch (Preset, p = 0,03) | 95 % Nulltage; Croston und SBA 0,793 vor der einfachen Glättung (0,907); **MAE: Null 1,85 Stück gegen 2,23 beim Orakel.** | `test_sporadic_preset_and_the_mae_trap` |
| Klumpige Mengen (Preset, CV 1,5) | Croston 0,879, SBA 0,875, einfache Glättung 0,982, TSB (α gewählt) 0,852; Orakel 0,839. | `test_lumpy_preset` |
| Auslaufende Artikel (Preset, 30 %, Drift 0,02) | 64 auslaufende Artikel; Verzerrung Croston +16,5 %, SBA +10,7 %, einfache Glättung +0,7 %, TSB +0,5 %; RMSSE TSB (α gewählt) 0,662, einfache Glättung 0,740, SBA 0,688, Croston 0,694. | `test_obsolescence_preset` |
| Grobes α = 0,3 (Preset) | Einfache Glättung 1,085, TSB 1,026, Croston 0,859 (Verzerrung +10,4 %), SBA 0,827; mit je Artikel gewähltem α 0,756 bis 0,764. | `test_coarse_alpha_preset` |
| Wiederbeschaffungszeit 1 Tag (Preset) | Null: MAE 1,21 Stück (Orakel 1,54), RMSSE 0,81 (Orakel 0,717). | `test_one_day_preset` |

Die Preset-Zeilen sind **Einzelportfolios** (Seed 3); belastbar sind die Zeilen über sechs Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Rate zählt** | Alle Verfahren liefern eine einzige Zahl je Tag; die Verteilung der Nachfrage in der Wiederbeschaffungszeit (für Sicherheitsbestände) kennen sie nicht. | Prognoseintervalle, Bestandsrechnung (beide gebaut) |
| **Kein Kalender, keine Aktionen** | Ein Artikel mit Aktionen oder Saison bekommt eine glatte Rate; das Muster bleibt im Fehler. | Dynamische Regression, Boosting |
| **Jeder Artikel für sich** | Wenige Bedarfe je Artikel machen jede Schätzung unsicher; ähnliche Artikel teilen ihr Wissen nicht. | Hierarchische Abstimmung (gebaut), globale Modelle |
| **Die Kennzahl passt zum Zweck** | Der MAE prämiert die Null-Prognose; der RMSE misst die Prognose der Rate, nicht die Kosten von Fehl- und Überbeständen. | Bestandskosten (gebaut) |
| **Feste Parameter aus den Trainingstagen** | α und Anfangswerte stammen aus zwei Jahren und werden nicht nachgeführt. | – |
| **Erzeugte Portfolios, sechs Seeds** | Das Vehikel erzeugt genau die Muster (Bernoulli-Bedarf, log-normale Mengen, Drift, Auslaufen); echte Artikel sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 20 Sekunden, 60 Tests): die Verfahren von Hand und gegen eine unabhängige Schleife (Croston, SBA, TSB, einfache Glättung; statsmodels für die einfache Glättung), ihre Eigenschaften, die Wahl von α, das Portfolio (Häufigkeit, Mengen, Drift, Auslaufen), Kennzahlen, Skala und ADI/CV² von Hand, Auswertung und Experimentzeilen, Preset- und Permalink-Klemmen,
AppTest-Rauchtests (Standard, jedes Preset, Artikel-Regler bei kleinerem Portfolio, Extremwerte, vier Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen; die Verfahren sind deterministisch, die Bänder großzügiger als die Rundung).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `cr_constants.py` | Regler-Grenzen, Verfahren, Experiment-Seeds |
| `cr_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `cr_scenario.py` | Das Portfolio (Bedarf, Mengen, Drift, Auslaufen, Orakel) |
| `cr_methods.py` | Die Verfahren und die Wahl von α |
| `cr_evaluation.py` | Analyse, Kennzahlen, ADI/CV², vier Experimente |
| `cr_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Die Klassifikation nach ADI und CV² (Syntetos-Boylan-Croston: glatt, sporadisch, erratisch, klumpig) und eine Verfahrenswahl je Klasse; ADI und CV² sind berechnet und stehen im Artikel-Titel, eine Wahl nach Klasse ist nicht getestet.
- Kostenbasierte Bewertung (Fehl- und Überbestände) und Verteilungsprognosen; beides gehört zu Prognoseintervallen und Bestand.
- Verfahren mit Einflussgrößen (Aktionen, Kalender) und neuere Familien (Aggregation über die Zeit, ADIDA/IMAPA, Boosting auf Lags).
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Gegenprobe im Test: statsmodels).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Zeitreihen-Prognose: von Naiv bis Vortraining](https://sebastianhanisch.net/konzepte-zeitreihen-prognose.html).
