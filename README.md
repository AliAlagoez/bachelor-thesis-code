# Code zur Bachelorarbeit

**Titel der Arbeit:**  
Robustheit von Prognosemodellen für Kryptowährungen:  
Eine Analyse der Vorhersagequalität in stabilen und volatilen Marktphasen


**Autor:**          Ali Alagöz  
**Studiengang:**    Wirtschaftsinformatik (B.Sc.)  
**Hochschule:**     HTW Berlin  
**Erstprüfer:**     Prof. Dr. Axel Hochstein  
**Zweitprüfer:**    Prof. Dr. Martin Spott  
**Abgabedatum:**    06.02.2026


## Projektübersicht
Dieses Repository enthält den vollständigen Python-Code zur empirischen Analyse der Prognosegüte und Robustheit ausgewählter Prognosemodelle für Kryptowährungen im Rahmen der Bachelorarbeit.

Untersucht werden ein naives Baseline-Modell, ein klassisches ARIMA-Zeitreihenmodell sowie ein datengetriebenes maschinelles Lernverfahren (Gradient Boosting) anhand täglicher Renditedaten von Bitcoin (BTC) und Ethereum (ETH).


### Arten von Daten  
Verwendet werden tägliche Schlusskurse der Kryptowährungen Bitcoin (BTC) und Ethereum (ETH).  
Konkret umfasst der Datensatz historische Preisdaten von Bitcoin (BTC) und Ethereum (ETH).  
Aus den Preisdaten werden logarithmische Renditen berechnet, die als Grundlage für die Prognosemodelle dienen.


### Software  
Es wurden Python-Skripte zur Datenaufbereitung, Modellschätzung und statistischen Auswertung entwickelt.  
Die Software dient der empirischen Untersuchung der Prognosegüte und Robustheit von Prognosemodellen in stabilen und volatilen Marktphasen.


### Sprache  
- Daten:                                sprachneutral (numerische Zeitreihen)  
- Code und Kommentare:                  Englisch  
- Wissenschaftliche Ausarbeitung:       Deutsch  


## Datenursprung

- **Autor der Software:**               Ali Alagöz  
- **Datenquelle:**                      Yahoo Finance

Die verwendeten Preisdaten wurden über Yahoo Finance bezogen und können mit dem im Repository enthaltenen Skript `download_yahoo.py` reproduziert werden.


## Zeitraum

- Datenerhebung:                        01.01.2016 - 30.12.2024 (BTC), 09.11.2017 - 30.12.2024 (ETH)  
- Softwareentwicklung und Auswertung:   12/2025 – 01/2026  


## Datenformate

- Preisdaten:                           BTC.csv, ETH.csv  
- Ergebnisdateien:                      metrics.csv 
- Quellcode:                            Python-Dateien (.py)  


## Qualitätssicherung

- Zeitkonsistente Modellschätzung (kein Look-ahead-Bias)  
- Ausschluss fehlender oder ungültiger Beobachtungen  
- Reproduzierbare Backtests durch feste Modellparameter  
- Vollständige Dokumentation der Analyse über Python-Code im Repository  


## Ablageort

Der vollständige Code zur empirischen Analyse ist in einem GitHub-Repository abgelegt.

- Link:             https://github.com/AliAlagoez/bachelor-thesis-code


## Projektstruktur


thesis-code
data                # Eingabedaten (BTC.csv, ETH.csv)
results             # Ergebnisdateien (MAE, RMSE, Robustheit)
src                 # Implementierung der Modelle und Backtests
    backtest.py     # Walk-forward Backtest & Fehlermaße
    features.py     # Feature Engineering (Lags, rollierende Statistiken)
    load_data.py    # Datenimport und -aufbereitung
    models.py       # Prognosemodelle (Baseline, ARIMA, Gradient Boosting)
    regimes.py      # Volatilitätsbasierte Marktphasen
main.py             # Hauptskript


## Setup

```bash
# Erstellen einer virtuellen Python-Umgebung
python -m venv venv

# Aktivieren der virtuellen Umgebung (Windows / PowerShell)
.\venv\Scripts\Activate.ps1

# Installation aller benötigten Python-Abhängigkeiten
pip install -r requirements.txt

# Herunterladen der historischen Preisdaten von Yahoo Finance
python src/download_yahoo.py

# Start der empirischen Analyse
python main.py
```