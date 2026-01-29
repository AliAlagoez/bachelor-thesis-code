import pandas as pd


def load_price_csv(path: str) -> pd.DataFrame:
    """
    Lädt Preis-CSV und gibt DataFrame mit Index 'date' und Spalte 'close' zurück.

    Unterstützt:
    - CoinMarketCap-Export mit 'timeClose' + 'close'
    - Klassische CSV mit 'Date'/'date' + 'Close'/'close'
    """

    df = pd.read_csv(path, sep=None, engine="python")

    # --- Datumsspalte finden ---
    if "timeClose" in df.columns:
        date_col = "timeClose"
    elif "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    else:
        raise ValueError(f"Keine Datumsspalte gefunden in {path}. Erwartet z.B. 'timeClose' oder 'Date'.")

    # --- Close-Spalte finden ---
    if "close" in df.columns:
        close_col = "close"
    elif "Close" in df.columns:
        close_col = "Close"
    elif "Adj Close" in df.columns:
        close_col = "Adj Close"
    else:
        raise ValueError(f"Keine Close-Spalte gefunden in {path}. Erwartet z.B. 'close' oder 'Close'.")

    out = df[[date_col, close_col]].copy()

    # Datum parsen (bei CoinMarketCap ist es ISO-Zeitstempel)
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Close numeric machen
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")

    # Aufräumen
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    out = out.sort_values("date")

    # Doppelte Tage: letzten Wert nehmen (falls vorhanden)
    out = out.groupby("date", as_index=False)["close"].last()

    out = out.set_index("date")

    return out
