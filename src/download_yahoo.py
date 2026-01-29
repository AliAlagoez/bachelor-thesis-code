import yfinance as yf
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

START = "2016-01-01"
END = "2024-12-31"

def download_coin(ticker, filename):
    print(f"Download {ticker} ...")
    df = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError(f"Keine Daten für {ticker}")

    out = df[["Close"]].reset_index()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna()

    path = os.path.join(DATA_DIR, filename)
    out.to_csv(path, index=False)

    print(f"Gespeichert: {path} | Rows: {len(out)} | {out['date'].min()} → {out['date'].max()}")

def main():
    download_coin("BTC-USD", "BTC.csv")
    download_coin("ETH-USD", "ETH.csv")

if __name__ == "__main__":
    main()
