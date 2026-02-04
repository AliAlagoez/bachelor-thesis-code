import yfinance as yf
import pandas as pd
import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Target directory for downloaded price data
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data directory if it does not exist
os.makedirs(DATA_DIR, exist_ok=True)

# Time period for data collection
START = "2016-01-01"
END = "2024-12-31"

def download_coin(ticker, filename):
    """
    Downloads daily price data for a given cryptocurrency ticker from
    Yahoo Finance and stores the cleaned closing prices as a CSV file.

    Parameters:
    - ticker (str): Yahoo Finance ticker symbol (e.g. 'BTC-USD', 'ETH-USD')
    - filename (str): Name of the output CSV file (e.g. 'BTC.csv')
    """
    print(f"Download {ticker} ...")

    # Download daily price data via yfinance
    df = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    # Abort if no data is returned
    if df.empty:
        raise RuntimeError(f"No data available for {ticker}")

    # Keep only the closing price and move the date index into a column
    out = df[["Close"]].reset_index()

    # Standardize column names for further processing
    out.columns = ["date", "close"]

    # Ensure consistent data types
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    # Remove missing or invalid values
    out = out.dropna()

    # Output file path
    path = os.path.join(DATA_DIR, filename)

    # Save cleaned data as CSV without index
    out.to_csv(path, index=False)

    # Print short summary of the downloaded dataset
    print(f"Saved: {path} | Rows: {len(out)} | {out['date'].min()} → {out['date'].max()}")

def main():
    """
    Executes the download of daily price data for Bitcoin and Ethereum.
    """
    download_coin("BTC-USD", "BTC.csv")
    download_coin("ETH-USD", "ETH.csv")

if __name__ == "__main__":
    main()
