import pandas as pd


def load_price_csv(path: str) -> pd.DataFrame:
    """
    Loads a price CSV file and returns a DataFrame indexed by date
    with a single column containing closing prices.
    """
    # Read CSV
    df = pd.read_csv(path, sep=None, engine="python")

    # Find date column
    if "timeClose" in df.columns:
        date_col = "timeClose"
    elif "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    else:
        raise ValueError(f"No date column found in {path}. Expected e.g. 'timeClose' or 'Date'.")

    # Find the Close column
    if "close" in df.columns:
        close_col = "close"
    elif "Close" in df.columns:
        close_col = "Close"
    elif "Adj Close" in df.columns:
        close_col = "Adj Close"
    else:
        raise ValueError(f"No close price column found in {path}. Expected e.g. 'close' or 'Close'.")

    # Select and rename relevant columns
    out = df[[date_col, close_col]].copy()

    # Parse date column 
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Close to numeric
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")

    # Remove invalid or missing observations
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    out = out.sort_values("date")

    # Handle duplicate dates by keeping the last available observation
    out = out.groupby("date", as_index=False)["close"].last()

    # Set date as index
    out = out.set_index("date")

    return out
