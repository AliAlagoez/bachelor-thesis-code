import pandas as pd

def label_volatility_regimes(df, vol_col="vol_30", low_q=0.25, high_q=0.75):
    """
    Labels market regimes based on rolling volatility quantiles.

    Observations in the lower quantile are classified as 'stable',
    while observations in the upper quantile are classified as 'volatile'.
    All remaining observations are labeled as 'normal'.
    """

    df = df.copy()

    # Determine volatility thresholds based on empirical quantiles
    low_thresh = df[vol_col].quantile(low_q)
    high_thresh = df[vol_col].quantile(high_q)

    # Assign regime labels
    df["regime"] = "normal"
    df.loc[df[vol_col] <= low_thresh, "regime"] = "stable"
    df.loc[df[vol_col] >= high_thresh, "regime"] = "volatile"

    return df
