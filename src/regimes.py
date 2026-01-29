import pandas as pd

def label_volatility_regimes(df, vol_col="vol_30", low_q=0.25, high_q=0.75):
    """
    Labelt Marktphasen basierend auf rollierender Volatilität.
    - stable: unteres Quantil
    - volatile: oberes Quantil
    """

    df = df.copy()

    # Einteilung der Quantile
    low_thresh = df[vol_col].quantile(low_q)
    high_thresh = df[vol_col].quantile(high_q)

    df["regime"] = "normal"
    df.loc[df[vol_col] <= low_thresh, "regime"] = "stable"
    df.loc[df[vol_col] >= high_thresh, "regime"] = "volatile"

    return df
