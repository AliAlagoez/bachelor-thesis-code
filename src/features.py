def add_features(df):
    """
    Erstellt einfache, stabile Features aus Renditen.
    Wichtig: Nur Vergangenheitswerte (shift/rolling) -> kein Leakage.
    """
    df = df.copy()

    # Lags der Renditen
    for lag in [1, 2, 3, 7, 14]:
        df[f"ret_lag_{lag}"] = df["returns"].shift(lag)

    # Rollierende Statistiken (aus Vergangenheit)
    df["ret_mean_7"] = df["returns"].rolling(7).mean()
    df["ret_mean_30"] = df["returns"].rolling(30).mean()
    df["ret_vol_7"] = df["returns"].rolling(7).std()
    df["ret_vol_30"] = df["returns"].rolling(30).std()

    return df
