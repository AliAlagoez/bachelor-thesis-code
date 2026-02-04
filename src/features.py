def add_features(df):
    """
    Constructs simple and robust feature variables based on historical returns.

    All features are computed exclusively from past information using
    lagged values and rolling statistics to prevent look-ahead bias.
    """
    df = df.copy()

    # Lagged return features
    for lag in [1, 2, 3, 7, 14]:
        df[f"ret_lag_{lag}"] = df["returns"].shift(lag)

    # Rolling statistics based on past returns only
    past_ret = df["returns"].shift(1)

    df["ret_mean_7"] = past_ret.rolling(7).mean()
    df["ret_mean_30"] = past_ret.rolling(30).mean()
    df["ret_vol_7"] = past_ret.rolling(7).std()
    df["ret_vol_30"] = past_ret.rolling(30).std()

    return df
