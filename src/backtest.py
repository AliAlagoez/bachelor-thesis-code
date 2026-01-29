import numpy as np
import pandas as pd

def walk_forward_predict(df, model_fn, y_col="returns", feature_cols=None, min_train=60):
    """
    Walk-forward Backtest:
    - train: df[:t]
    - predict: t
    model_fn(train_df, x_row) -> y_hat
    """
    preds = []
    idxs = []

    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        test_row = df.iloc[i:i+1]

        # skip if target NaN
        y_true = test_row[y_col].values[0]
        if pd.isna(y_true):
            continue

        # optional: require feature cols not NaN
        if feature_cols is not None:
            if test_row[feature_cols].isna().any(axis=1).values[0]:
                continue
            if train[feature_cols].isna().any(axis=1).any():
                # drop rows with NaN in training features
                train = train.dropna(subset=feature_cols + [y_col])

        y_hat = model_fn(train, test_row)

        preds.append((y_true, y_hat))
        idxs.append(test_row.index[0])

    out = pd.DataFrame(preds, columns=["y_true", "y_pred"], index=idxs)
    return out

def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
