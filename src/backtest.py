import numpy as np
import pandas as pd

def walk_forward_predict(df, model_fn, y_col="returns", feature_cols=None, min_train=60):
    """
    Performs a walk-forward (rolling origin) backtest.

    At each time step t:
    - The model is trained on all observations up to t-1.
    - A one-step-ahead prediction is generated for observation t.
    """
    preds = []
    idxs = []

    # Iterate over the dataset using an expanding training window
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        test_row = df.iloc[i:i+1]

        # skip if target NaN
        y_true = test_row[y_col].values[0]
        if pd.isna(y_true):
            continue

        # If feature columns are specified, ensure no missing values
        if feature_cols is not None:
            # Skip prediction if test features contain missing values
            if test_row[feature_cols].isna().any(axis=1).values[0]:
                continue
            # Remove training rows with missing feature or target values
            if train[feature_cols].isna().any(axis=1).any():
                train = train.dropna(subset=feature_cols + [y_col])

        # Generate one-step-ahead prediction
        y_hat = model_fn(train, test_row)

        preds.append((y_true, y_hat))
        idxs.append(test_row.index[0])

    # Store predictions and true values in a DataFrame
    out = pd.DataFrame(preds, columns=["y_true", "y_pred"], index=idxs)
    return out

def mae(y_true, y_pred):
    """
    Computes the Mean Absolute Error (MAE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
