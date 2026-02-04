import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA


def baseline_zero_return(train_df, test_row, y_col="returns"):
    """
    Naive benchmark forecast following Hyndman's persistence principle.

    The one-step-ahead forecast equals the last observed return:
    r̂_{t+1} = r_t
    """
    y = train_df[y_col].dropna()

    # Fallback if no historical data is available
    if len(y) == 0:
        return 0.0
    return float(y.iloc[-1])

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress non-critical warnings for cleaner console output during rolling estimation
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

def arima_101(train_df, test_row, y_col="returns"):
    """
    Fits an ARIMA(1,0,1) model to historical returns and produces
    a one-step-ahead forecast.
    """
    y = train_df[y_col].dropna().values

    # If there is insufficient data, fallback
    if len(y) < 30:
        return 0.0

    try:
        model = ARIMA(y, order=(1, 0, 1))
        fitted = model.fit()
        y_hat = fitted.forecast(steps=1)[0]
        return float(y_hat)
    except Exception:
        # Fallback in case of issues
        return 0.0

def ml_hist_gb(train_df, test_row, y_col="returns", feature_cols=None):
    """
    Histogram-based Gradient Boosting regression model.    

    The model is trained on engineered return-based features and used
    to generate a one-step-ahead prediction.
    """

    if feature_cols is None:
        raise ValueError("feature_cols must be provided as a list of feature column names.")

    # Training data without NaNs
    train = train_df.dropna(subset=feature_cols + [y_col])

    # Fallback if the training sample is too small
    if len(train) < 80:
        return 0.0

    X_train = train[feature_cols].values
    y_train = train[y_col].values
    X_test = test_row[feature_cols].values

    # Configure a stable gradient boosting model
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.1,
        max_iter=200,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)[0]
    return float(y_hat)
