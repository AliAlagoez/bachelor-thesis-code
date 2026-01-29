import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA


def baseline_zero_return(train_df, test_row, y_col="returns"):
    """
    Naive Forecast nach Hyndman: Prognose für t+1 ist der zuletzt beobachtete Wert (Persistenz).
    Für Renditen: r̂_{t+1} = r_t
    """
    y = train_df[y_col].dropna()
    if len(y) == 0:
        return 0.0
    return float(y.iloc[-1])

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)


def arima_101(train_df, test_row, y_col="returns"):
    """
    Kleines, stabiles ARIMA(1,0,1) auf Renditen.
    Trainiert nur auf historischen Returns und prognostiziert den nächsten Wert.
    """
    y = train_df[y_col].dropna().values

    # Falls zu wenige Daten da sind, fallback auf Baseline
    if len(y) < 30:
        return 0.0

    try:
        model = ARIMA(y, order=(1, 0, 1))
        fitted = model.fit()
        y_hat = fitted.forecast(steps=1)[0]
        return float(y_hat)
    except Exception:
        # falls ARIMA mal zickt -> Baseline, damit Backtest nie crasht
        return 0.0

def ml_hist_gb(train_df, test_row, y_col="returns", feature_cols=None):
    """
    Schnelles Gradient Boosting Modell (scikit-learn).
    - Kein langes Training
    - Kein GPU
    - Sehr robust
    """

    if feature_cols is None:
        raise ValueError("feature_cols muss angegeben werden (Liste der Feature-Spalten).")

    # Trainingsdaten ohne NaNs
    train = train_df.dropna(subset=feature_cols + [y_col])

    # Wenn zu wenig Daten -> fallback
    if len(train) < 80:
        return 0.0

    X_train = train[feature_cols].values
    y_train = train[y_col].values
    X_test = test_row[feature_cols].values

    # stabiles Modell
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.1,
        max_iter=200,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)[0]
    return float(y_hat)
