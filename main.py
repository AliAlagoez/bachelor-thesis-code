import pandas as pd
import numpy as np

from src.load_data import load_price_csv
from src.regimes import label_volatility_regimes
from src.backtest import walk_forward_predict, mae, rmse
from src.features import add_features
from src.models import baseline_zero_return, arima_101, ml_hist_gb


# Feature columns for ML
feature_cols = [
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_7", "ret_lag_14",
    "ret_mean_7", "ret_mean_30", "ret_vol_7", "ret_vol_30"
]


def eval_model(df, name, model_fn, feature_cols_for_backtest=None):
    """
    Runs a walk-forward backtest for a given model and computes performance
    metrics for the full sample and for volatility-based regimes.
    """
    # Generate one-step-ahead predictions using an expanding training window
    preds = walk_forward_predict(
        df,
        model_fn,
        y_col="returns",
        feature_cols=feature_cols_for_backtest,
        min_train=60
    )
    # Attach the regime labels for regime-specific evaluation
    preds = preds.join(df[["regime"]], how="left")

    stable = preds[preds["regime"] == "stable"]
    volatile = preds[preds["regime"] == "volatile"]

    # Safety check: regime-specific metrics require both subsets to be non-empty
    if len(stable) == 0 or len(volatile) == 0:
        return None

    # Overall performance
    mae_total = mae(preds["y_true"], preds["y_pred"])
    rmse_total = rmse(preds["y_true"], preds["y_pred"])

    # Regime-specific performance
    mae_stable = mae(stable["y_true"], stable["y_pred"])
    mae_volatile = mae(volatile["y_true"], volatile["y_pred"])
    rmse_stable = rmse(stable["y_true"], stable["y_pred"])
    rmse_volatile = rmse(volatile["y_true"], volatile["y_pred"])

    # Robustness ratios: how much error increases in volatile vs. stable regimes
    mae_ratio = mae_volatile / mae_stable
    rmse_ratio = rmse_volatile / rmse_stable

    # Terminal output 
    print(f"\n=== {name} ===")
    print("MAE total :", round(mae_total, 6), "| RMSE total:", round(rmse_total, 6))
    print("MAE stable:", round(mae_stable, 6), "| MAE volatile:", round(mae_volatile, 6))
    print("RMSE stable:", round(rmse_stable, 6), "| RMSE volatile:", round(rmse_volatile, 6))
    print("Robustness MAE Ratio:", round(mae_ratio, 3), "| RMSE Ratio:", round(rmse_ratio, 3))


    return {
        "model": name,

        # Full sample
        "mae_total": mae_total,
        "rmse_total": rmse_total,

        # Stable regime
        "mae_stable": mae_stable,
        "rmse_stable": rmse_stable,

        # Volatile regime
        "mae_volatile": mae_volatile,
        "rmse_volatile": rmse_volatile,

        # Robustness ratios
        "mae_ratio": mae_ratio,
        "rmse_ratio": rmse_ratio,

        # Number of observations per regime
        "n_stable": int(len(stable)),
        "n_volatile": int(len(volatile)),
    }




def run_for_coin(coin_name: str, csv_path: str):
    """
    Loads price data for a given cryptocurrency, computes returns and features,
    assigns volatility regimes, and evaluates all forecasting models.
    """
    print(f"\n\n#############################")
    print(f"Coin: {coin_name} | File: {csv_path}")
    print(f"#############################")

    # Load the price series
    df = load_price_csv(csv_path)

    print("Rows:", len(df), "| from", df.index.min(), "to", df.index.max())
    print(df.head(3))


    # Target variable: Logarithmic returns
    df["returns"] = np.log(df["close"]).diff()

    # Feature engineering
    df = add_features(df)

    # Volatility regimes based on lagged rolling volatility
    df["vol_30"] = df["returns"].shift(1).rolling(30).std()
    df = label_volatility_regimes(df)

    results = []

    # Baseline model (persistence forecast)
    r = eval_model(df, "Baseline", baseline_zero_return)
    if r:
        r["coin"] = coin_name
        results.append(r)

    # ARIMA model
    r = eval_model(df, "ARIMA(1,0,1)", arima_101)
    if r:
        r["coin"] = coin_name
        results.append(r)

    # Machine learning model
    r = eval_model(
        df,
        "ML (HistGradientBoosting)",
        lambda train, row: ml_hist_gb(train, row, feature_cols=feature_cols),
        feature_cols_for_backtest=feature_cols
    )
    if r:
        r["coin"] = coin_name
        results.append(r)

    return results


def main():
    """
    Runs the full evaluation pipeline for all configured cryptocurrencies.
    """
    coin_files = {
        "BTC": "data/BTC.csv",
        "ETH": "data/ETH.csv",
    }

    all_results = []
    for coin, path in coin_files.items():
        all_results.extend(run_for_coin(coin, path))

    results_df = pd.DataFrame(all_results)

    # Print aggregated results
    print("\n\n=== Results overview (all coins) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(results_df)

    # Save results
    results_df.to_csv("results/metrics.csv", index=False)
    print("\nSaved: results/metrics.csv")


if __name__ == "__main__":
    main()
