import pandas as pd
import numpy as np

from src.load_data import load_price_csv
from src.regimes import label_volatility_regimes
from src.backtest import walk_forward_predict, mae, rmse
from src.features import add_features
from src.models import baseline_zero_return, arima_101, ml_hist_gb


# Feature-Spalten für ML
feature_cols = [
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_7", "ret_lag_14",
    "ret_mean_7", "ret_mean_30", "ret_vol_7", "ret_vol_30"
]


def eval_model(df, name, model_fn, feature_cols_for_backtest=None):
    preds = walk_forward_predict(
        df,
        model_fn,
        y_col="returns",
        feature_cols=feature_cols_for_backtest,
        min_train=60
    )
    preds = preds.join(df[["regime"]], how="left")

    stable = preds[preds["regime"] == "stable"]
    volatile = preds[preds["regime"] == "volatile"]

    # Sicherheit
    if len(stable) == 0 or len(volatile) == 0:
        return None

    mae_total = mae(preds["y_true"], preds["y_pred"])
    rmse_total = rmse(preds["y_true"], preds["y_pred"])

    mae_stable = mae(stable["y_true"], stable["y_pred"])
    mae_volatile = mae(volatile["y_true"], volatile["y_pred"])
    rmse_stable = rmse(stable["y_true"], stable["y_pred"])
    rmse_volatile = rmse(volatile["y_true"], volatile["y_pred"])

    mae_ratio = mae_volatile / mae_stable
    rmse_ratio = rmse_volatile / rmse_stable

    # Terminal-Ausgabe (optional, aber hilfreich)
    print(f"\n=== {name} ===")
    print("MAE total :", round(mae_total, 6), "| RMSE total:", round(rmse_total, 6))
    print("MAE stable:", round(mae_stable, 6), "| MAE volatile:", round(mae_volatile, 6))
    print("RMSE stable:", round(rmse_stable, 6), "| RMSE volatile:", round(rmse_volatile, 6))
    print("Robustheit MAE-Ratio:", round(mae_ratio, 3), "| RMSE-Ratio:", round(rmse_ratio, 3))


    return {
        "model": name,

        # Gesamt
        "mae_total": mae_total,
        "rmse_total": rmse_total,

        # Stabile Phase
        "mae_stable": mae_stable,
        "rmse_stable": rmse_stable,

        # Volatile Phase
        "mae_volatile": mae_volatile,
        "rmse_volatile": rmse_volatile,

        # Robustheit
        "mae_ratio": mae_ratio,
        "rmse_ratio": rmse_ratio,

        # Beobachtungen
        "n_stable": int(len(stable)),
        "n_volatile": int(len(volatile)),
    }




def run_for_coin(coin_name: str, csv_path: str):
    print(f"\n\n#############################")
    print(f"Coin: {coin_name} | Datei: {csv_path}")
    print(f"#############################")

    df = load_price_csv(csv_path)

    print("Rows:", len(df), "| from", df.index.min(), "to", df.index.max())
    print(df.head(3))


    # Zielvariable + Features 
    # Berechnung der logarithmischen Rendite
    df["returns"] = np.log(df["close"]).diff()
    df = add_features(df)

    # Regimes
    # Berechnung der rollierenden Volatilität

    df["vol_30"] = df["returns"].rolling(30).std()
    df = label_volatility_regimes(df)

    results = []

    # Baseline
    r = eval_model(df, "Baseline", baseline_zero_return)
    if r: 
        r["coin"] = coin_name
        results.append(r)

    # ARIMA
    r = eval_model(df, "ARIMA(1,0,1)", arima_101)
    if r:
        r["coin"] = coin_name
        results.append(r)

    # ML
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
    
    coin_files = {
        "BTC": "data/BTC.csv",
        "ETH": "data/ETH.csv",
    }

    all_results = []
    for coin, path in coin_files.items():
        all_results.extend(run_for_coin(coin, path))

    results_df = pd.DataFrame(all_results)

    print("\n\n=== Ergebnisübersicht (alle Coins) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(results_df)

    # speichern (für Thesis super)
    results_df.to_csv("results/metrics.csv", index=False)
    print("\nGespeichert: results/metrics.csv")


if __name__ == "__main__":
    main()
