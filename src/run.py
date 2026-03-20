import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS
from catboost import CatBoostRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer


def baselines_predict(df, horizon):
    sf = StatsForecast(
        models=[
            Naive(),
            SeasonalNaive(season_length=7),
            AutoTheta(season_length=7),
            AutoETS(season_length=7),
        ],
        freq=1,
    )
    forecasts = sf.forecast(df=df, h=horizon)
    return forecasts
    
def get_features(df):
    lags=[1, 2, 7, 14]
    df = df.copy()
    cols = []
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
        cols += [f"lag_{lag}"]
    return df, cols

def catboost_fit(df):
    df, cols = get_features(df)
    df = df.dropna()
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0
    ).fit(df[cols], df["y"])
    return model

def catboost_predict(df, horizon):
    model = catboost_fit(df)
    all_preds = []
    for i, group in df.groupby("unique_id"):
        group = group.sort_values("ds").copy()
        hist = group.iloc[-20:].copy()
        series_preds = []
        for _ in range(horizon):
            df_feat, cols = get_features(hist)
            df_feat = df_feat.dropna()
            last_row = df_feat.iloc[-1:]
            pred = model.predict(last_row[cols])[0]
            new_row = pd.DataFrame({
                "unique_id": [i],
                "ds": [last_row["ds"] + 1],
                "y": [pred],
            })
            hist = pd.concat([hist, new_row], ignore_index=True)
            series_preds.append(pred)
        all_preds.append(pd.DataFrame({
            "unique_id": i,
            "ds": np.arange(group["ds"].max() + 1, group["ds"].max() + horizon + 1),
            "CatBoost": series_preds,
        }))
    return pd.concat(all_preds, ignore_index=True)

def patchtst_predict(df, horizon):
    model = PatchTST(
        h=horizon,
        input_size=30,
        max_steps=200,
        logger=False,
        enable_progress_bar=False,
    )

    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df)
    preds = nf.predict()
    return preds

def fit_scalers(df, scaler_class, n=None):
    scalers = {}
    for i, group in df.groupby("unique_id"):
        y = group["y"].values.reshape(-1, 1)
        if n is not None:
            scaler = scaler_class(n_quantiles=n)
        else:
            scaler = scaler_class()
        scaler.fit(y)
        scalers[i] = scaler

    return scalers

def transform_scalers(df, scalers):
    df = df.copy()
    if scalers is None:
        return df
    
    for i, group in df.groupby("unique_id"):
        y = group["y"].values.reshape(-1, 1)
        scaled = scalers[i].transform(y).flatten()
        df.loc[group.index, "y"] = scaled
    return df

def inv_transform_scalers(df, scalers, val_col="y"):
    df = df.copy()
    if scalers is None:
        return df
        
    for i, group in df.groupby("unique_id"):
        y = group[val_col].values.reshape(-1, 1)
        inv = scalers[i].inverse_transform(y).flatten()
        df.loc[group.index, val_col] = inv
    return df
    
    
def main():
    df = pd.read_csv("data.csv", header=0, index_col=0)
    horizon = 14

    train_list = []
    test_list = []
    for _, group in df.groupby("unique_id"):
        group = group.sort_values("ds")
        train_list.append(group.iloc[:-horizon])
        test_list.append(group.iloc[-horizon:])
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    
    standard = fit_scalers(train_df, StandardScaler)
    robust = fit_scalers(train_df, RobustScaler)
    quantile = fit_scalers(train_df, QuantileTransformer, 50)

    scalers = {
        "Original": None,
        "Standard": standard,
        "Robust": robust,
        "Quantile": quantile
    }
    
    models = ["Naive", "SeasonalNaive", "AutoTheta", 
              "AutoETS", "CatBoost", "PatchTST"]
    metrics_df = pd.DataFrame(index=models, columns=list(scalers.keys()), dtype=float)
    
    for k in scalers:
        scaled_train = transform_scalers(train_df, scalers[k])
        forecast_df = baselines_predict(scaled_train, horizon)
        catboost_df = catboost_predict(scaled_train, horizon)
        patchtst_df = patchtst_predict(scaled_train, horizon)
        merged = forecast_df.merge(catboost_df, on=["unique_id", "ds"])
        merged = merged.merge(patchtst_df, on=["unique_id", "ds"])
        for m in models:
            df_cnt = merged[["unique_id", "ds", m]]
            df_cnt = inv_transform_scalers(df_cnt, scalers[k], val_col=m)
            metrics_df.loc[m, k] = mean_absolute_error(test_df["y"].values, df_cnt[m].values)
            #print(mean_absolute_error(test_df["y"].values, df_cnt[m].values))
        #print(merged)
    metrics_df.to_csv("../results/mae.csv")

if __name__ == "__main__":
    main()