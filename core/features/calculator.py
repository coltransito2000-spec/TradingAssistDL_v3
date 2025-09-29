import pandas as pd
import ta


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        if "close" in df.columns:
            df = df.drop(columns=["close"])
        df["close"] = df["adj close"]
        df = df.drop(columns=["adj close"])
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("ts").copy()
    price_cols = [c for c in ["ts", "open", "high", "low", "close", "volume"] if c in df.columns]
    price = _normalize_ohlc(df[price_cols].set_index("ts")).reset_index()

    price["atr_14"] = ta.volatility.AverageTrueRange(
        price["high"], price["low"], price["close"], window=14
    ).average_true_range()
    price["rsi_14"] = ta.momentum.RSIIndicator(price["close"], window=14).rsi()
    price["sma_20"] = ta.trend.SMAIndicator(price["close"], window=20).sma_indicator()
    price["sma_50"] = ta.trend.SMAIndicator(price["close"], window=50).sma_indicator()
    price["sma_200"] = ta.trend.SMAIndicator(price["close"], window=200).sma_indicator()

    price = price.dropna()
    out = df[["ts", "ticker"]].merge(
        price[["ts", "atr_14", "rsi_14", "sma_20", "sma_50", "sma_200"]],
        on="ts",
        how="inner",
    )
    return out
