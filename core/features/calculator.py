# Path: core/features/calculator.py

import pandas as pd
import ta


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas MultiIndex -> nivel 0 (primero)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas OHLCV:
    - Aplana MultiIndex si existe.
    - Pasa a minúsculas.
    - Usa 'adj close' como 'close' (evita saltos por dividendos/splits).
    - Convierte a numérico (coerce) y dedup de columnas.
    """
    df = _flatten_columns(df).copy()
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


def _normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el df base para asegurar que ['ts','ticker'] sean columnas planas
    y sin MultiIndex.
    """
    df = _flatten_columns(df).copy()
    df = df.rename(columns=str.lower)
    return df


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos básicos:
    ATR(14), RSI(14), SMA(20), SMA(50), SMA(200).

    Requisitos mínimos de entrada:
    - df tiene columnas de precio: open/high/low/(adj close|close)/volume
    - df tiene 'ts' (timestamp/fecha) y 'ticker'
    """
    if df.empty:
        return df

    # 1) Normaliza df base (aplana columnas y minúsculas) antes de cualquier selección
    df = _normalize_base(df).sort_values("ts").copy()

    # 2) Construye dataframe de precios normalizado (OHLCV)
    price_cols = [
        c for c in ["ts", "open", "high", "low", "close", "adj close", "volume"] if c in df.columns
    ]
    if "ts" not in price_cols:
        raise ValueError("❌ Falta columna 'ts' en el DataFrame de entrada.")
    price = _normalize_ohlc(df[price_cols].set_index("ts")).reset_index()

    # 3) Calcula indicadores
    price["atr_14"] = ta.volatility.AverageTrueRange(
        price["high"], price["low"], price["close"], window=14
    ).average_true_range()
    price["rsi_14"] = ta.momentum.RSIIndicator(price["close"], window=14).rsi()
    price["sma_20"] = ta.trend.SMAIndicator(price["close"], window=20).sma_indicator()
    price["sma_50"] = ta.trend.SMAIndicator(price["close"], window=50).sma_indicator()
    price["sma_200"] = ta.trend.SMAIndicator(price["close"], window=200).sma_indicator()

    # 🔽 En vez de dropna total, exigimos solo que los indicadores "rápidos" existan
    price = price.dropna(subset=["atr_14", "rsi_14"])

    # 4) Base para merge: asegura columnas planas y únicas
    if "ticker" not in df.columns:
        raise ValueError("❌ Falta columna 'ticker' en el DataFrame de entrada.")
    base = df[["ts", "ticker"]].drop_duplicates()

    # 5) Merge final
    out = base.merge(
        price[["ts", "atr_14", "rsi_14", "sma_20", "sma_50", "sma_200"]],
        on="ts",
        how="inner",
    )

    return out
