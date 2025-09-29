# -*- coding: utf-8 -*-
import pandas as pd
import ta
import yfinance as yf

# 1) Descarga sin ajuste automático para tener OHLC y Adj Close
df = yf.download("AAPL", period="6mo", auto_adjust=False, progress=False)

# Aplana MultiIndex si aparece
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

print("Cols crudas:", list(df.columns))
if df.empty:
    raise SystemExit("DataFrame vacio")

# 2) Normaliza columnas
df = df.rename(columns=str.lower).reset_index()
df.rename(columns={"date": "ts"}, inplace=True)

# 2.1) Define 'close' usando Adj Close (preferido), y evita duplicados
if "adj close" in df.columns:
    # Si ya hay 'close' (de Close), elimínalo antes para no duplicar
    if "close" in df.columns:
        df = df.drop(columns=["close"])
    df["close"] = df["adj close"]
    df = df.drop(columns=["adj close"])

# Elimina cualquier duplicado remanente por seguridad
df = df.loc[:, ~df.columns.duplicated()]

# 3) Validación de columnas requeridas
required = ["high", "low", "close"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Faltan columnas requeridas: {missing}. Cols actuales: {list(df.columns)}")

# 4) Tipos numéricos
for c in ["high", "low", "close"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 5) Indicadores
df["atr_14"] = ta.volatility.AverageTrueRange(
    high=df["high"], low=df["low"], close=df["close"], window=14
).average_true_range()

df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
df["sma_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()

print(
    "✅ ta OK | filas:",
    len(df),
    "| cols:",
    [c for c in ("atr_14", "rsi_14", "sma_20", "sma_50", "sma_200") if c in df.columns],
)
