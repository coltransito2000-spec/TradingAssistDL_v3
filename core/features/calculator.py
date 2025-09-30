# -*- coding: utf-8 -*-
# path: core/features/calculator.py
"""
Calcula features técnicos SIN librerías externas (usa common.utils).

Entrada:
    DataFrame OHLCV con columnas: ts, open, high, low, close, volume, (opcional) ticker

Salida:
    DataFrame con indicadores:
      - ATR(14): atr_14
      - RSI(14): rsi_14
      - SMA(20/50/200): sma_20, sma_50, sma_200
      - MACD(12,26,9): macd, macd_signal, macd_hist

Ejemplo CLI (líneas cortas para ruff):
    python -m core.features.calculator --input data/prices_ohlcv/AAPL.csv \
                                       --output data/features/AAPL.feat.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from common.utils import atr, macd, rsi, sma

DATA_DIR = Path("data")
PRICES_DIR = DATA_DIR / "prices_ohlcv"
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres y asegura 'close' preferida desde 'adj close' si existe."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.lower)

    if "adj close" in df.columns:
        if "close" in df.columns:
            df = df.drop(columns=["close"])
        df["close"] = df["adj close"]
        df = df.drop(columns=["adj close"])

    # tipa numéricos
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # fecha/hora
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # elimina duplicados de nombre de columna, si hubiera
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def calculate_technical_features(df: pd.DataFrame, dropna_mode: str = "partial") -> pd.DataFrame:
    """
    Calcula indicadores técnicos con utilidades puras (common.utils).

    dropna_mode:
      - 'partial': elimina filas donde falte SMA200, conserva precio
      - 'full': elimina filas con cualquier NaN
      - 'none': no elimina (puedes imputar después)
    """
    df = _ensure_columns(df)
    if df.empty:
        return df

    required = {"ts", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.sort_values("ts").set_index("ts")

    # Indicadores
    df["atr_14"] = atr(df["high"], df["low"], df["close"], n=14)
    df["rsi_14"] = rsi(df["close"], n=14)
    df["sma_20"] = sma(df["close"], n=20)
    df["sma_50"] = sma(df["close"], n=50)
    df["sma_200"] = sma(df["close"], n=200)

    macd_df = macd(df["close"], fast=12, slow=26, signal=9).rename(
        columns={"signal": "macd_signal", "hist": "macd_hist"}
    )
    df = pd.concat([df, macd_df], axis=1)

    df = df.reset_index()

    if dropna_mode == "partial":
        return df.dropna(subset=["sma_200"]).reset_index(drop=True)
    if dropna_mode == "full":
        return df.dropna().reset_index(drop=True)
    return df.reset_index(drop=True)


def _cli(input_csv: Path, output_csv: Optional[Path], dropna_mode: str) -> None:
    df = pd.read_csv(input_csv)
    feats = calculate_technical_features(df, dropna_mode=dropna_mode)
    if output_csv is None:
        output_csv = FEATURES_DIR / (input_csv.stem + ".features.csv")
    feats.to_csv(output_csv, index=False)
    print(f"✅ Features escritos en: {output_csv} | filas={len(feats)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Calcula features técnicos.")
    p.add_argument("--input", type=Path, required=True, help="CSV OHLCV con columna 'ts'")
    p.add_argument("--output", type=Path, default=None, help="CSV de salida para features")
    p.add_argument(
        "--dropna-mode",
        type=str,
        default="partial",
        choices=("partial", "full", "none"),
        help="Tratamiento de NaNs (default: partial)",
    )
    args = p.parse_args()
    _cli(args.input, args.output, args.dropna_mode)


if __name__ == "__main__":
    main()
