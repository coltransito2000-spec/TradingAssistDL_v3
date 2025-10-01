# path: core/models/baseline_xgb.py
# -*- coding: utf-8 -*-

"""
Baseline de clasificación con XGBoost + calibración Isotónica.

Uso CLI (ejemplos):
  - Un ticker y un horizonte, guardando a una ruta concreta:
      python -m core.models.baseline_xgb \
        --tickers AAPL \
        --horizons 10 \
        --out-model artifacts\\baseline_xgb_AAPL.joblib

  - Varios tickers y horizontes (genera un modelo por combinación):
      python -m core.models.baseline_xgb --tickers AAPL MSFT --horizons 10 20

Convenciones de datos (CSV):

  FEATURES  (data/features/<TICKER>.features.csv)
    columnas mínimas:
      ts, ticker,
      atr_14, rsi_14, sma_20, sma_50, sma_200,
      macd, macd_signal, macd_hist
    ts en UTC (datetime64[ns, UTC])

  LABELS  (data/labels/<TICKER>.tb.csv)
    columnas mínimas:
      ts, ticker, y_<h>  (una por horizonte)
    ts en UTC (datetime64[ns, UTC])
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# Dependencias de modelado
try:
    from joblib import dump
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Faltan dependencias críticas (joblib, scikit-learn, xgboost). "
        "Instala con:\n  conda install xgboost scikit-learn joblib\n"
        f"Error: {e}"
    )

# --------------------------------------------------------------------------------------
# Directorios
# --------------------------------------------------------------------------------------
DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas de features por defecto
FEATURE_COLS_DEFAULT = [
    "atr_14",
    "rsi_14",
    "sma_20",
    "sma_50",
    "sma_200",
    "macd",
    "macd_signal",
    "macd_hist",
]


# --------------------------------------------------------------------------------------
# Utilidades de carga y preparación
# --------------------------------------------------------------------------------------
def _read_csv_normalized(path: Path) -> pd.DataFrame:
    """Lee CSV con parse de 'ts' y normaliza 'ticker'."""
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_csv(path, low_memory=False, parse_dates=["ts"])
    if "ticker" not in df.columns:
        raise ValueError(f"{path} no tiene columna 'ticker'")
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _load_features(ticker: str) -> pd.DataFrame:
    p = FEATURES_DIR / f"{ticker}.features.csv"
    return _read_csv_normalized(p)


def _load_labels(ticker: str, h: int) -> pd.DataFrame:
    p = LABELS_DIR / f"{ticker}.tb.csv"
    df = _read_csv_normalized(p)
    y_col = f"y_{h}"
    if y_col not in df.columns:
        raise ValueError(f"{p} no contiene la columna de etiqueta '{y_col}'")
    return df[["ts", "ticker", y_col]].copy()


def _infer_feature_cols(feats_df: pd.DataFrame) -> List[str]:
    """Devuelve la intersección entre columnas del CSV y las por defecto."""
    cols = [c for c in FEATURE_COLS_DEFAULT if c in feats_df.columns]
    if not cols:
        raise ValueError(
            "No se encontraron columnas de features esperadas en el CSV. "
            f"Se esperaba alguna de: {FEATURE_COLS_DEFAULT}"
        )
    return cols


def _join_datasets(
    feats: pd.DataFrame, labels: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Une features y labels en ts+ticker (inner). Devuelve X (features) e y (Series).
    """
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)
    labels["ts"] = pd.to_datetime(labels["ts"], utc=True)

    y_cols = [c for c in labels.columns if c.startswith("y_")]
    if len(y_cols) != 1:
        raise ValueError("Se esperaba exactamente una columna y_<h> en labels.")

    base = feats[["ts", "ticker", *feature_cols]].merge(
        labels[["ts", "ticker", *y_cols]],
        how="inner",
        on=["ts", "ticker"],
    )

    if base.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = base[feature_cols].copy()
    y = base[y_cols[0]].astype(int).copy()
    return X, y


def _temporal_split(
    X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split temporal simple: primeras N para train, resto para test.
    """
    n = len(X)
    if n == 0:
        return X, y, X, y
    cut = int(n * split_ratio)
    X_train = X.iloc[:cut].reset_index(drop=True)
    y_train = y.iloc[:cut].reset_index(drop=True)
    X_test = X.iloc[cut:].reset_index(drop=True)
    y_test = y.iloc[cut:].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: List[str]


# --------------------------------------------------------------------------------------
# Entrenamiento y evaluación
# --------------------------------------------------------------------------------------
def _fit_baseline(ds: Dataset):
    """Entrena XGBoost y lo calibra con Isotonic para probabilidades precisas."""
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # Calibración Isotónica
    calib_model = CalibratedClassifierCV(model, cv=5, method="isotonic")

    # Evitar NaN: usar .bfill() (sin FutureWarning)
    Xtr = ds.X_train.bfill().fillna(0)
    Xte = ds.X_test.bfill().fillna(0)

    calib_model.fit(Xtr, ds.y_train)
    p_test = calib_model.predict_proba(Xte)[:, 1]
    y_test = ds.y_test.values

    metrics = {
        "brier": float(brier_score_loss(y_test, p_test)),
        "auc": float(roc_auc_score(y_test, p_test)),
    }
    return calib_model, metrics


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(tickers: Iterable[str], horizons: Iterable[int], out_model: Optional[str]) -> int:
    tickers = list(dict.fromkeys([t.strip() for t in tickers if t.strip()]))
    horizons = list(sorted(set(int(h) for h in horizons)))
    if not tickers or not horizons:
        print("⚠️ Debes indicar al menos un ticker y un horizonte.")
        return 2

    multiple = len(tickers) * len(horizons) > 1
    out_model_path = Path(out_model) if out_model and not multiple else None

    for t in tickers:
        feats_raw = _load_features(t)
        feature_cols = _infer_feature_cols(feats_raw)

        for h in horizons:
            labels = _load_labels(t, h)

            X, y = _join_datasets(feats_raw, labels, feature_cols)
            if X.empty:
                print(f"⚠️ Sin intersección de datos para {t} h={h}.")
                continue

            X_train, y_train, X_test, y_test = _temporal_split(X, y, split_ratio=0.8)
            ds = Dataset(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_cols=feature_cols,
            )
            model, metrics = _fit_baseline(ds)
            n = len(X)
            print(f"✅ {t} h={h} | n={n} | metrics={metrics}")

            # Guardado
            if out_model_path and not multiple:
                out_path = out_model_path
            else:
                out_path = ARTIFACTS_DIR / f"baseline_xgb_{t}_h{h}.joblib"

            dump(
                {
                    "model": model,
                    "feature_cols": feature_cols,
                    "ticker": t,
                    "horizon": h,
                    "metrics": metrics,
                },
                out_path,
            )
            print(f"💾 Modelo guardado en: {out_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline XGB + calibración isotónica")
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Lista de tickers (ej. AAPL MSFT)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        required=True,
        type=int,
        help="Horizontes para y_h (ej. 10 20)",
    )
    parser.add_argument(
        "--out-model",
        default=None,
        help=(
            "Ruta de salida para un solo modelo. "
            "Si hay varias combinaciones, se ignora y se guardan múltiples archivos."
        ),
    )
    args = parser.parse_args()
    raise SystemExit(main(args.tickers, args.horizons, args.out_model))
