# Path: tests/test_features_smoke.py

import yfinance as yf

from core.features.calculator import calculate_technical_features


def test_calculate_technical_features_smoke():
    """Smoke test: asegura que los features se calculan sin romper."""
    df = yf.download("AAPL", period="3mo", auto_adjust=False, progress=False).reset_index()
    df = df.rename(columns=str.lower)
    df.rename(columns={"date": "ts"}, inplace=True)
    df["ticker"] = "AAPL"

    features = calculate_technical_features(df)

    assert not features.empty, "❌ Features vacíos"
    for col in ["atr_14", "rsi_14", "sma_20", "sma_50", "sma_200"]:
        assert col in features.columns, f"❌ Falta columna {col}"
