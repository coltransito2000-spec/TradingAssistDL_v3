# Path: README.md

# TradingAssistDL

Pipeline y utilidades para ingesta de datos OHLCV, cálculo de features técnicos y entrenamiento/serving de modelos DL/ML.

## 1) Requisitos

- Windows 10/11 x64
- Miniforge/Conda + **mamba** instalado para el *user* (no global)
- Git instalado en el entorno `tradingassistdl`
- Python 3.11 (en el entorno conda)

## 2) Crear/activar entorno

```powershell
mamba create -n tradingassistdl python=3.11 -c conda-forge
mamba activate tradingassistdl
python -V
