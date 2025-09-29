# Path: tests/conftest.py

# Fuerza a que la raíz del repo actual (donde está este archivo) esté primero en sys.path.
# Evita que Python tome un "core" de otra ruta (p. ej. F:\TradingAssistDL) por delante.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # F:\TradingAssistDL_v3
REPO_STR = str(REPO_ROOT)

# Inserta al inicio si no está ya
if REPO_STR not in sys.path:
    sys.path.insert(0, REPO_STR)
