# Path: core/config.py

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Cargar variables de entorno desde .env si existe
load_dotenv()


def get_db_engine():
    """
    Retorna un engine de SQLAlchemy según la variable de entorno DB_URL.

    Ejemplo en .env:
        DB_URL=postgresql+psycopg2://user:pass@localhost:5432/tradingassist_prod
    """
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("❌ Falta DB_URL en variables de entorno o archivo .env")
    return create_engine(db_url)
