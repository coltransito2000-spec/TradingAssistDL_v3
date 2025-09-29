import os

from sqlalchemy import create_engine

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://user:pass@localhost:5432/tradingassist_prod")
TZ = os.getenv("TZ", "America/Bogota")


def get_db_engine():
    return create_engine(DB_URL, pool_pre_ping=True, future=True)
