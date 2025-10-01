import aiosqlite
from .config import DB_PATH

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER, side TEXT, entry REAL, stop REAL, target REAL, pt1 REAL,
  qty REAL, leverage REAL, status TEXT, meta_json TEXT
);
CREATE TABLE IF NOT EXISTS equity_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER, equity REAL
);
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER, side TEXT, entry REAL, stop REAL, target REAL, pt1 REAL
);
"""
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_SQL)
        await db.commit()
