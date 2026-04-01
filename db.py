import sqlite3
from pathlib import Path

import config

def _db_path() -> Path:
    cfg = config.load()
    path = Path(cfg.get("storage", {}).get("mods_db", Path(__file__).parent / "mods.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id              TEXT PRIMARY KEY,
            slug            TEXT NOT NULL,
            title           TEXT NOT NULL,
            description     TEXT,
            body            TEXT,
            loader          TEXT NOT NULL,
            mc_version      TEXT NOT NULL,
            categories      TEXT,
            downloads       INTEGER DEFAULT 0,
            follows         INTEGER DEFAULT 0,
            author          TEXT,
            license         TEXT,
            client_side     TEXT,
            server_side     TEXT,
            date_created    TEXT,
            date_modified   TEXT,
            embedded_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            project_id  TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
            vector      BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key     TEXT PRIMARY KEY,
            value   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_projects_loader_version
            ON projects(loader, mc_version);

        CREATE INDEX IF NOT EXISTS idx_projects_embedded
            ON projects(embedded_at);
    """)
    conn.commit()
