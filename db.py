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

        CREATE TABLE IF NOT EXISTS vector_backups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            model       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            data        BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_projects_loader_version
            ON projects(loader, mc_version);

        CREATE INDEX IF NOT EXISTS idx_projects_embedded
            ON projects(embedded_at);

        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            args        TEXT NOT NULL DEFAULT '{}',
            status      TEXT NOT NULL DEFAULT 'pending',
            started_at  TEXT,
            finished_at TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS job_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id     TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            line       TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_job_logs_job ON job_logs(job_id);

        CREATE TABLE IF NOT EXISTS query_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text  TEXT NOT NULL,
            loader      TEXT NOT NULL,
            version     TEXT NOT NULL,
            query_vec   BLOB NOT NULL,
            results     TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_query_cache_loader_version
            ON query_cache(loader, version);
    """)
    conn.commit()
