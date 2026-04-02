import io
import json
import pickle
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
import db

app = FastAPI(title="Mod Semantic Search")

_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
templates = Jinja2Templates(directory=_BASE / "templates")

MODRINTH_URL = "https://modrinth.com/mod"
EMBED_ENDPOINT = "/v1/embeddings"
RERANK_ENDPOINT = "/v1/rerank"

_RUNNER_SCRIPT = str(Path(__file__).parent.parent / "workers" / "runner.py")
_HEARTBEAT_STALE_SEC = 30

# ── runner auto-spawn ─────────────────────────────────────────────────────────

def _runner_alive() -> bool:
    conn = db.connect()
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'runner_heartbeat'"
    ).fetchone()
    conn.close()
    if not row:
        return False
    try:
        ts = datetime.fromisoformat(row["value"])
        # SQLite stores as naive UTC; make tz-aware for comparison
        if ts.tzinfo is None:
            from datetime import timezone as _tz
            ts = ts.replace(tzinfo=_tz.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age < _HEARTBEAT_STALE_SEC
    except Exception:
        return False


def _spawn_runner() -> None:
    subprocess.Popen(
        [sys.executable, "-m", "workers.runner"],
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


@app.on_event("startup")
async def _startup() -> None:
    conn = db.connect()
    db.init(conn)
    conn.close()
    if not _runner_alive():
        print("[web] Runner not detected — spawning workers.runner")
        _spawn_runner()


async def _watchdog() -> None:
    """Periodically re-spawn the runner if its heartbeat goes stale."""
    import asyncio
    while True:
        await asyncio.sleep(30)
        if not _runner_alive():
            print("[web] Runner heartbeat stale — re-spawning")
            _spawn_runner()


@app.on_event("startup")
async def _start_watchdog() -> None:
    import asyncio
    asyncio.create_task(_watchdog())


# ── helpers ───────────────────────────────────────────────────────────────────

def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()

def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q

def _embed(text: str, base_url: str) -> np.ndarray:
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{base_url}{EMBED_ENDPOINT}", json={"input": [text]})
        r.raise_for_status()
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)

def _rerank(query: str, docs: list, base_url: str) -> list:
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{base_url}{RERANK_ENDPOINT}", json={"query": query, "documents": docs})
        r.raise_for_status()
    scores = [0.0] * len(docs)
    for item in r.json()["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores

def _fmt_side(client_side: str, server_side: str) -> str:
    parts = []
    if client_side in ("required", "optional"):
        parts.append("client")
    if server_side in ("required", "optional"):
        parts.append("server")
    return "+".join(parts) if parts else "unknown"


# ── conflict groups (mirrored from workers/runner.py) ─────────────────────────

_CONFLICT_GROUPS = {
    "retrieval": {"retrieval"},
    "prune":     {"retrieval"},
    "embedding": {"embedding"},
    "reset":     {"embedding"},
    "pipeline":  {"retrieval", "embedding"},
}


# ── pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conn = db.connect()
    db.init(conn)
    loaders = [r[0] for r in conn.execute(
        "SELECT DISTINCT loader FROM projects ORDER BY loader"
    ).fetchall()]
    versions = [r[0] for r in conn.execute(
        "SELECT DISTINCT mc_version FROM projects ORDER BY mc_version DESC"
    ).fetchall()]
    conn.close()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "loaders": loaders,
        "versions": versions,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    conn = db.connect()
    db.init(conn)
    loaders = [r[0] for r in conn.execute(
        "SELECT DISTINCT loader FROM projects ORDER BY loader"
    ).fetchall()]
    versions = [r[0] for r in conn.execute(
        "SELECT DISTINCT mc_version FROM projects ORDER BY mc_version DESC"
    ).fetchall()]
    backups = conn.execute(
        "SELECT id, name, model, created_at FROM vector_backups ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "loaders": loaders or ["neoforge", "fabric"],
        "versions": versions or ["1.21.1"],
        "backups": [{"id": b[0], "name": b[1], "model": b[2], "created_at": b[3]} for b in backups],
    })


# ── API: status ───────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    conn = db.connect()
    db.init(conn)
    total = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    last_sync = conn.execute("SELECT value FROM meta WHERE key = 'last_sync'").fetchone()
    breakdown = conn.execute("""
        SELECT loader, mc_version, COUNT(*) as count
        FROM projects GROUP BY loader, mc_version
        ORDER BY loader, mc_version DESC
    """).fetchall()
    conn.close()
    return {
        "total": total,
        "embedded": embedded,
        "pending_embedding": total - embedded,
        "last_sync": last_sync[0] if last_sync else None,
        "breakdown": [{"loader": r[0], "version": r[1], "count": r[2]} for r in breakdown],
    }


@app.get("/api/inference/health")
async def inference_health():
    cfg = config.load()
    servers = {
        "embedder": cfg["inference"]["embedding_url"].rstrip("/"),
        "reranker": cfg["inference"]["reranker_url"].rstrip("/"),
    }
    result = {}
    for name, base_url in servers.items():
        try:
            with httpx.Client(timeout=3) as client:
                r = client.get(f"{base_url}/health")
            result[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as exc:
            result[name] = "unreachable"
    return result


class JobRequest(BaseModel):
    action: str
    loader: Optional[str] = None
    version: Optional[str] = None
    sync: Optional[bool] = False
    batch_size: Optional[int] = 32

@app.post("/api/jobs/start")
async def start_job(req: JobRequest):
    action = req.action
    if action not in _CONFLICT_GROUPS:
        raise HTTPException(400, f"Unknown action: {action}")

    conn = db.connect()
    running_types = [
        r["type"] for r in conn.execute(
            "SELECT type FROM jobs WHERE status = 'running'"
        ).fetchall()
    ]
    running_groups: set = set()
    for t in running_types:
        running_groups |= _CONFLICT_GROUPS.get(t, {t})

    job_groups = _CONFLICT_GROUPS[action]
    if job_groups & running_groups:
        conn.close()
        conflict = next(iter(job_groups & running_groups))
        raise HTTPException(409, f"A {conflict} job is already running")

    job_id = str(uuid.uuid4())[:8]
    args = {
        "loader":     req.loader,
        "version":    req.version,
        "sync":       req.sync,
        "batch_size": req.batch_size,
    }
    conn.execute(
        "INSERT INTO jobs (id, type, args) VALUES (?, ?, ?)",
        (job_id, action, json.dumps(args)),
    )
    conn.commit()
    conn.close()
    return {"job_id": job_id}


@app.get("/api/jobs/status")
async def jobs_status():
    conn = db.connect()
    # Jobs that are running, or finished within the last 30 seconds
    jobs = conn.execute("""
        SELECT id, type, args, status, started_at, finished_at, created_at
        FROM jobs
        WHERE status IN ('pending', 'running')
           OR (status IN ('done', 'error')
               AND finished_at >= datetime('now', '-30 seconds'))
        ORDER BY created_at DESC
        LIMIT 20
    """).fetchall()

    result = []
    for j in jobs:
        last_log = conn.execute(
            "SELECT line FROM job_logs WHERE job_id = ? ORDER BY id DESC LIMIT 1",
            (j["id"],),
        ).fetchone()
        result.append({
            "id":          j["id"],
            "type":        j["type"],
            "status":      j["status"],
            "log":         [last_log["line"]] if last_log else [],
            "started_at":  j["started_at"],
            "finished_at": j["finished_at"],
        })
    conn.close()
    return {"jobs": result}


# ── API: backups ──────────────────────────────────────────────────────────────

@app.post("/api/backups/create")
async def create_backup(name: Optional[str] = Query(default=None)):
    cfg = config.load()
    model = cfg["inference"]["embedding_url"]
    if not name:
        name = datetime.now(timezone.utc).strftime("backup-%Y%m%d-%H%M%S")

    conn = db.connect()
    rows = conn.execute("SELECT project_id, vector FROM embeddings").fetchall()
    if not rows:
        conn.close()
        raise HTTPException(400, "No embeddings to back up")

    data = pickle.dumps([(r[0], bytes(r[1])) for r in rows])
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO vector_backups (name, model, created_at, data) VALUES (?, ?, ?, ?)",
        (name, model, created_at, data)
    )
    conn.commit()
    backup_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"id": backup_id, "name": name, "model": model, "created_at": created_at, "count": len(rows)}


@app.get("/api/backups")
async def list_backups():
    conn = db.connect()
    db.init(conn)
    rows = conn.execute(
        "SELECT id, name, model, created_at FROM vector_backups ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return {"backups": [{"id": r[0], "name": r[1], "model": r[2], "created_at": r[3]} for r in rows]}


@app.post("/api/backups/{backup_id}/restore")
async def restore_backup(backup_id: int):
    conn = db.connect()
    row = conn.execute("SELECT name, model, data FROM vector_backups WHERE id = ?", (backup_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Backup not found")

    entries = pickle.loads(bytes(row[2]))
    conn.execute("DELETE FROM embeddings")
    conn.execute("UPDATE projects SET embedded_at = NULL")
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (project_id, vector) VALUES (?, ?)",
        entries
    )
    ids = [e[0] for e in entries]
    if ids:
        conn.execute(
            f"UPDATE projects SET embedded_at = ? WHERE id IN ({','.join('?'*len(ids))})",
            [now] + ids
        )
    conn.commit()
    conn.close()
    return {"restored": len(entries), "from": row[0], "model": row[1]}


# ── API: search ───────────────────────────────────────────────────────────────

@app.get("/api/search")
async def search(
    query: str = Query(..., min_length=1),
    loader: str = Query(...),
    version: str = Query(...),
    top: int = Query(default=10, ge=1, le=50),
    pool: int = Query(default=50, ge=10, le=200),
):
    cfg = config.load()
    embed_url = cfg["inference"]["embedding_url"].rstrip("/")
    rerank_url = cfg["inference"]["reranker_url"].rstrip("/")

    conn = db.connect()
    rows = conn.execute("""
        SELECT p.id, p.slug, p.title, p.description, p.author,
               p.downloads, p.client_side, p.server_side, e.vector
        FROM projects p
        JOIN embeddings e ON e.project_id = p.id
        WHERE p.loader = ? AND p.mc_version = ?
    """, (loader, version)).fetchall()
    conn.close()

    if not rows:
        return {"results": [], "error": f"No embedded mods found for {loader} {version}"}

    vectors = np.stack([_unpack(r["vector"]) for r in rows])
    query_vec = _embed(query, embed_url)
    scores = _cosine(query_vec, vectors)

    candidate_indices = np.argsort(scores)[::-1][:min(pool, len(rows))]
    candidates = [rows[i] for i in candidate_indices]

    docs = [f"{r['title']}\n{r['description'] or ''}" for r in candidates]
    rerank_scores = _rerank(query, docs, rerank_url)

    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:top]

    return {
        "results": [
            {
                "rank": i + 1,
                "id": r["id"],
                "slug": r["slug"],
                "title": r["title"],
                "description": r["description"] or "",
                "author": r["author"] or "",
                "downloads": r["downloads"],
                "side": _fmt_side(r["client_side"], r["server_side"]),
                "url": f"{MODRINTH_URL}/{r['slug']}",
                "score": round(score, 4),
            }
            for i, (r, score) in enumerate(ranked)
        ],
        "total_searched": len(rows),
    }
