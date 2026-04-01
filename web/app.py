import io
import pickle
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
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

# ── job state ─────────────────────────────────────────────────────────────────

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _new_job(job_type: str, args: dict) -> str:
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "args": args,
            "status": "running",
            "log": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
    return job_id

def _job_running(job_type: str) -> bool:
    with _jobs_lock:
        return any(j["type"] == job_type and j["status"] == "running" for j in _jobs.values())

def _finish_job(job_id: str, success: bool, msg: str = "") -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done" if success else "error"
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            if msg:
                _jobs[job_id]["log"].append(msg)

def _log(job_id: str, line: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["log"].append(line)


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


# ── background workers ────────────────────────────────────────────────────────

def _run_retrieval(job_id: str, loader: str, version: str, sync: bool) -> None:
    try:
        import sys, io as _io
        buf = _io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        from workers.retrieval import run as retrieval_run
        retrieval_run(loader=loader, version=version, sync=sync)

        sys.stdout = old_stdout
        for line in buf.getvalue().splitlines():
            _log(job_id, line)
        _finish_job(job_id, True)
    except Exception as e:
        import sys as _sys
        _sys.stdout = _sys.__stdout__
        _finish_job(job_id, False, f"Error: {e}")


def _run_embedding(job_id: str, batch_size: int) -> None:
    try:
        import sys, io as _io
        buf = _io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        from workers.embedding import run as embedding_run
        embedding_run(batch_size=batch_size)

        sys.stdout = old_stdout
        for line in buf.getvalue().splitlines():
            _log(job_id, line)
        _finish_job(job_id, True)
    except Exception as e:
        import sys as _sys
        _sys.stdout = _sys.__stdout__
        _finish_job(job_id, False, f"Error: {e}")


def _run_pipeline(job_id: str, loader: str, version: str, sync: bool, batch_size: int) -> None:
    try:
        import sys, io as _io

        for label, fn, kwargs in [
            ("retrieval", _run_retrieval, {"loader": loader, "version": version, "sync": sync}),
            ("embedding", _run_embedding, {"batch_size": batch_size}),
        ]:
            _log(job_id, f"=== Starting {label} ===")
            buf = _io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf

            if label == "retrieval":
                from workers.retrieval import run as retrieval_run
                retrieval_run(loader=loader, version=version, sync=sync)
            else:
                from workers.embedding import run as embedding_run
                embedding_run(batch_size=batch_size)

            sys.stdout = old_stdout
            for line in buf.getvalue().splitlines():
                _log(job_id, line)

        _finish_job(job_id, True)
    except Exception as e:
        import sys as _sys
        _sys.stdout = _sys.__stdout__
        _finish_job(job_id, False, f"Error: {e}")


def _run_prune(job_id: str, loader: str, version: str) -> None:
    try:
        import sys, io as _io
        buf = _io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        from workers.retrieval import run as retrieval_run
        retrieval_run(loader=loader, version=version, prune=True)

        sys.stdout = old_stdout
        for line in buf.getvalue().splitlines():
            _log(job_id, line)
        _finish_job(job_id, True)
    except Exception as e:
        import sys as _sys
        _sys.stdout = _sys.__stdout__
        _finish_job(job_id, False, f"Error: {e}")


def _run_reset(job_id: str) -> None:
    try:
        conn = db.connect()
        conn.execute("DELETE FROM embeddings")
        conn.execute("UPDATE projects SET embedded_at = NULL")
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        conn.close()
        _log(job_id, f"Reset complete. {total} mods queued for re-embedding.")
        _finish_job(job_id, True)
    except Exception as e:
        _finish_job(job_id, False, f"Error: {e}")


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


# ── API: jobs ─────────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    action: str
    loader: Optional[str] = None
    version: Optional[str] = None
    sync: Optional[bool] = False
    batch_size: Optional[int] = 32

@app.post("/api/jobs/start")
async def start_job(req: JobRequest):
    action = req.action

    CONFLICT_GROUPS = {
        "retrieval": "retrieval",
        "embedding": "embedding",
        "pipeline": "pipeline",
        "prune": "retrieval",
        "reset": "embedding",
    }
    group = CONFLICT_GROUPS.get(action, action)
    if _job_running(group):
        raise HTTPException(409, f"A {group} job is already running")

    job_id = _new_job(group, req.dict())

    if action == "retrieval":
        t = threading.Thread(target=_run_retrieval, args=(job_id, req.loader, req.version, req.sync), daemon=True)
    elif action == "embedding":
        t = threading.Thread(target=_run_embedding, args=(job_id, req.batch_size), daemon=True)
    elif action == "pipeline":
        t = threading.Thread(target=_run_pipeline, args=(job_id, req.loader, req.version, req.sync, req.batch_size), daemon=True)
    elif action == "prune":
        t = threading.Thread(target=_run_prune, args=(job_id, req.loader, req.version), daemon=True)
    elif action == "reset":
        t = threading.Thread(target=_run_reset, args=(job_id,), daemon=True)
    else:
        raise HTTPException(400, f"Unknown action: {action}")

    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs/status")
async def jobs_status():
    with _jobs_lock:
        # Return most recent 20 jobs
        recent = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)[:20]
        return {"jobs": [dict(j) for j in recent]}


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
