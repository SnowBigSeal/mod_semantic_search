import asyncio
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
    global _CATEGORY_LIST
    conn = db.connect()
    db.init(conn)
    conn.close()
    if not _runner_alive():
        print("[web] Runner not detected — spawning workers.runner")
        _spawn_runner()
    # Load or fetch Modrinth category list
    cats = _load_categories()
    if cats:
        _CATEGORY_LIST = cats
        print(f"[web] Loaded {len(cats)} mod categories from DB")
    else:
        try:
            cfg = config.load()
            user_agent = cfg.get("modrinth", {}).get("user_agent", "mod-search/1.0")
            _CATEGORY_LIST = _fetch_and_store_categories(user_agent)
            print(f"[web] Fetched {len(_CATEGORY_LIST)} mod categories from Modrinth")
        except Exception as exc:
            print(f"[web] Failed to fetch categories: {exc}")


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
        if not r.is_success:
            raise HTTPException(502, f"Embedding server error {r.status_code}: {r.text[:200]}")
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)

def _rerank(query: str, docs: list, base_url: str) -> list:
    """
    Qwen3-Reranker scoring via /v1/completions with logprobs.
    Formats each query-doc pair with the Qwen3 chat template, forces the
    model to predict yes/no, and returns softmax(yes, no) as the score.
    Requests are fired in parallel (up to 16 at a time).
    """
    import math
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _SYS = (
        "Judge whether the Document meets the requirements based on the "
        "Query and the Instruct, and give a judgment result of yes or no."
    )
    _INSTR = "Given a Minecraft mod search query, determine if this mod matches what the player is looking for."
    _URL = f"{base_url}/v1/completions"

    def _score_one(idx: int, doc: str):
        prompt = (
            f"<|im_start|>system\n{_SYS}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {_INSTR}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        with httpx.Client(timeout=60) as client:
            r = client.post(_URL, json={"prompt": prompt, "max_tokens": 1, "logprobs": 5, "temperature": 0})
            r.raise_for_status()
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        # Gather logprobs for yes/no variants
        lp = {t["token"].lower(): t["logprob"] for t in top}
        yes_lp = lp.get("yes", -20.0)
        no_lp  = lp.get("no",  -20.0)
        # Softmax over just yes/no
        yes_p = math.exp(yes_lp) / (math.exp(yes_lp) + math.exp(no_lp))
        return idx, yes_p

    scores = [0.0] * len(docs)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_score_one, i, doc): i for i, doc in enumerate(docs)}
        for fut in as_completed(futures):
            try:
                idx, score = fut.result()
                scores[idx] = score
            except Exception as exc:
                print(f"[rerank] doc {futures[fut]} failed: {exc}")
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

def _pack(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

def _cache_lookup(query_vec: np.ndarray, loader: str, version: str, threshold: float):
    """Return (cache_id, [{id, score}]) if a similar query exists, else (None, None)."""
    conn = db.connect()
    rows = conn.execute(
        "SELECT rowid AS id, query_vec, results FROM query_cache WHERE loader = ? AND version = ?",
        (loader, version),
    ).fetchall()
    conn.close()
    if not rows:
        return None, None
    vecs = np.stack([_unpack(r["query_vec"]) for r in rows])
    if vecs.shape[1] != query_vec.shape[0]:
        print(f"[cache] dimension mismatch ({vecs.shape[1]} stored vs {query_vec.shape[0]} current) — clearing cache")
        conn = db.connect()
        conn.execute("DELETE FROM query_cache")
        conn.commit()
        conn.close()
        return None, None
    sims = _cosine(query_vec, vecs)
    best = int(np.argmax(sims))
    if sims[best] >= threshold:
        return rows[best]["id"], json.loads(rows[best]["results"])
    return None, None

def _cache_store(query_text: str, query_vec: np.ndarray, loader: str, version: str, ranked: list) -> None:
    """Store only id + score per result — project data is fetched fresh on hit."""
    slim = [{"id": r["id"], "score": r["score"]} for r in ranked]
    conn = db.connect()
    conn.execute(
        "INSERT INTO query_cache (query_text, loader, version, query_vec, results) VALUES (?, ?, ?, ?, ?)",
        (query_text, loader, version, _pack(query_vec), json.dumps(slim)),
    )
    conn.commit()
    conn.close()


# ── category list (fetched once from Modrinth, stored in meta) ────────────────

_CATEGORY_LIST: List[str] = []

def _load_categories() -> List[str]:
    conn = db.connect()
    row = conn.execute("SELECT value FROM meta WHERE key = 'categories'").fetchone()
    conn.close()
    if row:
        return json.loads(row["value"])
    return []

def _fetch_and_store_categories(user_agent: str) -> List[str]:
    with httpx.Client(timeout=15) as client:
        r = client.get(
            "https://api.modrinth.com/v2/tag/category",
            headers={"User-Agent": user_agent},
        )
        r.raise_for_status()
    cats = [c["name"] for c in r.json() if c.get("project_type") == "mod"]
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('categories', ?)",
        (json.dumps(cats),),
    )
    conn.commit()
    conn.close()
    return cats


# ── query expansion ───────────────────────────────────────────────────────────

def _expand_query(query: str, category_list: List[str], chat_url: str) -> dict:
    """
    Use Qwen3 chat to expand the query into:
      - cleaned: a cleaner/expanded version for embedding
      - keywords: terms for FTS5 + LIKE matching
      - tags: matching Modrinth categories from the official list
    Returns dict with those keys (falls back gracefully on failure).
    Hard timeout: 5s — if the chat model is busy, fall back immediately.
    """
    cats_str = ", ".join(category_list)
    prompt = (
        f"You are helping improve a Minecraft mod search engine.\n\n"
        f"User query: \"{query}\"\n\n"
        f"Official Modrinth mod categories: {cats_str}\n\n"
        f"Respond with a JSON object (no markdown, no explanation) with exactly these keys:\n"
        f"  cleaned: a cleaner, more descriptive version of the query for semantic search\n"
        f"  keywords: array of 3-6 short search terms that would match relevant mod titles/descriptions\n"
        f"  tags: array of 0-3 category names from the official list above that best match this query\n\n"
        f"Example output: {{\"cleaned\": \"inventory management and item storage\", "
        f"\"keywords\": [\"storage\", \"inventory\", \"chest\", \"items\"], \"tags\": [\"storage\", \"utility\"]}}"
    )
    try:
        with httpx.Client(timeout=5) as client:
            r = client.post(
                f"{chat_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0,
                },
            )
            r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        return {
            "cleaned": str(parsed.get("cleaned", query)),
            "keywords": [str(k) for k in parsed.get("keywords", [])],
            "tags": [str(t) for t in parsed.get("tags", []) if t in category_list],
        }
    except Exception as exc:
        print(f"[expand] failed ({type(exc).__name__}): {exc}")
        return {"cleaned": query, "keywords": query.lower().split(), "tags": []}


# ── FTS5 keyword search ───────────────────────────────────────────────────────

def _fts_search(keywords: List[str], loader: str, version: str, conn) -> List[str]:
    """BM25 ranked mod IDs matching any keyword via FTS5."""
    if not keywords:
        return []
    import re
    # Strip non-alphanumeric chars — FTS5 treats ? * " etc. as syntax
    clean_kws = [re.sub(r'[^\w\s]', '', k).strip() for k in keywords]
    clean_kws = [k for k in clean_kws if k]
    if not clean_kws:
        return []
    fts_query = " OR ".join(f'"{k}"' for k in clean_kws)
    try:
        rows = conn.execute(
            """
            SELECT p.id FROM projects_fts f
            JOIN projects p ON p.id = f.id
            WHERE projects_fts MATCH ?
              AND p.loader = ? AND p.mc_version = ?
            ORDER BY rank
            LIMIT 100
            """,
            (fts_query, loader, version),
        ).fetchall()
        return [r["id"] for r in rows]
    except Exception as exc:
        print(f"[fts5] query failed: {exc}")
        return []


# ── tag search ────────────────────────────────────────────────────────────────

def _tag_search(tags: List[str], loader: str, version: str, conn, limit: int = 200) -> List[str]:
    """Mod IDs whose categories JSON contains any of the matched tags (capped to avoid broad tags swamping RRF)."""
    if not tags:
        return []
    results = []
    seen = set()
    for tag in tags:
        rows = conn.execute(
            "SELECT id FROM projects WHERE loader=? AND mc_version=? AND categories LIKE ? LIMIT ?",
            (loader, version, f"%{tag}%", limit),
        ).fetchall()
        for r in rows:
            if r["id"] not in seen:
                results.append(r["id"])
                seen.add(r["id"])
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    """Merge multiple ranked ID lists using RRF. Returns IDs sorted by fused score."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


@app.get("/api/categories")
async def get_categories():
    return {"categories": _CATEGORY_LIST}


async def get_cache_entry(cache_id: int):
    conn = db.connect()
    row = conn.execute(
        "SELECT rowid AS id, query_text, loader, version, created_at, results FROM query_cache WHERE rowid = ?",
        (cache_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Cache entry not found")
    return {
        "id": row["id"],
        "query_text": row["query_text"],
        "loader": row["loader"],
        "version": row["version"],
        "created_at": row["created_at"],
        "results": json.loads(row["results"]),
    }

@app.get("/api/search")
async def search(
    query: str = Query(..., min_length=1),
    loader: str = Query(...),
    version: str = Query(...),
    top: int = Query(default=10, ge=1, le=100),
    pool: int = Query(default=25, ge=10, le=200),
):
    cfg = config.load()
    embed_url = cfg["inference"]["embedding_url"].rstrip("/")
    rerank_url = cfg["inference"]["reranker_url"].rstrip("/")
    chat_url   = cfg["inference"]["chat_url"].rstrip("/")
    threshold  = float(cfg.get("search", {}).get("cache_threshold", 0.97))
    use_expand = str(cfg.get("search", {}).get("expansion", "true")).lower() == "true"
    pool = max(pool, top)  # pool must always be at least as large as the requested results

    # ── query expansion (run in thread so it doesn't block the event loop) ───
    import time
    t0 = time.time()
    if use_expand and _CATEGORY_LIST:
        loop = asyncio.get_event_loop()
        expansion = await loop.run_in_executor(
            None, _expand_query, query, _CATEGORY_LIST, chat_url
        )
    else:
        expansion = {"cleaned": query, "keywords": [], "tags": []}

    t_expand = time.time()
    print(f"[search] expand={t_expand-t0:.2f}s  cleaned='{expansion['cleaned']}'  keywords={expansion['keywords']}  tags={expansion['tags']}")

    cleaned   = expansion["cleaned"]
    keywords  = expansion["keywords"]
    tags      = expansion["tags"]

    # ── embed cleaned query ───────────────────────────────────────────────────
    query_vec = _embed(cleaned, embed_url)
    t_embed = time.time()
    print(f"[search] embed={t_embed-t_expand:.2f}s")

    # ── cache lookup (on cleaned query) ──────────────────────────────────────
    matched_id, cached = _cache_lookup(query_vec, loader, version, threshold)
    if cached is not None:
        ids = [e["id"] for e in cached]
        score_map = {e["id"]: e["score"] for e in cached}
        placeholders = ",".join("?" * len(ids))
        conn = db.connect()
        proj_rows = conn.execute(
            f"SELECT id, slug, title, description, author, downloads, client_side, server_side "
            f"FROM projects WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        conn.close()
        proj_map = {r["id"]: r for r in proj_rows}
        results = []
        for i, entry in enumerate(cached[:top]):
            r = proj_map.get(entry["id"])
            if not r:
                continue
            results.append({
                "rank": i + 1,
                "id": r["id"], "slug": r["slug"], "title": r["title"],
                "description": r["description"] or "",
                "author": r["author"] or "", "downloads": r["downloads"],
                "side": _fmt_side(r["client_side"], r["server_side"]),
                "url": f"{MODRINTH_URL}/{r['slug']}",
                "score": score_map[r["id"]],
            })
        return {
            "results": results, "total_searched": None, "cache_hit": True,
            "query": query, "cache_id": matched_id,
            "expansion": expansion,
        }

    # ── full pipeline ─────────────────────────────────────────────────────────
    conn = db.connect()
    rows = conn.execute("""
        SELECT p.id, p.slug, p.title, p.description, p.body, p.author,
               p.downloads, p.client_side, p.server_side, e.vector
        FROM projects p
        JOIN embeddings e ON e.project_id = p.id
        WHERE p.loader = ? AND p.mc_version = ?
    """, (loader, version)).fetchall()

    if not rows:
        conn.close()
        return {"results": [], "error": f"No embedded mods found for {loader} {version}"}

    row_index = {r["id"]: i for i, r in enumerate(rows)}

    # [2a] Dense cosine top-N
    vectors = np.stack([_unpack(r["vector"]) for r in rows])
    cos_scores = _cosine(query_vec, vectors)
    dense_ranked = [rows[i]["id"] for i in np.argsort(cos_scores)[::-1][:pool].tolist()]

    # [2b] FTS5 BM25
    fts_ranked = _fts_search(keywords, loader, version, conn)

    # [2c] Exact LIKE on keywords + original query terms
    like_terms = list({t.strip().lower() for t in (keywords + query.split()) if len(t.strip()) > 2})
    like_ids: List[str] = []
    seen_like: set = set()
    for term in like_terms:
        like = f"%{term}%"
        hits = conn.execute(
            "SELECT id FROM projects WHERE loader=? AND mc_version=? AND (LOWER(title) LIKE ? OR LOWER(slug) LIKE ?)",
            (loader, version, like, like),
        ).fetchall()
        for h in hits:
            if h["id"] not in seen_like:
                like_ids.append(h["id"])
                seen_like.add(h["id"])

    # [2d] Tag filter
    tag_ranked = _tag_search(tags, loader, version, conn)
    conn.close()

    # [3] RRF merge — cap total candidates before reranking
    merged_ids = _rrf([dense_ranked, fts_ranked, like_ids, tag_ranked])[:pool]
    t_retrieve = time.time()
    print(f"[search] retrieve={t_retrieve-t_embed:.2f}s  dense={len(dense_ranked)} fts5={len(fts_ranked)} like={len(like_ids)} tags={len(tag_ranked)} merged={len(merged_ids)}")

    # Resolve to row objects (only those with embeddings)
    candidates = [rows[row_index[mid]] for mid in merged_ids if mid in row_index]

    # ── rerank ────────────────────────────────────────────────────────────────
    docs = [
        f"{r['title']}\n{r['description'] or ''}".strip()
        for r in candidates
    ]
    rerank_scores = _rerank(cleaned, docs, rerank_url)
    t_rerank = time.time()
    print(f"[search] rerank={t_rerank-t_retrieve:.2f}s  candidates={len(candidates)}")

    if all(s == 0.0 for s in rerank_scores):
        print("[rerank] All scores zero — falling back to cosine")
        cos_map = {rows[i]["id"]: float(cos_scores[i]) for i in range(len(rows))}
        rerank_scores = [cos_map.get(r["id"], 0.0) for r in candidates]

    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    results = [
        {
            "rank": i + 1,
            "id": r["id"], "slug": r["slug"], "title": r["title"],
            "description": r["description"] or "",
            "author": r["author"] or "", "downloads": r["downloads"],
            "side": _fmt_side(r["client_side"], r["server_side"]),
            "url": f"{MODRINTH_URL}/{r['slug']}",
            "score": round(score, 4),
        }
        for i, (r, score) in enumerate(ranked)
    ]

    _cache_store(cleaned, query_vec, loader, version, results)

    t_total = time.time()
    print(f"[search] total={t_total-t0:.2f}s  results={len(results[:top])}")
    debug = {
        "timings": {
            "expand_s": round(t_expand - t0, 2),
            "embed_s": round(t_embed - t_expand, 2),
            "retrieve_s": round(t_retrieve - t_embed, 2),
            "rerank_s": round(t_rerank - t_retrieve, 2),
            "total_s": round(t_total - t0, 2),
        },
        "candidates": {
            "dense": len(dense_ranked),
            "fts5": len(fts_ranked),
            "like": len(like_ids),
            "tags": len(tag_ranked),
            "merged": len(merged_ids),
            "reranked": len(candidates),
        },
    }
    return {
        "results": results[:top],
        "total_searched": len(rows),
        "cache_hit": False,
        "query": query,
        "expansion": expansion,
        "debug": debug,
    }
