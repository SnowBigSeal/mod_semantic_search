import struct
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import db

app = FastAPI(title="Mod Semantic Search")

_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
templates = Jinja2Templates(directory=_BASE / "templates")

MODRINTH_URL = "https://modrinth.com/mod"
EMBED_ENDPOINT = "/v1/embeddings"
RERANK_ENDPOINT = "/v1/rerank"


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conn = db.connect()
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


@app.get("/api/status")
async def status():
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    last_sync = conn.execute(
        "SELECT value FROM meta WHERE key = 'last_sync'"
    ).fetchone()
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
