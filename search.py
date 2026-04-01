"""
Mod search CLI — semantic search over archived mod descriptions.

Usage:
    python search.py "tech and automation mods" --loader neoforge --version 1.21.1
    python search.py "magic spells and rituals" --loader fabric --version 1.21.1 --top 5
"""

import argparse
import struct

import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

import config
import db

MODRINTH_URL = "https://modrinth.com/mod"
RERANK_ENDPOINT = "/v1/rerank"
EMBED_ENDPOINT = "/v1/embeddings"

console = Console()


def _embed_query(query: str, base_url: str) -> np.ndarray:
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{base_url}{EMBED_ENDPOINT}",
            json={"input": [query]},
        )
        resp.raise_for_status()
        vec = resp.json()["data"][0]["embedding"]
    return np.array(vec, dtype=np.float32)


def _unpack_vector(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32).copy()


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normalized = matrix / norms
    return normalized @ query_norm


def _rerank(query: str, documents: list[str], base_url: str) -> list[float]:
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{base_url}{RERANK_ENDPOINT}",
            json={"query": query, "documents": documents},
        )
        resp.raise_for_status()
        results = resp.json()["results"]

    scores = [0.0] * len(documents)
    for item in results:
        scores[item["index"]] = item["relevance_score"]
    return scores


def run(query: str, loader: str, version: str, top_k: int, candidate_pool: int) -> None:
    cfg = config.load()
    embed_url = cfg["inference"]["embedding_url"].rstrip("/")
    rerank_url = cfg["inference"]["reranker_url"].rstrip("/")

    conn = db.connect()

    # Load all embeddings for this loader+version
    rows = conn.execute("""
        SELECT p.id, p.slug, p.title, p.description, p.author, p.downloads, p.client_side, p.server_side,
               e.vector
        FROM projects p
        JOIN embeddings e ON e.project_id = p.id
        WHERE p.loader = ? AND p.mc_version = ?
    """, (loader, version)).fetchall()

    if not rows:
        console.print(f"[red]No embedded mods found for {loader} {version}. Run retrieval + embedding workers first.[/red]")
        return

    console.print(f"Searching {len(rows)} mods for: [bold]{query}[/bold]")

    # Stage 1: cosine similarity to find candidates
    vectors = np.stack([_unpack_vector(r["vector"]) for r in rows])
    query_vec = _embed_query(query, embed_url)
    scores = _cosine_similarity(query_vec, vectors)

    # Pick top candidates
    pool = min(candidate_pool, len(rows))
    candidate_indices = np.argsort(scores)[::-1][:pool]
    candidates = [rows[i] for i in candidate_indices]

    # Stage 2: rerank candidates
    documents = [
        f"{r['title']}\n{r['description'] or ''}" for r in candidates
    ]
    rerank_scores = _rerank(query, documents, rerank_url)

    # Sort by rerank score and take top_k
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

    # Display results
    table = Table(title=f"Top {top_k} results for '{query}' ({loader} {version})", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Mod", style="bold cyan", min_width=20)
    table.add_column("Author", style="green", width=16)
    table.add_column("Downloads", justify="right", width=10)
    table.add_column("Side", width=10)
    table.add_column("Description", min_width=40)
    table.add_column("Link", style="blue")

    for rank, (row, score) in enumerate(ranked, start=1):
        side = _fmt_side(row["client_side"], row["server_side"])
        table.add_row(
            str(rank),
            row["title"],
            row["author"] or "",
            f"{row['downloads']:,}",
            side,
            (row["description"] or "")[:120],
            f"{MODRINTH_URL}/{row['slug']}",
        )

    console.print(table)
    conn.close()


def _fmt_side(client: str, server: str) -> str:
    parts = []
    if client in ("required", "optional"):
        parts.append("client")
    if server in ("required", "optional"):
        parts.append("server")
    return "+".join(parts) if parts else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search archived mods by natural language")
    parser.add_argument("query", help="What you're looking for (e.g. 'tech and automation')")
    parser.add_argument("--loader", required=True, help="Mod loader (e.g. neoforge)")
    parser.add_argument("--version", required=True, help="Minecraft version (e.g. 1.21.1)")
    parser.add_argument("--top", type=int, default=None, help="Number of results to show")
    parser.add_argument("--pool", type=int, default=None, help="Candidate pool size before reranking")
    args = parser.parse_args()

    cfg = config.load()
    top_k = args.top or cfg.get("search", {}).get("top_k", 10)
    pool = args.pool or cfg.get("search", {}).get("candidate_pool", 50)

    run(args.query, args.loader, args.version, top_k=top_k, candidate_pool=pool)


if __name__ == "__main__":
    main()
