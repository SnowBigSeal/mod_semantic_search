"""
Embedding worker — generates vectors for mods that have no embedding yet.

Usage:
    python -m workers.embedding
    python -m workers.embedding --batch-size 32
"""

import argparse
import re
import struct
import time
from datetime import datetime, timezone
from typing import List

import httpx
import numpy as np
from bs4 import BeautifulSoup
from markdown import markdown

import config
import db

EMBED_ENDPOINT = "/v1/embeddings"


def _embed_batch(texts: List[str], base_url: str, client: httpx.Client) -> List[List[float]]:
    resp = client.post(
        f"{base_url}{EMBED_ENDPOINT}",
        json={"input": texts},
        timeout=120,
    )
    if not resp.is_success:
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}: {resp.text[:200]}",
            request=resp.request,
            response=resp,
        )
    data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32)


# Per-text token limit for the embedding server.
# Each text is processed independently - do NOT divide by batch_size.
# Set to the server's --ctx-size (llama.cpp default is 512; raise if you launched with more).
SERVER_CTX_TOKENS = 512
CHARS_PER_TOKEN = 4
CTX_SAFETY = 0.85

def _clean_body(body: str) -> str:
    """Convert markdown/HTML body to plain text, stripping formatting noise."""
    if not body:
        return ""
    # Render markdown → HTML, then strip tags
    html = markdown(body, extensions=["extra"])
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _max_chars_per_text() -> int:
    return int(SERVER_CTX_TOKENS * CHARS_PER_TOKEN * CTX_SAFETY)

def _build_text(row, max_chars: int) -> str:
    title = (row["title"] or "").strip()
    desc  = (row["description"] or "").strip()
    body  = _clean_body(row["body"] or "")
    text  = " ".join(p for p in [title, desc, body] if p)
    return text[:max_chars]


def run(batch_size: int = 32, re_embed: bool = False, parallel: int = 4) -> None:
    cfg = config.load()
    base_url = cfg["inference"]["embedding_url"].rstrip("/")

    conn = db.connect()
    db.init(conn)

    if re_embed:
        print("Clearing embedded_at for full re-embed...")
        conn.execute("UPDATE projects SET embedded_at = NULL")
        conn.execute("DELETE FROM embeddings")
        conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM projects WHERE embedded_at IS NULL"
    ).fetchone()[0]

    if total == 0:
        print("All mods are already embedded.")
        return

    print(f"Embedding {total} mods (batch size: {batch_size}, parallel: {parallel})")
    processed = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_batch(rows_batch):
        ids = [r["id"] for r in rows_batch]
        texts = [_build_text(r, _max_chars_per_text()) for r in rows_batch]
        with httpx.Client(timeout=120) as client:
            while True:
                try:
                    return ids, _embed_batch(texts, base_url, client)
                except (httpx.HTTPError, httpx.TimeoutException) as e:
                    status = getattr(getattr(e, 'response', None), 'status_code', None)
                    body = getattr(getattr(e, 'response', None), 'text', '')
                    msg = f"{type(e).__name__}: {e}" + (f" — {body[:300]}" if body else "")
                    wait = 30 if status == 503 else 10
                    print(f"\n[warn] batch failed ({msg}), retrying in {wait}s...")
                    time.sleep(wait)

    with httpx.Client(timeout=120) as _:
        # Prefetch all pending IDs so we can chunk into parallel batches
        while True:
            pending = conn.execute(
                "SELECT id, title, description, body FROM projects WHERE embedded_at IS NULL LIMIT ?",
                (batch_size * parallel,),
            ).fetchall()

            if not pending:
                break

            # Split into sub-batches, one per parallel slot
            sub_batches = [pending[i::parallel] for i in range(parallel) if pending[i::parallel]]

            results = {}
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(_process_batch, sb): sb for sb in sub_batches}
                for fut in as_completed(futures):
                    try:
                        ids, vectors = fut.result()
                        results[tuple(ids)] = vectors
                    except Exception as e:
                        body = getattr(getattr(e, 'response', None), 'text', '')
                        msg = str(e) + (f" — {body[:300]}" if body else "")
                        print(f"\n[error] batch failed, skipping: {msg}")

            now = datetime.now(timezone.utc).isoformat()
            for ids, vectors in results.items():
                for pid, vec in zip(ids, vectors):
                    blob = _pack_vector(vec)
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings (project_id, vector) VALUES (?, ?)",
                        (pid, blob),
                    )
                    conn.execute(
                        "UPDATE projects SET embedded_at = ? WHERE id = ?",
                        (now, pid),
                    )
            conn.commit()
            processed += len(pending)
            print(f"  embedded {processed} / {total}", end="\r")

    print(f"\nDone. {processed} mods embedded.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed mod descriptions")
    parser.add_argument("--batch-size", type=int, default=32, help="Texts per embedding request")
    parser.add_argument("--re-embed", action="store_true", help="Clear all existing embeddings and re-embed from scratch")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel llama.cpp slots to use (default: 4)")
    args = parser.parse_args()
    run(batch_size=args.batch_size, re_embed=args.re_embed, parallel=args.parallel)


if __name__ == "__main__":
    main()
