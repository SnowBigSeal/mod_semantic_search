"""
Embedding worker — generates vectors for mods that have no embedding yet.

Usage:
    python -m workers.embedding
    python -m workers.embedding --batch-size 32
"""

import argparse
import struct
from datetime import datetime, timezone

import httpx
import numpy as np

import config
import db

EMBED_ENDPOINT = "/v1/embeddings"


def _embed_batch(texts: list[str], base_url: str, client: httpx.Client) -> list[list[float]]:
    resp = client.post(
        f"{base_url}{EMBED_ENDPOINT}",
        json={"input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32)


def _build_text(row) -> str:
    parts = [row["title"] or "", row["description"] or "", row["body"] or ""]
    return " ".join(p.strip() for p in parts if p.strip())


def run(batch_size: int = 16) -> None:
    cfg = config.load()
    base_url = cfg["inference"]["embedding_url"].rstrip("/")

    conn = db.connect()
    db.init(conn)

    total = conn.execute(
        "SELECT COUNT(*) FROM projects WHERE embedded_at IS NULL"
    ).fetchone()[0]

    if total == 0:
        print("All mods are already embedded.")
        return

    print(f"Embedding {total} mods (batch size: {batch_size})")
    processed = 0

    with httpx.Client(timeout=120) as client:
        while True:
            rows = conn.execute(
                "SELECT id, title, description, body FROM projects WHERE embedded_at IS NULL LIMIT ?",
                (batch_size,),
            ).fetchall()

            if not rows:
                break

            ids = [r["id"] for r in rows]
            texts = [_build_text(r) for r in rows]

            try:
                vectors = _embed_batch(texts, base_url, client)
            except httpx.HTTPError as e:
                print(f"[error] embedding request failed: {e}")
                break

            now = datetime.now(timezone.utc).isoformat()
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
            processed += len(ids)
            print(f"  embedded {processed} / {total}", end="\r")

    print(f"\nDone. {processed} mods embedded.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed mod descriptions")
    parser.add_argument("--batch-size", type=int, default=16, help="Texts per embedding request")
    args = parser.parse_args()
    run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
