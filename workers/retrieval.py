"""
Retrieval worker — fetches mods from Modrinth for a given loader + MC version.

Usage:
    python -m workers.retrieval --loader neoforge --version 1.21.1
    python -m workers.retrieval --loader fabric --version 1.21.1 --sync
"""

import json
import argparse
from datetime import datetime, timezone
from typing import Optional

import httpx

import config
import db
from workers.ratelimit import RateLimiter, checked_get, APIDeprecatedError

MODRINTH_BASE = "https://api.modrinth.com/v2"
PAGE_SIZE = 100
BATCH_SIZE = 50  # max IDs per /projects bulk request


def _headers(user_agent: str) -> dict:
    return {"User-Agent": user_agent}


def _search_all(loader: str, version: str, user_agent: str, since: Optional[str] = None, max_results: Optional[int] = None) -> list:
    """Page through Modrinth search, returning all hits for loader+version."""
    facets = [
        [f"categories:{loader}"],
        [f"versions:{version}"],
        ["project_type:mod"],
    ]
    if since:
        facets.append([f"date_modified>={since}"])

    params = {
        "facets": json.dumps(facets),
        "limit": PAGE_SIZE if not max_results else min(PAGE_SIZE, max_results),
        "offset": 0,
        "index": "updated",
    }

    all_hits: list[dict] = []
    headers = _headers(user_agent)
    limiter = RateLimiter()

    with httpx.Client(timeout=30) as client:
        try:
            while True:
                resp = checked_get(client, f"{MODRINTH_BASE}/search", params=params, headers=headers)
                data = resp.json()
                hits = data["hits"]
                all_hits.extend(hits)
                print(f"  fetched {len(all_hits)} / {data['total_hits']}", end="\r")
                if len(all_hits) >= data["total_hits"]:
                    break
                if max_results and len(all_hits) >= max_results:
                    break
                params["offset"] += PAGE_SIZE
                limiter.wait(resp)
        except APIDeprecatedError as e:
            print(f"\n[error] {e}")
            raise

    print()
    return all_hits[:max_results] if max_results else all_hits


APPROVED_STATUSES = {"approved", "unlisted"}


def _fetch_bodies(project_ids: list[str], user_agent: str) -> tuple:
    """Bulk-fetch full project bodies. Returns ({id: body}, set of missing/non-approved IDs)."""
    bodies: dict[str, str] = {}
    removed_ids: set[str] = set()
    headers = _headers(user_agent)
    limiter = RateLimiter()

    with httpx.Client(timeout=30) as client:
        try:
            for i in range(0, len(project_ids), BATCH_SIZE):
                batch = project_ids[i : i + BATCH_SIZE]
                resp = checked_get(
                    client,
                    f"{MODRINTH_BASE}/projects",
                    params={"ids": json.dumps(batch)},
                    headers=headers,
                )
                returned = resp.json()
                returned_ids = set()
                for proj in returned:
                    pid = proj["id"]
                    returned_ids.add(pid)
                    status = proj.get("status", "approved")
                    if status not in APPROVED_STATUSES:
                        removed_ids.add(pid)
                    else:
                        bodies[pid] = proj.get("body", "")

                # Any ID we requested but didn't get back is deleted/gone
                missing = set(batch) - returned_ids
                if missing:
                    print(f"\n  [warn] {len(missing)} project(s) not found in batch — will remove from DB")
                    removed_ids.update(missing)

                print(f"  fetched bodies {min(i + BATCH_SIZE, len(project_ids))} / {len(project_ids)}", end="\r")
                if i + BATCH_SIZE < len(project_ids):
                    limiter.wait(resp)
        except APIDeprecatedError as e:
            print(f"\n[error] {e}")
            raise

    print()
    return bodies, removed_ids

def _upsert_projects(conn, hits: list[dict], bodies: dict[str, str], loader: str, version: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    upserted = 0
    for hit in hits:
        pid = hit["project_id"]
        existing = conn.execute(
            "SELECT date_modified FROM projects WHERE id = ?", (pid,)
        ).fetchone()

        if existing and existing["date_modified"] == hit.get("date_modified"):
            continue

        conn.execute("""
            INSERT INTO projects
                (id, slug, title, description, body, loader, mc_version,
                 categories, downloads, follows, author, license,
                 client_side, server_side, date_created, date_modified, embedded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
            ON CONFLICT(id) DO UPDATE SET
                slug          = excluded.slug,
                title         = excluded.title,
                description   = excluded.description,
                body          = excluded.body,
                categories    = excluded.categories,
                downloads     = excluded.downloads,
                follows       = excluded.follows,
                date_modified = excluded.date_modified,
                embedded_at   = NULL
        """, (
            pid,
            hit.get("slug", ""),
            hit.get("title", ""),
            hit.get("description", ""),
            bodies.get(pid, ""),
            loader,
            version,
            json.dumps(hit.get("categories", [])),
            hit.get("downloads", 0),
            hit.get("follows", 0),
            hit.get("author", ""),
            hit.get("license", ""),
            hit.get("client_side", ""),
            hit.get("server_side", ""),
            hit.get("date_created", ""),
            hit.get("date_modified", ""),
        ))
        upserted += 1

    conn.commit()
    return upserted


def _prune_deleted(conn, loader: str, version: str, user_agent: str) -> int:
    """
    Use /project/{id}/check to verify every stored project still exists.
    Removes any that return 404 (deleted or de-listed).
    """
    rows = conn.execute(
        "SELECT id FROM projects WHERE loader = ? AND mc_version = ?", (loader, version)
    ).fetchall()
    if not rows:
        return 0

    headers = _headers(user_agent)
    removed = 0
    print(f"Pruning: checking {len(rows)} stored projects for deletion...")

    with httpx.Client(timeout=30) as client:
        limiter = RateLimiter()
        for row in rows:
            pid = row["id"]
            try:
                resp = client.get(f"{MODRINTH_BASE}/project/{pid}/check", headers=headers)
                if resp.status_code == 404:
                    conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
                    conn.execute("DELETE FROM embeddings WHERE project_id = ?", (pid,))
                    removed += 1
                elif resp.status_code == 410:
                    raise APIDeprecatedError(
                        "Modrinth returned HTTP 410 — API version deprecated. Please update."
                    )
                limiter.wait(resp)
            except APIDeprecatedError:
                raise
            except httpx.HTTPError as e:
                print(f"\n  [warn] check failed for {pid}: {e}")

    conn.commit()
    return removed


def run(loader: str, version: str, sync: bool = False, test: bool = False, prune: bool = False) -> None:
    cfg = config.load()
    user_agent = cfg["modrinth"]["user_agent"]

    conn = db.connect()
    db.init(conn)

    since = None
    if sync:
        row = conn.execute("SELECT value FROM meta WHERE key = 'last_sync'").fetchone()
        if row:
            since = row["value"]
            print(f"Syncing changes since {since}")
        else:
            print("No previous sync found — performing full fetch")

    if test:
        print("[test mode] Limiting to first 50 mods")

    print(f"Searching Modrinth: loader={loader}, version={version}")
    hits = _search_all(loader, version, user_agent, since=since, max_results=50 if test else None)
    print(f"Found {len(hits)} mods")

    if test:
        print(f"[test mode] Using {len(hits)} mods")

    if not hits:
        print("Nothing to do.")
        return

    project_ids = [h["project_id"] for h in hits]
    print("Fetching full project bodies...")
    bodies, removed_ids = _fetch_bodies(project_ids, user_agent)

    if removed_ids:
        placeholders = ",".join("?" * len(removed_ids))
        conn.execute(f"DELETE FROM projects WHERE id IN ({placeholders})", list(removed_ids))
        conn.execute(f"DELETE FROM embeddings WHERE project_id IN ({placeholders})", list(removed_ids))
        conn.commit()
        print(f"  removed {len(removed_ids)} deleted/non-approved project(s) from DB")

    # Only upsert hits that survived the body fetch (approved + still exist)
    hits = [h for h in hits if h["project_id"] in bodies]

    print("Writing to database...")
    upserted = _upsert_projects(conn, hits, bodies, loader, version)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_sync', ?)", (now,)
    )
    conn.commit()

    print(f"Done. {upserted} new/updated mods stored.")

    if prune:
        pruned = _prune_deleted(conn, loader, version, user_agent)
        if pruned:
            print(f"Pruned {pruned} deleted project(s) from DB.")
        else:
            print("Prune complete — no deleted projects found.")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch mods from Modrinth")
    parser.add_argument("--loader", required=True, help="Mod loader (e.g. neoforge, fabric)")
    parser.add_argument("--version", required=True, help="Minecraft version (e.g. 1.21.1)")
    parser.add_argument("--sync", action="store_true", help="Only fetch mods changed since last sync")
    parser.add_argument("--test", action="store_true", help="Fetch only the first 50 mods (for testing)")
    parser.add_argument("--prune", action="store_true", help="Check all stored projects still exist on Modrinth and remove any that don't")
    args = parser.parse_args()
    run(args.loader, args.version, sync=args.sync, test=args.test, prune=args.prune)


if __name__ == "__main__":
    main()
