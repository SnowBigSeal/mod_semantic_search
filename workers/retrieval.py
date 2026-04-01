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
        facets.append([f"modified_timestamp>={_to_unix(since)}"])

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


def _fetch_bodies(project_ids: list[str], user_agent: str) -> dict[str, str]:
    """Bulk-fetch full project bodies. Returns {id: body}."""
    bodies: dict[str, str] = {}
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
                for proj in resp.json():
                    bodies[proj["id"]] = proj.get("body", "")
                print(f"  fetched bodies {min(i + BATCH_SIZE, len(project_ids))} / {len(project_ids)}", end="\r")
                if i + BATCH_SIZE < len(project_ids):
                    limiter.wait(resp)
        except APIDeprecatedError as e:
            print(f"\n[error] {e}")
            raise

    print()
    return bodies


def _to_unix(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


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


def run(loader: str, version: str, sync: bool = False, test: bool = False) -> None:
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
    bodies = _fetch_bodies(project_ids, user_agent)

    print("Writing to database...")
    upserted = _upsert_projects(conn, hits, bodies, loader, version)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_sync', ?)", (now,)
    )
    conn.commit()
    conn.close()

    print(f"Done. {upserted} new/updated mods stored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch mods from Modrinth")
    parser.add_argument("--loader", required=True, help="Mod loader (e.g. neoforge, fabric)")
    parser.add_argument("--version", required=True, help="Minecraft version (e.g. 1.21.1)")
    parser.add_argument("--sync", action="store_true", help="Only fetch mods changed since last sync")
    parser.add_argument("--test", action="store_true", help="Fetch only the first 50 mods (for testing)")
    args = parser.parse_args()
    run(args.loader, args.version, sync=args.sync, test=args.test)


if __name__ == "__main__":
    main()
