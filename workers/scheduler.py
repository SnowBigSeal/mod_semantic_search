"""
Scheduler — runs a daily incremental sync from Modrinth, then re-embeds any new/changed mods.

Usage:
    python -m workers.scheduler --loader neoforge --version 1.21.1
    python -m workers.scheduler --loader neoforge --version 1.21.1 --interval 3600
"""

import argparse
import time

from workers.retrieval import run as retrieval_run
from workers.embedding import run as embedding_run


def run(loader: str, version: str, interval_seconds: int = 86400) -> None:
    print(f"Scheduler started — syncing every {interval_seconds}s")
    print(f"Target: loader={loader}, version={version}")
    print("Press Ctrl+C to stop.\n")

    while True:
        print(f"[sync] {_now()} — starting incremental sync")
        try:
            retrieval_run(loader, version, sync=True)
            embedding_run()
        except Exception as e:
            print(f"[sync] error during sync: {e}")

        print(f"[sync] done — sleeping {interval_seconds}s\n")
        time.sleep(interval_seconds)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily sync scheduler")
    parser.add_argument("--loader", required=True, help="Mod loader (e.g. neoforge)")
    parser.add_argument("--version", required=True, help="Minecraft version (e.g. 1.21.1)")
    parser.add_argument(
        "--interval",
        type=int,
        default=86400,
        help="Sync interval in seconds (default: 86400 = 24h)",
    )
    args = parser.parse_args()
    run(args.loader, args.version, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
