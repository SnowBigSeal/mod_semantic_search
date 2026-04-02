"""
Worker runner — polls the jobs table and executes queued jobs.

Start with:  python -m workers.runner
The web process will also spawn this automatically on startup.
"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import Set

import db

# Conflict groups — jobs in the same group cannot run concurrently.
# "pipeline" is special: it blocks both the retrieval AND embedding groups.
_GROUPS = {
    "retrieval": {"retrieval"},
    "prune":     {"retrieval"},
    "embedding": {"embedding"},
    "reset":     {"embedding"},
    "pipeline":  {"retrieval", "embedding"},
}

HEARTBEAT_INTERVAL = 5   # seconds between heartbeat writes
POLL_INTERVAL      = 1   # seconds between queue polls


class _DBLogger:
    """Captures stdout line-by-line and writes each line to job_logs."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._buf = ""

    def write(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._flush(line)

    def flush(self) -> None:
        pass

    def _flush(self, line: str) -> None:
        try:
            conn = db.connect()
            conn.execute(
                "INSERT INTO job_logs (job_id, line) VALUES (?, ?)",
                (self._job_id, line),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


def _running_groups(conn) -> Set[str]:
    rows = conn.execute(
        "SELECT type FROM jobs WHERE status = 'running'"
    ).fetchall()
    result: Set[str] = set()
    for row in rows:
        result |= _GROUPS.get(row["type"], {row["type"]})
    return result


def _claim_job(conn, job_id: str) -> bool:
    """Atomically transition a job from pending → running. Returns True if claimed."""
    cur = conn.execute(
        "UPDATE jobs SET status = 'running', started_at = datetime('now') "
        "WHERE id = ? AND status = 'pending'",
        (job_id,),
    )
    conn.commit()
    return cur.rowcount == 1


def _finish_job(conn, job_id: str, success: bool) -> None:
    status = "done" if success else "error"
    conn.execute(
        "UPDATE jobs SET status = ?, finished_at = datetime('now') WHERE id = ?",
        (status, job_id),
    )
    conn.commit()


def _run_job(job_id: str, job_type: str, args: dict) -> None:
    old_stdout = sys.stdout
    sys.stdout = _DBLogger(job_id)
    success = False
    try:
        if job_type == "retrieval":
            from workers.retrieval import run as retrieval_run
            retrieval_run(
                loader=args.get("loader", "neoforge"),
                version=args.get("version", "1.21.1"),
                sync=args.get("sync", False),
            )
        elif job_type == "embedding":
            from workers.embedding import run as embedding_run
            embedding_run(batch_size=args.get("batch_size", 32))
        elif job_type == "pipeline":
            from workers.retrieval import run as retrieval_run
            from workers.embedding import run as embedding_run
            print("=== Starting retrieval ===")
            retrieval_run(
                loader=args.get("loader", "neoforge"),
                version=args.get("version", "1.21.1"),
                sync=args.get("sync", False),
            )
            print("=== Starting embedding ===")
            embedding_run(batch_size=args.get("batch_size", 32))
        elif job_type == "prune":
            from workers.retrieval import run as retrieval_run
            retrieval_run(
                loader=args.get("loader", "neoforge"),
                version=args.get("version", "1.21.1"),
                prune=True,
            )
        elif job_type == "reset":
            conn = db.connect()
            conn.execute("DELETE FROM embeddings")
            conn.execute("UPDATE projects SET embedded_at = NULL")
            conn.commit()
            total = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            conn.close()
            print(f"Reset complete. {total} mods queued for re-embedding.")
        else:
            print(f"Unknown job type: {job_type}")
            sys.stdout = old_stdout
            return
        success = True
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        sys.stdout = old_stdout

    conn = db.connect()
    _finish_job(conn, job_id, success)
    conn.close()


def _beat(conn) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('runner_heartbeat', datetime('now'))"
    )
    conn.commit()


def _heartbeat_thread() -> None:
    """Runs on a daemon thread so the heartbeat updates even during long jobs."""
    while True:
        try:
            conn = db.connect()
            _beat(conn)
            conn.close()
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)


def _recover_orphans(conn) -> None:
    """On startup, mark any stale running jobs as error (previous crash)."""
    cur = conn.execute(
        "UPDATE jobs SET status = 'error', finished_at = datetime('now') "
        "WHERE status = 'running'"
    )
    conn.commit()
    if cur.rowcount:
        print(f"[runner] Recovered {cur.rowcount} orphaned job(s) → error")


def main() -> None:
    import threading
    conn = db.connect()
    db.init(conn)
    _recover_orphans(conn)
    conn.close()

    # Heartbeat runs independently so long jobs don't block it
    t = threading.Thread(target=_heartbeat_thread, daemon=True)
    t.start()

    print("[runner] Started. Polling for jobs…")

    while True:
        conn = db.connect()

        # Check for a claimable pending job
        running = _running_groups(conn)
        pending = conn.execute(
            "SELECT id, type, args FROM jobs WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()

        claimed = False
        for row in pending:
            job_groups = _GROUPS.get(row["type"], {row["type"]})
            if job_groups & running:
                continue  # conflict — skip
            if _claim_job(conn, row["id"]):
                conn.close()
                args = json.loads(row["args"])
                print(f"[runner] Starting job {row['id']} ({row['type']})")
                _run_job(row["id"], row["type"], args)
                print(f"[runner] Finished job {row['id']}")
                claimed = True
                break  # re-poll after each job

        if not claimed:
            conn.close()
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
