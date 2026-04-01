"""
Adaptive rate limiter for Modrinth API requests.

Reads X-Ratelimit-Remaining and X-Ratelimit-Reset headers from each response
and paces subsequent requests accordingly — no fixed delays.
"""

import sys
import time

import httpx

FALLBACK_DELAY = 0.5   # seconds to wait if headers are absent
LOW_WATER_MARK = 10    # remaining requests considered "critically low"
MAX_RETRIES = 5        # max retries on 429 before giving up


class APIDeprecatedError(Exception):
    pass


class RateLimiter:
    def wait(self, response: httpx.Response) -> None:
        """Call after every Modrinth API response to pace the next request."""
        try:
            remaining = int(response.headers.get("X-Ratelimit-Remaining", -1))
            reset = int(response.headers.get("X-Ratelimit-Reset", -1))
        except (ValueError, TypeError):
            time.sleep(FALLBACK_DELAY)
            return

        if remaining < 0 or reset < 0:
            time.sleep(FALLBACK_DELAY)
            return

        if remaining <= LOW_WATER_MARK:
            # Critically low — wait out the entire reset window
            sleep_for = reset + 1
            print(f"\n[rate limit] only {remaining} requests remaining — waiting {sleep_for}s for window reset")
        else:
            # Spread remaining budget evenly across the window
            sleep_for = reset / remaining

        time.sleep(sleep_for)


def checked_get(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    """
    Wrapper around client.get that handles Modrinth-specific error codes:
      - 410: API version deprecated — abort with a clear message
      - 429: Rate limit hit — retry with backoff up to MAX_RETRIES times
    All other non-2xx responses raise normally via raise_for_status().
    """
    for attempt in range(1, MAX_RETRIES + 1):
        resp = client.get(url, **kwargs)

        if resp.status_code == 400:
            try:
                body = resp.json()
                msg = f"{body.get('error', 'bad_request')}: {body.get('description', resp.text)}"
            except Exception:
                msg = resp.text
            raise ValueError(f"Modrinth API returned 400 — {msg}")

        if resp.status_code == 410:
            raise APIDeprecatedError(
                "Modrinth returned HTTP 410 — the API version in use has been deprecated.\n"
                "Please update mod_semantic_search to use the latest Modrinth API version."
            )

        if resp.status_code == 429:
            reset = int(resp.headers.get("X-Ratelimit-Reset", 60))
            wait = reset + 1
            print(f"\n[429] Rate limit hit — waiting {wait}s before retry {attempt}/{MAX_RETRIES}")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp

    print(f"[error] Still rate-limited after {MAX_RETRIES} retries — giving up.", file=sys.stderr)
    sys.exit(1)
