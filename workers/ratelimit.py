"""
Adaptive rate limiter for Modrinth API requests.

Reads X-Ratelimit-Remaining and X-Ratelimit-Reset headers from each response
and paces subsequent requests accordingly — no fixed delays.
"""

import time

import httpx

FALLBACK_DELAY = 0.5   # seconds to wait if headers are absent
LOW_WATER_MARK = 10    # remaining requests considered "critically low"


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
