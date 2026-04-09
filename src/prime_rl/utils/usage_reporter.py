"""Reports training token usage per step to the platform API for billing.

Activated entirely via environment variables — not exposed in user-facing
config. Self-hosted users are unaffected.

The platform API is idempotent on (run_id, step, usage_type), so replays
after crash-resume are safe.
"""

import asyncio
import os

from prime_rl.utils.background_async import BackgroundAsync
from prime_rl.utils.logger import get_logger


class UsageReporter:
    """Fire-and-forget training token usage reporter.

    Activated when ``PI_USAGE_BASE_URL`` is set in the environment.
    Wraps a ``BackgroundAsync`` runner so reporting never blocks training.
    """

    def __init__(self) -> None:
        self.logger = get_logger()
        api_key = os.environ.get("PI_USAGE_API_KEY")
        base_url = os.environ.get("PI_USAGE_BASE_URL", "").rstrip("/")
        # Both must be set. The orchestrator already gates on this, but
        # validate here too so any future caller fails fast at construction
        # rather than crashing inside httpx (which rejects None header
        # values with an obscure .encode() error on every retry).
        if not api_key or not base_url:
            raise ValueError(
                "UsageReporter requires PI_USAGE_BASE_URL and PI_USAGE_API_KEY to be set in the environment."
            )
        self._api_key = api_key
        self._base_url = base_url
        self._runner = BackgroundAsync(client_timeout=30.0)
        self.logger.debug(f"Usage reporter initialized (base_url={self._base_url})")

    def report_training_usage(self, run_id: str, step: int, tokens: int) -> None:
        url = f"{self._base_url}/usage"
        payload = {"run_id": run_id, "step": step, "usage_type": "training", "tokens": tokens}
        self._runner.submit(lambda: self._post(url, payload))

    def close(self) -> None:
        self._runner.close()

    async def _post(self, url: str, data: dict, max_retries: int = 3) -> None:
        headers = {"x-api-key": self._api_key, "Content-Type": "application/json"}
        for attempt in range(max_retries):
            try:
                response = await self._runner.client.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    body = response.json()
                    if body.get("status") == "duplicate":
                        self.logger.debug(
                            "Usage already recorded: run=%s step=%s type=%s",
                            data.get("run_id"),
                            data.get("step"),
                            data.get("usage_type"),
                        )
                    return
                response.raise_for_status()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.warning(
                        "Failed to report usage after %d attempts: %s: %s",
                        max_retries,
                        type(e).__name__,
                        e,
                    )
                else:
                    delay = 2**attempt
                    self.logger.debug(
                        "Retrying usage report in %ds (attempt %d/%d): %s",
                        delay,
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    await asyncio.sleep(delay)
