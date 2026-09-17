"""Per-iteration LLM cost/token accounting, read off the chia profiler.

`ClaudeCodeLLM.prompt` pushes `cost_usd`, `input_tokens`, `output_tokens`,
`cache_*_tokens` and `num_turns` into the profiler as the call's `extra` dict
(chia/models/claude.py ~L1065-1098 -> `profiler.add_info`). Those land on the
`complete` event the worker sends to the `ChiaProfileCollector` actor, which
`loop.py` starts with `start_collector()`.

This module turns that stream into "what did iteration N cost": the meter keeps
a cursor into the collector's event list and each `poll()` returns the sum of
every cost-bearing event that appeared since the previous poll.

The worker's event send is fire-and-forget, so a poll immediately after `get()`
can race it — `poll()` therefore waits (briefly) for at least one new
cost-bearing event before giving up.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("aether.costs")

# The keys claude.py pushes; everything here is additive across turns.
_SUM_KEYS = ("cost_usd", "input_tokens", "output_tokens",
             "cache_read_input_tokens", "cache_creation_input_tokens",
             "num_turns")

EMPTY: dict = {k: 0 for k in _SUM_KEYS}


class CostMeter:
    """Cursor over the profiler's event log, yielding per-iteration usage."""

    def __init__(self) -> None:
        self._collector = None
        self._cursor = 0
        try:
            from chia.trace.profiler import get_collector
            self._collector = get_collector()
        except Exception as exc:  # profiler unavailable -> zeros, never fatal
            logger.warning("cost accounting disabled (%s)", exc)
        if self._collector is None:
            logger.warning("no ChiaProfileCollector actor — costs will read 0. "
                           "Did start_collector() run?")
        else:
            # Don't bill this run for events left over from an earlier driver.
            self._cursor = len(self._all_events())

    # -- internals ---------------------------------------------------------

    def _all_events(self) -> list[dict]:
        if self._collector is None:
            return []
        try:
            import ray
            return ray.get(self._collector.get_events.remote())
        except Exception as exc:
            logger.warning("could not read profiler events: %s", exc)
            return []

    @staticmethod
    def _usage(event: dict) -> dict | None:
        extra = event.get("extra") or {}
        if "cost_usd" not in extra and "output_tokens" not in extra:
            return None
        return extra

    # -- public ------------------------------------------------------------

    def poll(self, wait_seconds: float = 20.0) -> dict:
        """Usage accumulated since the last poll.

        Blocks up to *wait_seconds* for the in-flight `complete` event to reach
        the collector; returns zeros (never raises) if it never shows up.
        """
        deadline = time.time() + wait_seconds
        while True:
            events = self._all_events()
            new = events[self._cursor:]
            usages = [u for e in new if (u := self._usage(e)) is not None]
            if usages or time.time() >= deadline or self._collector is None:
                self._cursor = len(events)
                break
            time.sleep(1.0)

        if not usages:
            logger.warning("no LLM usage event seen for this iteration")
            return dict(EMPTY)
        out = {k: sum(u.get(k, 0) or 0 for u in usages) for k in _SUM_KEYS}
        out["cost_usd"] = round(float(out["cost_usd"]), 6)
        return out

    @staticmethod
    def add(a: dict, b: dict) -> dict:
        """Running total helper."""
        return {k: (a.get(k, 0) or 0) + (b.get(k, 0) or 0) for k in _SUM_KEYS}


def fmt(c: dict) -> str:
    """One-line human summary of a usage dict."""
    return (f"${c.get('cost_usd', 0):.4f}  "
            f"in={c.get('input_tokens', 0)} out={c.get('output_tokens', 0)} "
            f"cache_rd={c.get('cache_read_input_tokens', 0)} "
            f"cache_cr={c.get('cache_creation_input_tokens', 0)} "
            f"turns={c.get('num_turns', 0)}")
