import asyncio
import logging
from typing import Optional

from rtp_llm.metrics import GaugeMetrics, kmonitor

_SAMPLE_INTERVAL_S = 0.05
# Only overshoot large enough to threaten an RPC budget is worth a sample;
# ordinary loop busyness under load would otherwise flood the series.
_REPORT_THRESHOLD_MS = 50.0

_task: Optional[asyncio.Task] = None


def start_event_loop_lag_sampler() -> None:
    """Start the sampler on the running loop, once per process.

    The task reference is kept module-level because asyncio only holds a weak
    reference to bare tasks.
    """
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.get_running_loop().create_task(run_event_loop_lag_sampler())


async def run_event_loop_lag_sampler(
    interval_s: float = _SAMPLE_INTERVAL_S,
    report_threshold_ms: float = _REPORT_THRESHOLD_MS,
) -> None:
    """Measure how late a fixed-interval task gets scheduled on this loop.

    The overshoot is what a pending RPC completion experiences when something
    blocks the loop: gRPC's deadline keeps running in C while Python cannot
    drain completions, so the request fails as DEADLINE_EXCEEDED even though
    the engine already answered.
    """
    loop = asyncio.get_running_loop()
    next_at: Optional[float] = loop.time() + interval_s
    while True:
        try:
            await asyncio.sleep(max(0.0, next_at - loop.time()))
            now = loop.time()
            lag_ms = (now - next_at) * 1000.0
            if lag_ms >= report_threshold_ms:
                kmonitor.report(GaugeMetrics.EVENT_LOOP_LAG_MS_METRIC, lag_ms)
                kmonitor.report(GaugeMetrics.EVENT_LOOP_LAG_COUNT_METRIC, 1)
            # Rebase instead of catching up: after a long stall, draining the
            # missed ticks back-to-back would report a burst of fake zeros.
            next_at = now + interval_s
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - sampling must never break the server
            logging.exception("event loop lag sampler failed")
            next_at = loop.time() + interval_s
