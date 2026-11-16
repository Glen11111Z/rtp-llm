import gc
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from rtp_llm.metrics import GaugeMetrics, kmonitor

# Bounded so a stalled reporter can never grow the queue without limit; the
# oldest samples are the ones we would have wanted, but losing them is better
# than unbounded memory in a process that is already under pressure.
_MAX_PENDING = 65536
_REPORT_INTERVAL_S = 1.0

_pending: Deque[Tuple[int, float]] = deque(maxlen=_MAX_PENDING)
_lock = threading.Lock()
_tls = threading.local()
_installed = False
_reporting = False


def _on_gc(phase: str, info: dict) -> None:
    # The callback runs inside the collection, so it must stay allocation-free
    # and short: a perf_counter read and a deque append only.
    if phase == "start":
        _tls.started_at = time.perf_counter()
        return
    started_at: Optional[float] = getattr(_tls, "started_at", None)
    if started_at is None:
        return
    _tls.started_at = None
    pause_ms = (time.perf_counter() - started_at) * 1000.0
    with _lock:
        _pending.append((info.get("generation", -1), pause_ms))


def drain_gc_metrics() -> None:
    """Report the worst pause and collection count per generation since last drain.

    One gauge per generation keeps the series small when gen0 fires thousands of
    times per second, while still exposing the max pause: that is the number
    that competes with the 150ms RPC budget.
    """
    if not _pending:
        return
    with _lock:
        pending = list(_pending)
        _pending.clear()
    per_gen: Dict[int, List[float]] = {}
    for gen, pause_ms in pending:
        stats = per_gen.setdefault(gen, [0.0, 0])
        if pause_ms > stats[0]:
            stats[0] = pause_ms
        stats[1] += 1
    for gen, (max_pause_ms, count) in per_gen.items():
        # kmonitor serializes tags with str.join; a non-str value would raise in
        # the shared report thread and kill it for the whole process.
        tags = {"gen": str(gen)}
        kmonitor.report(GaugeMetrics.GC_PAUSE_MS_METRIC, max_pause_ms, tags)
        kmonitor.report(GaugeMetrics.GC_COLLECTION_COUNT_METRIC, count, tags)


def _report_loop() -> None:
    while True:
        time.sleep(_REPORT_INTERVAL_S)
        try:
            drain_gc_metrics()
        except Exception:  # noqa: BLE001 - a metric failure must not kill the thread
            logging.exception("failed to report gc metrics")


def install_gc_metric_hook() -> None:
    """Install the GC callback and the thread that reports its samples.

    Reporting must not depend on request paths: the stall we are looking for
    happens on the event loop itself, and the batch/generate entry points on
    this node may not be hit at all.
    """
    global _installed, _reporting
    if _installed:
        return
    _installed = True
    gc.callbacks.append(_on_gc)
    if not _reporting:
        _reporting = True
        threading.Thread(
            target=_report_loop, name="gc-metric-reporter", daemon=True
        ).start()
