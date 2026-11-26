"""Sampling profiler for the FE event-loop thread.

Gap-B work showed that per-request latency is dominated by event-loop
scheduling delay, not by any single slow call. To find which synchronous
Python work keeps the loop busy, a daemon thread samples the main thread's
leaf frame every few ms and keeps a short rolling window. A summary (top
functions + busy ratio) is refreshed once per second and embedded into
aux_info, so it can be read from ordinary chat responses without any log or
exec access to the pod.

Enable with RTP_FE_PROFILE=1. Overhead is a few percent; samples taken while
the loop sits in epoll are counted as idle, which makes the busy ratio a
direct measure of per-process loop utilization.
"""

import os
import sys
import threading
import time
from collections import Counter, deque
from typing import Optional, Tuple

_SAMPLE_INTERVAL_S = 0.002
_WINDOW_S = 15.0
_MAX_SAMPLES = 10000
_SUMMARY_REFRESH_S = 1.0

_started = False
_main_ident: Optional[int] = None
_samples: deque = deque(maxlen=_MAX_SAMPLES)
_current: Tuple[Optional[str], float, int] = (None, 0.0, 0)

_IDLE_FILES = ("selectors.py",)


def _is_idle(frame) -> bool:
    code = frame.f_code
    if code.co_filename.endswith(_IDLE_FILES):
        return True
    return code.co_name in ("epoll_wait", "poll", "select")


def _leaf_key(frame) -> str:
    # sys._current_frames() already returns the innermost frame; walking
    # f_back would land on the module-level frame instead.
    code = frame.f_code
    return f"{code.co_filename.rsplit('/', 1)[-1]}:{code.co_name}:{frame.f_lineno}"


def start() -> None:
    global _started, _main_ident
    if _started:
        return
    if os.getenv("RTP_FE_PROFILE") not in ("1", "true", "True"):
        return
    _started = True
    _main_ident = threading.main_thread().ident
    threading.Thread(target=_run, name="fe-loop-profiler", daemon=True).start()


def _run() -> None:
    global _current
    last_summary = 0.0
    while True:
        time.sleep(_SAMPLE_INTERVAL_S)
        frame = sys._current_frames().get(_main_ident)
        if frame is None:
            continue
        now = time.monotonic()
        idle = _is_idle(frame)
        _samples.append((now, "<idle>" if idle else _leaf_key(frame)))
        if now - last_summary >= _SUMMARY_REFRESH_S:
            last_summary = now
            _current = _summarize(now)


def _summarize(now: float) -> Tuple[Optional[str], float, int]:
    counts = Counter(key for ts, key in _samples if now - ts <= _WINDOW_S)
    total = sum(counts.values())
    if not total:
        return None, 0.0, 0
    busy = 1.0 - counts.get("<idle>", 0) / total
    top = [f"{k}:{round(100 * c / total)}%" for k, c in counts.most_common(6)]
    return ",".join(top), round(busy, 3), total


def snapshot() -> Tuple[Optional[str], float, int]:
    return _current