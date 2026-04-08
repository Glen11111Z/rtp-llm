import threading
import time
from contextlib import contextmanager
from typing import Optional

_lock = threading.Lock()
_tracer = None
_tracing = False


def _get_tracer():
    global _tracer
    if _tracer is None:
        with _lock:
            if _tracer is None:
                import viztracer

                _tracer = viztracer.VizTracer(tracer_entries=1000000)
    return _tracer


@contextmanager
def trace_scope(enable_trace: bool = True, file_name: Optional[str] = None):
    global _tracing
    if not enable_trace:
        yield
        return

    # skip if already started tracing
    with _lock:
        if _tracing:
            skip = True
        else:
            skip = False
            _tracing = True

    if skip:
        yield
        return

    tracer = _get_tracer()
    if file_name is None:
        file_name = f"frontend_trace_{int(time.time() * 1000)}.json"

    tracer.start()
    try:
        yield
    finally:
        tracer.stop()
        tracer.save(file_name)
        tracer.clear()
        with _lock:
            _tracing = False
