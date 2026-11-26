"""Protocol-layer timestamps for the pre-ASGI part of a request.

``TimingMiddleware`` only starts counting once the ASGI app is entered. Everything before
that - kernel accept, header parsing, scope construction, task scheduling - had no
timestamp, so it remained an opaque residual. This protocol stamps those points and hands
the values to the app through the ASGI scope.
"""

import contextvars
import time
from typing import Any

from uvicorn.protocols.http.auto import AutoHTTPProtocol

# Scope keys consumed by rtp_llm.frontend.frontend_server.TimingMiddleware.
SCOPE_DATA_NS = "rtp_req_data_ns"
SCOPE_ACCEPT_NS = "rtp_conn_accept_ns"
SCOPE_FIRST_REQ = "rtp_first_request"

CURRENT_REQUEST_TIMING = contextvars.ContextVar("rtp_request_timing", default=None)


class TimedHTTPProtocol(AutoHTTPProtocol):
    """HTTP protocol that stamps accept and first-byte times for each request."""

    def connection_made(self, transport: Any) -> None:
        self._rtp_accept_ns = time.perf_counter_ns()
        self._rtp_req_seen = 0
        super().connection_made(transport)

    def data_received(self, data: bytes) -> None:
        t_data_ns = time.perf_counter_ns()
        previous = getattr(self, "cycle", None)
        super().data_received(data)
        cycle = getattr(self, "cycle", None)
        if cycle is None or cycle is previous:
            return
        scope = getattr(cycle, "scope", None)
        if not isinstance(scope, dict):
            return
        first = self._rtp_req_seen == 0
        self._rtp_req_seen += 1
        scope[SCOPE_DATA_NS] = t_data_ns
        scope[SCOPE_FIRST_REQ] = first
        if first:
            scope[SCOPE_ACCEPT_NS] = self._rtp_accept_ns
