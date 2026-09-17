"""Structured logging, correlated to traces.

A log line that cannot be joined to a trace is a log line you will read once
and learn nothing from. ``bind_trace`` puts the current trace id into a
context variable, and the formatter writes it onto every record emitted while
that trace is active -- including records from libraries that know nothing
about any of this.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_TRACE_ID: ContextVar[str | None] = ContextVar("aie_trace_id", default=None)
_REQUEST_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("aie_request_context", default={})

# Record attributes the stdlib puts on every LogRecord; anything else the
# caller attached via `extra=` is request data worth emitting.
RESERVED_LOG_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}
_STANDARD = RESERVED_LOG_FIELDS


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


@contextmanager
def bind_trace(trace_id: str, **context: Any) -> Iterator[None]:
    trace_token = _TRACE_ID.set(trace_id)
    context_token = _REQUEST_CONTEXT.set({**_REQUEST_CONTEXT.get(), **context})
    try:
        yield
    finally:
        _TRACE_ID.reset(trace_token)
        _REQUEST_CONTEXT.reset(context_token)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        trace_id = _TRACE_ID.get()
        if trace_id:
            payload["trace_id"] = trace_id
        payload.update(_REQUEST_CONTEXT.get())

        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, *, json_format: bool = True) -> None:
    """Install the formatter on the root handler. Call once, at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        JSONFormatter()
        if json_format
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # The trace exporter emits its own JSON payload as the message; letting it
    # through at INFO is the point, but access logs at INFO are noise.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
