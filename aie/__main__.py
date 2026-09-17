"""Entry point: ``python -m aie``.

Owns process-level setup (logging configuration) that must not happen as an
import side effect -- an app factory that reconfigures root logging also stomps
on whatever the importing process had set up, test harnesses included.
"""

from __future__ import annotations

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aie", description="Serve the AIE platform.")
    parser.add_argument("--host", default=os.getenv("AIE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AIE_PORT", "8000")))
    parser.add_argument("--log-level", default=os.getenv("AIE_LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--plain-logs", action="store_true", help="human-readable logs instead of JSON"
    )
    args = parser.parse_args(argv)

    import uvicorn

    from aie.observe.logging import configure_logging

    configure_logging(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        json_format=not args.plain_logs,
    )

    from aie.api.app import create_app

    # log_config=None: uvicorn otherwise installs its own formatters over ours.
    uvicorn.run(create_app(), host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
