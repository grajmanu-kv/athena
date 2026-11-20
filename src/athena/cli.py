"""
athena.cli
==========
Command line entry point.

    athena serve [--host 127.0.0.1] [--port 8000] [--reload]

`serve` starts the web test harness (requires the `web` extra). Configuration
is read from the environment / a .env file; see athena.web.settings.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="athena", description="Athena epistemic validator")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web test harness")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1; the UI has no auth)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")

    args = parser.parse_args(argv)

    if args.command == "serve":
        from athena.web import require_web_extra

        try:
            require_web_extra()
        except ImportError as exc:
            print(f"athena: {exc}", file=sys.stderr)
            return 2
        import uvicorn

        from athena.web.settings import Settings

        if not Settings.from_env().key_configured:
            print("athena: warning: no GEMINI_API_KEY / GOOGLE_API_KEY set; "
                  "the UI will load but validation requests will return 503.", file=sys.stderr)
        uvicorn.run("athena.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
