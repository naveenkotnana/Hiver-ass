"""CLI: python -m src.cli --message \"I was charged twice for my order\""""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.agent.support_agent import SupportAgent, format_cli


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Apple Support grounded agent")
    parser.add_argument("--message", required=True, help="Inbound customer message")
    parser.add_argument("--context", default="", help="Optional previous conversation context")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    try:
        agent = SupportAgent()
    except FileNotFoundError:
        print("Models not found. Run: python scripts/run_demo.py", file=sys.stderr)
        return 2

    result = agent.handle(args.message, context=args.context)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_cli(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
