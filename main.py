"""
EPOCH ZERO — Orbital Defense System
Entry point for the Dash application.

Usage:
    uv run python main.py              # default: http://0.0.0.0:8051
    uv run python main.py --port 8080  # custom port
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="EPOCH ZERO — Fleet Surveillance System")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8051")), help="Bind port (default: 8051)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable Dash debug mode")
    args = parser.parse_args()

    from src.app import app

    print(f"\n{'='*60}")
    print(f"  EPOCH ZERO — Fleet Surveillance System")
    print(f"  Starting on http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(debug=args.debug, port=args.port, host=args.host, use_reloader=False)


if __name__ == "__main__":
    main()
