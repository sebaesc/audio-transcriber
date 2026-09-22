"""Entry point for `python -m audio_transcriber`.

The CLI itself lands in Phase 5; until then this reports that it is not wired up
rather than failing with an import error.
"""

import sys


def main() -> int:
    """Run the command-line interface."""
    print("error: the CLI is not implemented yet.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
