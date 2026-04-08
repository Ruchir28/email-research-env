#!/usr/bin/env python3

"""Build the SQLite inbox DB from the synthetic email blueprint file."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server.inbox_repository import ensure_inbox_db


def main() -> None:
    db_path = ensure_inbox_db(force=True)
    print(f"Built inbox DB at {db_path}")


if __name__ == "__main__":
    main()
