#!/usr/bin/env python3
"""Convenience entry point for the DQN policy-integration experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_policy_integration import main

if __name__ == "__main__":
    main()
