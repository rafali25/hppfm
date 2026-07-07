#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parent / "tests" / "test_reconstruction_decoders.py"),
    run_name="__main__",
)
