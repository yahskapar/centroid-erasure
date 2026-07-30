#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akshay Paruchuri
"""Summarize the released TCCD/LCD/VCD alpha and cross-task CV records."""

from __future__ import annotations

import json
from pathlib import Path


RESULT = Path(__file__).resolve().parent.parent / "results" / "cd_baselines_cv.json"


def main() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    for name, method in sorted(payload["per_method"].items()):
        print(
            f"{name}: "
            f"cross-task-CV={100 * method['cv_mean']:+.2f} pp, "
            f"fixed-0.4={100 * method['fixed04_mean']:+.2f} pp, "
            f"oracle={100 * method['oracle_mean']:+.2f} pp"
        )


if __name__ == "__main__":
    main()
