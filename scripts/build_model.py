#!/usr/bin/env python3
from __future__ import annotations

import traceback

from device_builder.cli import parse_args
from device_builder.io_utils import log
from device_builder.pipelines.build_pipeline import run_build


if __name__ == "__main__":
    try:
        raise SystemExit(run_build(parse_args()))
    except Exception as exc:
        log("\n[FATAL] Build failed:")
        log(str(exc))
        log(traceback.format_exc())
        raise