"""Shared CLI plumbing for the pipeline scripts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from geoloc_tr.config import Config, load_config, parse_override  # noqa: E402


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", "-c", default=None, help="YAML config (configs/*.yaml)")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                   help="override, e.g. -s train.epochs=5 -s data.city=izmir")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup(args) -> Config:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return load_config(args.config, dict(parse_override(s) for s in args.set))


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
