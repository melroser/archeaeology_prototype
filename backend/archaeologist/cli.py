#!/usr/bin/env python3
import os
import argparse
import logging
import sys
from pathlib import Path
from pipeline import run_pipeline
from dotenv import load_dotenv

ACOUSTID_API_KEY = os.getenv("ACOUSTID_API_KEY")

LOG = logging.getLogger("archaeologist.cli")


def setup_logging(verbosity: int):
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    h.setFormatter(fmt)
    logging.basicConfig(level=logging.WARNING, handlers=[h])

    if verbosity >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbosity == 1:
        logging.getLogger().setLevel(logging.INFO)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="archaeologist",
        description="Archaeologist CLI — create curated chops from a track",
    )
    parser.add_argument(
            "source", 
            type=Path, 
            help="Source audio file")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path.cwd() / "output",
        help="Output base folder",
    )
    parser.add_argument(
        "--acoustid-key",
        type=str,
        default=ACOUSTID_API_KEY,
        help="AcoustID API key (optional)",
    )
    parser.add_argument(
        "--skip-separate",
        action="store_true",
        help="Skip stem separation (useful for debugging)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=12,
        help="Number of curated chops to produce (producer mode)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["vocal", "energy", "lyrical", "perc", "balanced", "hybrid"],
        default="vocal",
        help="vocal=Vocal Priority, energy=Energy Dominant, lyric=Lyrical Meaning, perc=Percussion Driven, balanced=Balanced Diversity, hybrid=Weighted 60% Vocal 40% Energy",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="Whisper model (small/medium/large)",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="Comma-separated keywords to boost (e.g. wake up, suicide)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["producer", "nerd"],
        default="producer",
        help="Producer=curated few, Nerd=more candidates",
    )
    parser.add_argument(
            "--verbose",
            "-v",
            action="count",
            default=0
    )
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    LOG.info("Starting Archaeologist pipeline...")
    # Back/forward-compatible call (works with or without new params)
    from inspect import signature

    kwargs = dict(
        source=args.source.resolve(),
        out_base=args.out.resolve(),
        acoustid_key=args.acoustid_key,
        skip_sep=args.skip_separate,
        topk=args.topk,
        strat=args.strategy,
        whisper_model=args.whisper_model,
    )

    sig = signature(run_pipeline)
    if "keywords" in sig.parameters:
        kwargs["keywords"] = []
    if "mode" in sig.parameters:
        kwargs["mode"] = "producer"

    run_pipeline(**kwargs)


if __name__ == "__main__":
    main()
