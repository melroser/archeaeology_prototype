#!/usr/bin/env python3
import os
import argparse
import logging
import sys
from pathlib import Path
from pipeline import run_pipeline
from dotenv import load_dotenv
import zipfile
import tempfile

# Load environment variables from .env file
load_dotenv()

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

def expand_sources(source: Path):
    """Expand a source into a list of audio files."""
    if source.is_dir():
        return list(source.rglob("*.wav")) + list(source.rglob("*.mp3")) + list(source.rglob("*.flac")) + list(source.rglob("*.m4a"))
    elif source.suffix.lower() == ".zip":
        tmpdir = Path(tempfile.mkdtemp(prefix="arch_"))
        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(tmpdir)
        return list(tmpdir.rglob("*.wav")) + list(tmpdir.rglob("*.mp3")) + list(tmpdir.rglob("*.flac")) + list(tmpdir.rglob("*.m4a"))
    elif source.is_file():
        return [source]
    else:
        return []

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
        default="vocal",
        help="Comma-separated strategies: vocal,energy,lyrical,perc,balanced,hybrid",
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
    sources = expand_sources(args.source)
    if not sources:
        LOG.error(f"No valid sources found for {args.source}")
        sys.exit(1)

    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
    if not strategies:
        strategies = ["vocal"]

    for src in sources:
        for strat in strategies:
            LOG.info(f"Processing {src} with strategy {strat}")

            kwargs = dict(
                source=src.resolve(),
                out_base=args.out.resolve(),
                acoustid_key=args.acoustid_key,
                skip_sep=args.skip_separate,
                topk=args.topk,
                strat=strat,
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
