# Archaeologist — Prototype (MVP)
Standalone Python prototype for Archaeologist: stem-separate, transcribe (local), and produce curated vocal chops for producers.
This prototype is local-first. Cloud backends are optional and can be enabled via flags.

How to run (example)
1. Create venv and install dependencies (see requirements.txt)
2. Run:

python -m archaeologist.cli /path/to/song.mp3 --out ./output --topk 12

This will produce a curated set of chops (default top 12) in the output folder.


# Archaeolog1st
Dig up the gold hiding in your music library.

## Features
- AI stem separation
- Lyric-aware chop detection
- Float32-safe on Apple Silicon
- Web-based local preview UI

## Quickstart
```bash
./run.sh
