"""
Core pipeline for Archaeologist prototype.
"""
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import shutil
import csv

LOG = logging.getLogger("archaeologist.pipeline")

def identify_track_acoustid(filepath: Path, acoustid_key: Optional[str]) -> Tuple[str,str]:
    if acoustid_key is None:
        LOG.debug("No AcoustID key; using filename as title.")
        return ("Unknown Artist", filepath.stem)
    try:
        cmd = ["fpcalc", "-json", str(filepath)]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf8")
        data = json.loads(out)
        fingerprint = data.get("fingerprint")
        duration = int(data.get("duration",0))
        params = {"client": acoustid_key, "fingerprint": fingerprint, "duration": duration, "meta": "recordings"}
        import requests
        r = requests.get("https://api.acoustid.org/v2/lookup", params=params, timeout=20)
        r.raise_for_status()
        res = r.json()
        results = res.get("results",[])
        if results:
            recs = results[0].get("recordings",[])
            if recs:
                title = recs[0].get("title", filepath.stem)
                artists = recs[0].get("artists",[])
                artist = artists[0].get("name") if artists else "Unknown Artist"
                LOG.info(f"AcoustID: {artist} - {title}")
                return (artist, title)
    except Exception as e:
        LOG.warning(f"AcoustID lookup failed: {e}")
    return ("Unknown Artist", filepath.stem)


def separate_stems_demucs(source_path: Path, out_dir: Path) -> Optional[Path]:
    """Call demucs CLI to separate stems. Returns path to separation folder or None."""
    try:
        LOG.info("Running demucs (this may take a while)...")
        cmd = ["demucs", "-n", "htdemucs", "-o", str(out_dir), str(source_path)]
        subprocess.run(cmd, check=True)
        LOG.info("Demucs finished.")
        return out_dir
    except FileNotFoundError:
        LOG.warning("Demucs not found; skipping separation. Install demucs or set --skip-separate to true.")
    except subprocess.CalledProcessError as e:
        LOG.warning(f"Demucs failed: {e}")
    return None

def load_whisperx_model(model_name: str = "small", device: str = None):
    """
    Load WhisperX model with safe defaults for Apple Silicon and CPU/GPU.
    Forces float32 on M-series to avoid unsupported fp16 ops.
    """
    import platform, whisperx, torch

    # Detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            LOG.info("using mps torch backend.")
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Always force float32 for Apple Silicon or CPU
    compute_type = "float32"
    if device == "cuda" and torch.cuda.is_available():
        compute_type = "float16"  # safe on most NVIDIA GPUs

    print(f"[INFO] Loading WhisperX '{model_name}' on {device} ({compute_type})")
    return whisperx.load_model(model_name, device=device, compute_type=compute_type)

def transcribe_whisperx(source_path: Path, model_name: str = "small") -> Dict:
    """Transcribe with WhisperX if available. Force float32 on Apple Silicon by using compute_type argument where supported."""
    try:
        import whisperx
    except Exception as e:
        LOG.warning("whisperx not installed; attempt to fallback to whisper if available.")
        try:
            import whisper as whisper_base
            LOG.info("Using whisper (no word-level alignment).")
            model = whisper_base.load_model(model_name, device="cpu")
            try:
                model = model.float()
            except Exception:
                LOG.debug("Could not call .float() on model; proceeding.")
            res = model.transcribe(str(source_path), fp16=False)
            return {"text": res.get("text",""), "segments": res.get("segments",[])}
        except Exception:
            LOG.error("No transcription model available. Please install whisperx or whisper.")
            return {"text":"", "segments": []}

    LOG.info("Loading whisperx model (may take time). Forcing float32 for Apple Silicon compatibility.")

    model = whisperx.load_model(model_name, device="cpu", compute_type="float32")
    # model = load_whisperx_model(model_name)
    result = model.transcribe(str(source_path))
    try:
        aligned = whisperx.align(result["segments"], model, device="cpu")
        result["segments"] = aligned.get("word_segments", result["segments"])
    except Exception as e:
        LOG.debug(f"WhisperX alignment step failed or is unavailable: {e}")
    return result


def analyze_and_score(source_path: Path, stems_dir: Optional[Path], transcription: dict, topk: int = 12) -> List[Dict]:
    try:
        import librosa, numpy as np
    except Exception:
        LOG.error("librosa/numpy not available. Please install requirements.")
        return []

    y, sr = librosa.load(str(source_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    LOG.info(f"Loaded audio, duration={duration:.2f}s")

    candidates = []
    for seg in transcription.get("segments", []):
        s = float(seg.get("start",0.0))
        e = float(seg.get("end", s+0.5))
        text = seg.get("text","").strip()
        dur = max(0.15, e - s)
        candidates.append({"start": s, "dur": min(4.0,dur*2), "type":"vocal-phrase", "label": text, "score": 0.0})
        words = [w for w in text.split() if w]
        if words:
            n = min(len(words), 6)
            step = (e - s) / max(1,n)
            for i,w in enumerate(words[:12]):
                st = s + i*step
                d = max(0.12, min(1.2, step))
                candidates.append({"start": st, "dur": d, "type":"vocal-word", "label": w, "score": 0.0})

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    for t in onset_times:
        candidates.append({"start": float(t), "dur": 0.25, "type":"perc-hit", "label":"", "score":0.0})

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
    import numpy as np
    thresh = np.median(rms) * 1.2
    peaks = np.where(rms > thresh)[0]
    if peaks.size > 0:
        groups = []
        start = peaks[0]
        prev = peaks[0]
        for p in peaks[1:]:
            if p - prev > 2:
                groups.append((start, prev))
                start = p
            prev = p
        groups.append((start, prev))
        for a,b in groups:
            s = float(times[a])
            e = float(times[b])
            dur = min(8.0, e - s)
            if dur >= 0.5:
                candidates.append({"start": s, "dur": dur, "type":"drone","label":"", "score":0.0})

    S = librosa.onset.onset_strength(y=y, sr=sr)
    S_times = librosa.frames_to_time(range(len(S)), sr=sr)
    def novelty_at(t):
        idx = np.argmin(np.abs(S_times - t))
        return float(S[idx]) if idx >= 0 and idx < len(S) else 0.0

    for c in candidates:
        start = c["start"]
        dur = c["dur"]
        novelty = float(novelty_at(start))
        score = novelty * (dur ** 0.5)
        if c["type"].startswith("vocal"):
            score *= 1.6
            if len(c.get("label","")) > 6:
                score *= 1.2
        c["score"] = float(score)

    candidates_sorted = sorted(candidates, key=lambda x: (-x["score"], x["start"]))
    selected = []
    used_intervals = []
    for c in candidates_sorted:
        s = c["start"]
        e = s + c["dur"]
        overlap = False
        for a,b in used_intervals:
            if not (e < a or s > b):
                overlap = True
                break
        if not overlap:
            selected.append(c)
            used_intervals.append((s, e))
        if len(selected) >= topk:
            break

    LOG.info(f"Selected {len(selected)} candidate chops (topk={topk})")
    return selected


def write_chops_and_metadata(source_path: Path, artist: str, title: str, selected: List[Dict], out_base: Path, stems_dir: Optional[Path]):
    import soundfile as sf
    import librosa
    out_base.mkdir(parents=True, exist_ok=True)
    project_dir = out_base / f"{artist} - {title}"
    project_dir.mkdir(exist_ok=True)
    src_folder = project_dir / "source"
    src_folder.mkdir(exist_ok=True)
    dest = src_folder / source_path.name
    if not dest.exists():
        shutil.copy2(str(source_path), str(dest))

    chops_folder = project_dir / "chops"
    chops_folder.mkdir(exist_ok=True)

    vocal_path = None
    if stems_dir:
        for p in Path(stems_dir).rglob("*vocals*.wav"):
            vocal_path = p
            break

    if vocal_path:
        LOG.info(f"Using vocal stem at {vocal_path}")
    else:
        LOG.info("No vocal stem found; using full mix for chops. Consider running demucs for better vocal isolation.")

    y_full, sr = librosa.load(str(vocal_path) if vocal_path else str(source_path), sr=22050, mono=True)

    rows = []
    for c in selected:
        s = max(0.0, float(c["start"]))
        d = float(c["dur"])
        st = int(s * sr)
        en = int(min(len(y_full), (s + d) * sr))
        if en - st <= 0:
            continue
        chunk = y_full[st:en]
        lab = c.get("label","").strip().replace(" ", "_")[:40]
        fname = f"{s:.3f}-{d:.3f}-{c['type']}"
        if lab:
            fname += f"-{lab}"
        fname += ".wav"
        out_path = chops_folder / fname
        sf.write(str(out_path), chunk, sr)
        rows.append({"start": s, "dur": d, "type": c["type"], "label": c.get("label",""), "score": c.get("score",0.0), "file": str(out_path.relative_to(project_dir))})

    csv_path = project_dir / "chops.csv"
    with open(csv_path, "w", newline="", encoding="utf8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["start","dur","type","label","score","file"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    LOG.info(f"Wrote {len(rows)} chops to {chops_folder}; metadata at {csv_path}")
    return project_dir


def run_pipeline(source: Path, out_base: Path, acoustid_key: Optional[str], skip_sep: bool, topk: int, whisper_model: str):
    if not source.exists():
        LOG.error(f"Source file not found: {source}")
        return
    artist, title = identify_track_acoustid(source, acoustid_key)
    stems_dir = None
    if not skip_sep:
        stems_dir = separate_stems_demucs(source, out_base / "stems")
        LOG.info(f"Separation completed stems at: {stems_dir}")
    transcription = transcribe_whisperx(source, model_name=whisper_model)
    LOG.info(f"Transcription completed result: {transcription}")
    selected = analyze_and_score(source, stems_dir, transcription, topk=topk)
    LOG.info(f"Analysis completed selected: {selected}")
    project_dir = write_chops_and_metadata(source, artist, title, selected, out_base, stems_dir)
    LOG.info(f"Pipeline complete. Project at: {project_dir}")
