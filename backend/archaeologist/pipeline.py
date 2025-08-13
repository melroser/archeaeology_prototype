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

def save_analysis_and_transcript(project_dir: Path,
                                 source_path: Path,
                                 transcription: dict,
                                 selected: List[Dict]) -> None:
    """
    Write:
      - transcript/transcript.json  (segments with start/end/text)
      - analysis.json               (novelty, vocal activity, epicness, boundaries, chops)
    Minimal deps; no sklearn. Boundaries via novelty peaks (simple & fast).
    """
    import librosa
    import numpy as np

    # Ensure dirs
    tdir = project_dir / "transcript"
    tdir.mkdir(exist_ok=True)

    # Transcript JSON
    tjson = tdir / "transcript.json"
    segs = transcription.get("segments", []) or []
    with tjson.open("w", encoding="utf8") as f:
        json.dump({"segments": segs}, f, ensure_ascii=False, indent=2)

    # Analysis (novelty + vocal mask + epicness)
    y, sr = librosa.load(str(source_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Novelty curve (fast, robust)
    novelty = librosa.onset.onset_strength(y=y, sr=sr)
    n_frames = len(novelty)
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr)

    # Normalize novelty
    if np.max(novelty) > 0:
        novelty_norm = novelty / np.max(novelty)
    else:
        novelty_norm = novelty

    # Vocal activity mask (same frame grid as novelty)
    vocal_mask = np.zeros_like(novelty_norm, dtype=float)
    for seg in segs:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        a = np.searchsorted(times, s, side="left")
        b = np.searchsorted(times, e, side="right")
        a = max(0, min(a, n_frames - 1))
        b = max(a + 1, min(b, n_frames))
        vocal_mask[a:b] = 1.0

    # Smooth vocal mask with small moving average (no scipy)
    if len(vocal_mask) >= 5:
        kernel = np.ones(5, dtype=float) / 5.0
        vocal_mask = np.convolve(vocal_mask, kernel, mode="same")

    # Blend into epicness (tweak weights later if desired)
    epicness = 0.6 * novelty_norm + 0.4 * vocal_mask
    if np.max(epicness) > 0:
        epicness = epicness / np.max(epicness)

    # Simple boundaries: local maxima of novelty with spacing
    def _pick_peaks(x: np.ndarray, min_gap_frames: int = 12, top_k: int = 24) -> List[float]:
        idxs = []
        last = -min_gap_frames
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] >= x[i + 1]:
                if i - last >= min_gap_frames:
                    idxs.append(i)
                    last = i
        # Rank by value and take top_k
        idxs = sorted(idxs, key=lambda i: x[i], reverse=True)[:top_k]
        idxs.sort()
        return [float(times[i]) for i in idxs]

    boundaries = _pick_peaks(novelty_norm)

    # Include selected chops in the analysis for UI convenience
    chops_json = [
        {
            "start": float(c.get("start", 0.0)),
            "dur": float(c.get("dur", 0.0)),
            "end": float(c.get("start", 0.0)) + float(c.get("dur", 0.0)),
            "type": c.get("type", ""),
            "label": c.get("label", ""),
            "score": float(c.get("score", 0.0)),
        }
        for c in (selected or [])
    ]

    analysis = {
        "duration": duration,
        "novelty": {"times": times.tolist(), "values": novelty_norm.astype(float).tolist()},
        "vocal_activity": {"times": times.tolist(), "values": vocal_mask.astype(float).tolist()},
        "epicness": {"times": times.tolist(), "values": epicness.astype(float).tolist()},
        "boundaries": boundaries,
        "chops": chops_json,
    }

    with (project_dir / "analysis.json").open("w", encoding="utf8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)




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


def separate_stems_demucs(source_path: Path, stems_dir: Path) -> Optional[Path]:
    """
    Call demucs CLI to separate stems. Returns path to final stems folder.
    """
    try:
        LOG.info("Running demucs (this may take a while)...")

        # demucs writes to: stems_dir/htdemucs/<track_name>/*.wav
        cmd = ["demucs", "-n", "htdemucs", "-o", str(stems_dir), str(source_path)]
        subprocess.run(cmd, check=True)

        LOG.info("Demucs finished.")

        track_name = source_path.stem
        src_dir = stems_dir / "htdemucs" / track_name
        final_dir = stems_dir
        final_dir.mkdir(parents=True, exist_ok=True)

        if src_dir.exists() and src_dir.is_dir():
            for stem_file in src_dir.glob("*.wav"):
                dest_file = final_dir / stem_file.name
                if dest_file.exists():
                    dest_file.unlink()
                shutil.move(str(stem_file), str(dest_file))

            # best-effort cleanup
            try:
                src_dir.rmdir()
                htd = stems_dir / "htdemucs"
                if htd.exists() and not any(htd.iterdir()):
                    htd.rmdir()
            except Exception:
                pass

            return final_dir

        return final_dir

    except FileNotFoundError:
        LOG.warning("Demucs not found; skipping separation. Install demucs or set --skip-separate.")
    except subprocess.CalledProcessError as e:
        LOG.warning(f"Demucs failed: {e}")

    return None


def _get_device_and_compute_type():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps", "float32"    # Apple Silicon
        if torch.cuda.is_available():
            return "cuda", "float16"   # NVIDIA: fp16 ok
    except Exception:
        pass
    return "cpu", "float32"
def transcribe_whisperx(source_path: Path, model_name: str = "small") -> Dict:
    """
    Transcribe with WhisperX and perform forced alignment for word-level timestamps.
    Forces float32 on Apple Silicon/CPU to avoid fp16 errors.
    Falls back to whisper if whisperx not available.
    """
    try:
        import whisperx
        device = "cpu"
        compute_type = "float32"

        # simple, robust device choice
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                compute_type = "float32"
            elif torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"  # ok on NVIDIA
        except Exception:
            pass

        LOG.info(f"Loading WhisperX model={model_name} device={device} compute_type={compute_type}")
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type)

        result = model.transcribe(str(source_path))

        # forced alignment for word-level timestamps
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result.get("language", "en"),
                device=device
            )
            aligned = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                str(source_path),
                device=device
            )
            if aligned and aligned.get("word_segments"):
                result["segments"] = aligned["word_segments"]
            elif aligned and aligned.get("segments"):
                result["segments"] = aligned["segments"]
        except Exception as e:
            LOG.warning(f"WhisperX alignment unavailable or failed: {e}")

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", [])
        }

    except Exception as e:
        LOG.warning(f"whisperx not available ({e}); falling back to whisper.")

        try:
            import whisper as whisper_base
            model = whisper_base.load_model(model_name, device="cpu")
            res = model.transcribe(str(source_path), fp16=False)
            return {"text": res.get("text", ""), "segments": res.get("segments", [])}
        except Exception as e2:
            LOG.error(f"No transcription model available. Install whisperx or whisper. {e2}")
            return {"text": "", "segments": []}


def analyze_and_score(source_path: Path,
                      stems_dir: Optional[Path],
                      transcription: dict,
                      topk: int = 12,
                      keywords: Optional[List[str]] = None,
                      mode: str = "producer") -> List[Dict]:
    try:
        import librosa, numpy as np
    except Exception:
        LOG.error("librosa/numpy not available. Please install requirements.")
        return []

    y, sr = librosa.load(str(source_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    LOG.info(f"Loaded audio, duration={duration:.2f}s")

    kw = [k.lower() for k in (keywords or [])]

    candidates = []
    for seg in transcription.get("segments", []):
        s = float(seg.get("start",0.0))
        e = float(seg.get("end", s+0.5))
        text = (seg.get("text","") or seg.get("word","")).strip()
        dur = max(0.15, e - s)
        # phrase
        candidates.append({"start": s, "dur": min(4.0,dur*2), "type":"vocal-phrase", "label": text, "score": 0.0})
        # words (from alignment or naive split)
        words = [w for w in text.split() if w]
        if words:
            n = min(len(words), 6)
            step = (e - s) / max(1,n)
            for i,w in enumerate(words[:12]):
                st = s + i*step
                d = max(0.12, min(1.2, step))
                candidates.append({"start": st, "dur": d, "type":"vocal-word", "label": w, "score": 0.0})

    # percussion using drums stem if present
    import librosa
    y_perc = None
    if stems_dir:
        drums_path = stems_dir / "drums.wav"
        if drums_path.exists():
            y_perc, _ = librosa.load(str(drums_path), sr=22050, mono=True)
    if y_perc is None:
        y_perc = y

    onset_frames = librosa.onset.onset_detect(y=y_perc, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    for t in onset_times:
        candidates.append({"start": float(t), "dur": 0.25, "type":"perc-hit", "label":"", "score":0.0})

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
    import numpy as np
    thresh = np.median(rms) * (1.2 if mode=="producer" else 1.0)
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

        # sustained/drone moments (use bass/other/vocals if available)
        y_drone = None
        if stems_dir:
            for stem_name in ["bass.wav", "other.wav", "vocals.wav"]:
                stem_path = stems_dir / stem_name
                if stem_path.exists():
                    y_drone, _ = librosa.load(str(stem_path), sr=22050, mono=True)
                    break
        if y_drone is None:
            y_drone = y

        for a,b in groups:
            s = float(times[a])
            e = float(times[b])
            dur = e - s
            if dur >= (4.0 if mode=="producer" else 2.0):
                candidates.append({"start": s, "dur": min(8.0, dur), "type":"drone","label":"", "score":0.0})

    # novelty curve for scoring
    S = librosa.onset.onset_strength(y=y, sr=sr)
    S_times = librosa.frames_to_time(range(len(S)), sr=sr)
    def novelty_at(t):
        idx = np.argmin(np.abs(S_times - t))
        return float(S[idx]) if 0 <= idx < len(S) else 0.0

    for c in candidates:
        start = c["start"]
        dur = c["dur"]
        novelty = float(novelty_at(start))
        score = novelty * (dur ** 0.5)

        if c["type"].startswith("vocal"):
            score *= 1.7
            label = (c.get("label","") or "").lower()
            # keyword boost
            if kw and any(k in label for k in kw):
                score *= 1.5
            # prefer non-stopwords-ish tokens
            if len(label) > 3:
                score *= 1.15

        c["score"] = float(score)

    candidates_sorted = sorted(candidates, key=lambda x: (-x["score"], x["start"]))
    selected = []
    used_intervals = []
    for c in candidates_sorted:
        s = c["start"]
        e = s + c["dur"]
        overlap = any(not (e < a or s > b) for a,b in used_intervals)
        if not overlap:
            selected.append(c)
            used_intervals.append((s, e))
        if len(selected) >= (topk if mode=="producer" else min(topk*4, 200)):
            break

    LOG.info(f"Selected {len(selected)} candidate chops (mode={mode}, topk={topk})")
    return selected

def write_chops_and_metadata(source_path: Path, selected: List[Dict], project_dir: Path):
    import soundfile as sf
    import librosa

    project_dir.mkdir(exist_ok=True)
    src_folder = project_dir / "source"
    src_folder.mkdir(exist_ok=True)
    dest = src_folder / source_path.name
    if not dest.exists():
        shutil.copy2(str(source_path), str(dest))

    chops_folder = project_dir / "chops"
    chops_folder.mkdir(exist_ok=True)

    stems_dir = project_dir / "stems"
    # prefer common demucs name first
    vocal_path = None
    if stems_dir.exists():
        vp = stems_dir / "vocals.wav"
        if vp.exists():
            vocal_path = vp
        else:
            for p in stems_dir.rglob("*vocals*.wav"):
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
        lab = (c.get("label","") or "").strip().replace(" ", "_")[:40]
        fname = f"{s:.3f}-{d:.3f}-{c['type']}"
        if lab:
            fname += f"-{lab}"
        score_rounded = round(c.get("score", 0.0), 3)
        fname += f"-{score_rounded:.3f}.wav"
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


def run_pipeline(
    source: Path,
    out_base: Path,
    acoustid_key: Optional[str],
    skip_sep: bool,
    topk: int,
    whisper_model: str
):
    if not source.exists():
        LOG.error(f"Source file not found: {source}")
        return

    artist, title = identify_track_acoustid(source, acoustid_key)

    out_base.mkdir(parents=True, exist_ok=True)
    project_dir = out_base / f"{artist} - {title}"
    project_dir.mkdir(exist_ok=True)

    stems_dir = None

    if not skip_sep:
        stems_dir = separate_stems_demucs(source, project_dir / "stems")
        LOG.info(f"Separation completed. Stems at: {stems_dir}")

    transcription = transcribe_whisperx(source, model_name=whisper_model)
    LOG.info("Transcription complete: %d segments", len(transcription.get("segments", [])))

    selected = analyze_and_score(source, stems_dir, transcription, topk=topk)
    write_dir = write_chops_and_metadata(source, selected, project_dir)

    # NEW: save analysis + transcript for UI (minimal addition; no new args)
    save_analysis_and_transcript(project_dir, source, transcription, selected)

    LOG.info(f"Pipeline complete. Project at: {write_dir}")


