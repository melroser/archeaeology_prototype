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

from abc import ABC, abstractmethod
import numpy as np

LOG = logging.getLogger("archaeologist.pipeline")

class ScoringStrategy(ABC):
    @abstractmethod
    def score(self, c, novelty_func):
        pass

class VocalPriorityStrategy(ScoringStrategy):
    def score(self, c, novelty_func):
        LOG.debug("Using Strategy: Vocal")
        novelty = novelty_func(c["start"])
        score = novelty * (c["dur"] ** 0.5)
        if c["type"].startswith("vocal"):
            score *= 1.6
            if len(c.get("label","")) > 6:
                score *= 1.2
        return score


class EnergyDominantStrategy(ScoringStrategy):
    def score(self, c, novelty_func):
        LOG.debug("Using Strategy: Energy")
        novelty = novelty_func(c["start"])        # Energy weighting via RMS proxy: novelty acts as proxy but could use RMS array directly
        return novelty * (c["dur"] ** 0.7)


class LyricalImportanceStrategy(ScoringStrategy):
    def score(self, c, novelty_func):
        LOG.debug("Using Strategy: Lyrical")
        novelty = novelty_func(c["start"])
        score = novelty
        if c["type"].startswith("vocal"):
            label = c.get("label","").lower()
            important_words = {"love", "hate", "die", "fire", "forever", "never", "always", "suicide"}
            if any(word in label for word in important_words):
                score *= 2.0
        return score

class PercussionDrivenStrategy(ScoringStrategy):
    def score(self, c, novelty_func):
        LOG.debug("Using Strategy: Perc")
        if "perc" in c["type"]:
            novelty = novelty_func(c["start"])
            return novelty * 2.0
        return 0.5 * novelty_func(c["start"])

class BalancedDiversityStrategy(ScoringStrategy):
    def __init__(self):
        self.type_counts = {}
    def score(self, c, novelty_func):
        LOG.debug("Using Strategy: Bal")
        novelty = novelty_func(c["start"])
        type_count = self.type_counts.get(c["type"], 0)
        # Penalize repetition of same type
        penalty = 1.0 / (1 + type_count)
        self.type_counts[c["type"]] = type_count + 1
        return novelty * penalty

class HybridScoringStrategy(ScoringStrategy):
    def __init__(self, strategies_with_weights):
        """
        strategies_with_weights: list of (strategy_instance, weight)
        Example:
            [
                (VocalPriorityStrategy(), 0.6),
                (EnergyDominantStrategy(), 0.4)
            ]
        """
        self.strategies_with_weights = strategies_with_weights

    def score(self, c, novelty_func):
        LOG.debug("Using Strategy: Hybrid")
        total_score = 0.0
        total_weight = 0.0
        for strat, weight in self.strategies_with_weights:
            s = strat.score(c, novelty_func)
            total_score += s * weight
            total_weight += weight
        return total_score / total_weight if total_weight > 0 else 0.0


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
    # First, extract info from filename as fallback
    filename_hint = filepath.stem.lower()

    if acoustid_key is None:
        LOG.debug("No AcoustID key; using filename as title.")
        return ("Unknown Artist", filepath.stem)

    try:
        cmd = ["fpcalc", "-json", str(filepath)]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf8")
        data = json.loads(out)
        fingerprint = data.get("fingerprint")
        duration = int(data.get("duration",0))
        params = {
                "client": acoustid_key, 
                "fingerprint": fingerprint, 
                "duration": duration, 
                "meta": "recordings"
                }
        import requests
        r = requests.get("https://api.acoustid.org/v2/lookup", params=params, timeout=20)
        r.raise_for_status()
        res = r.json()
        results = res.get("results",[])

        # Log all results
        LOG.info(f"AcoustID returned {len(results)} results:")
        for i, result in enumerate(results[:3]):
            score = result.get("score", 0.0)
            recs = result.get("recordings", [])
            if recs:
                title = recs[0].get("title", "Unknown")
                artists = recs[0].get("artists", [])
                artist = artists[0].get("name") if artists else "Unknown"
                LOG.info(f"  {i+1}. {artist} - {title} (confidence: {score:.3f})")

        # Check for suspicious matches
        if results:
            best_result = results[0]
            confidence = best_result.get("score", 0.0)
            recs = best_result.get("recordings",[])

            if recs:
                title = recs[0].get("title", filepath.stem)
                artists = recs[0].get("artists",[])
                artist = artists[0].get("name") if artists else "Unknown Artist"

                # Sanity check: Does this match make sense?
                suspicious = False

                # Check if result is generic like "Track 15"
                if title.lower().startswith("track ") and title[6:].isdigit():
                    LOG.warning(f"Suspicious generic title: {title}")
                    suspicious = True

                # Check if filename contains known artist/song info that doesn't match
                if "deer dance" in filename_hint and "deer" not in title.lower():
                    LOG.warning(f"Filename suggests 'Deer Dance' but got '{title}'")
                    suspicious = True

                if "system of a down" in filename_hint and "system" not in artist.lower():
                    LOG.warning(f"Filename suggests 'System of a Down' but got '{artist}'")
                    suspicious = True

                if suspicious and confidence < 0.99:  # Even high confidence can be wrong
                    LOG.warning(f"Suspicious match detected, using filename instead")
                    return ("Unknown Artist", filepath.stem)

                LOG.info(f"✓ Match accepted: {artist} - {title} (confidence: {confidence:.3f})")
                return (artist, title)

        LOG.warning("No valid matches found")

    except Exception as e:
        LOG.warning(f"AcoustID lookup failed: {e}")

    return ("Unknown Artist", filepath.stem)

# def identify_track_acoustid(filepath: Path, acoustid_key: Optional[str]) -> Tuple[str,str]:
#     if acoustid_key is None:
#         LOG.debug("No AcoustID key; using filename as title.")
#         return ("Unknown Artist", filepath.stem)
#     try:
#         cmd = ["fpcalc", "-json", str(filepath)]
#         out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf8")
#         data = json.loads(out)
#         fingerprint = data.get("fingerprint")
#         duration = int(data.get("duration",0))
#         params = {
#             "client": acoustid_key,
#             "fingerprint": fingerprint,
#             "duration": duration,
#             "meta": "recordings"
#         }
#         import requests
#         r = requests.get("https://api.acoustid.org/v2/lookup", params=params, timeout=20)
#         r.raise_for_status()
#         res = r.json()
#         results = res.get("results",[])
# 
#         # Log all results for debugging
#         LOG.info(f"AcoustID returned {len(results)} results:")
#         for i, result in enumerate(results[:3]):
#             score = result.get("score", 0.0)
#             recs = result.get("recordings", [])
#             if recs:
#                 title = recs[0].get("title", "Unknown")
#                 artists = recs[0].get("artists", [])
#                 artist = artists[0].get("name") if artists else "Unknown"
#                 LOG.info(f"  {i+1}. {artist} - {title} (confidence: {score:.3f})")
# 
#         # Filter by confidence score - only accept high confidence matches
#         high_confidence_results = [
#             result for result in results
#             if result.get("score", 0.0) >= 0.8  # 80% confidence threshold
#         ]
# 
#         if high_confidence_results:
#             # Use the highest confidence result
#             best_result = max(high_confidence_results, key=lambda x: x.get("score", 0.0))
#             recs = best_result.get("recordings",[])
#             confidence = best_result.get("score", 0.0)
# 
#             if recs:
#                 title = recs[0].get("title", filepath.stem)
#                 artists = recs[0].get("artists",[])
#                 artist = artists[0].get("name") if artists else "Unknown Artist"
#                 LOG.info(f"✓ High confidence match: {artist} - {title} (confidence: {confidence:.3f})")
#                 return (artist, title)
# 
#         # If no high-confidence matches, try medium confidence (60-80%)
#         medium_confidence_results = [
#             result for result in results
#             if 0.6 <= result.get("score", 0.0) < 0.8
#         ]
# 
#         if medium_confidence_results:
#             best_result = max(medium_confidence_results, key=lambda x: x.get("score", 0.0))
#             recs = best_result.get("recordings",[])
#             confidence = best_result.get("score", 0.0)
# 
#             if recs:
#                 title = recs[0].get("title", filepath.stem)
#                 artists = recs[0].get("artists",[])
#                 artist = artists[0].get("name") if artists else "Unknown Artist"
#                 LOG.warning(f"⚠ Medium confidence match: {artist} - {title} (confidence: {confidence:.3f})")
#                 return (artist, title)
# 
#         LOG.warning(f"No confident matches found (highest confidence: {max([r.get('score', 0.0) for r in results], default=0.0):.3f})")
# 
#     except Exception as e:
#         LOG.warning(f"AcoustID lookup failed: {e}")
# 
#     return ("Unknown Artist", filepath.stem)


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
            return "mps", "float16"    # Apple Silicon
        if torch.cuda.is_available():
            return "cuda", "float16"   # NVIDIA: fp16 ok
    except Exception:
        pass
    return "cpu", "float16"

def transcribe_whisperx(source_path: Path, model_name: str = "small") -> Dict:
    """
    Transcribe with WhisperX - force CPU for stability on Mac
    """
    try:
        import whisperx
        
        # FORCE CPU - MPS is broken in WhisperX
        device = "cpu"
        compute_type = "float32"
        
        LOG.info(f"Loading WhisperX model={model_name} device={device} compute_type={compute_type}")
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        
        result = model.transcribe(str(source_path))
        
        # Try alignment for word-level timestamps
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result.get("language", "en"),
                device=device  # Also CPU
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
                LOG.info("✓ Got word-level timestamps from WhisperX")
            elif aligned and aligned.get("segments"):
                result["segments"] = aligned["segments"]
        except Exception as e:
            LOG.warning(f"WhisperX alignment failed: {e}")
        
        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", [])
        }
        
    except Exception as e:
        LOG.warning(f"WhisperX failed: {e}; falling back to whisper.")
        
        try:
            import whisper
            model = whisper.load_model(model_name, device="cpu")
            res = model.transcribe(str(source_path), fp16=False)
            return {"text": res.get("text", ""), "segments": res.get("segments", [])}
        except Exception as e2:
            LOG.error(f"All transcription failed: {e2}")
            return {"text": "", "segments": []}



def analyze_and_score(source_path: Path,
                      stems_dir: Optional[Path],
                      transcription: dict,
                      topk: int = 12,
                      strategy: ScoringStrategy = VocalPriorityStrategy(),
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
        c["score"] = float(strategy.score(c, novelty_at))
        #start = c["start"]
        #dur = c["dur"]
        #novelty = novelty_at(start)
        #score = novelty * (dur ** 0.5)

        #if c["type"].startswith("vocal"):
        #    score *= 1.7
        #    label = (c.get("label","") or "").lower()
        #    # keyword boost
        #    if kw and any(k in label for k in kw):
        #        score *= 1.5
        #    # prefer non-stopwords-ish tokens
        #    if len(label) > 3:
        #        score *= 1.15


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
    strat: str,
    whisper_model: str
):
    if not source.exists():
        LOG.error(f"Source file not found: {source}")
        return

    LOG.info(f"Starting Track ID using {acoustid_key} as key")

    artist, title = identify_track_acoustid(source, acoustid_key)
    LOG.info(f"Results Acoustid -  Artist: {artist} Title: {title}")

    out_base.mkdir(parents=True, exist_ok=True)
    project_dir = out_base / f"{artist} - {title}"
    project_dir.mkdir(exist_ok=True)

    stems_dir = project_dir / "stems"

    # Check if stems already exist
    stems_exist = False
    if stems_dir.exists() and stems_dir.is_dir():
        # Check for common stem files
        expected_stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        existing_stems = [s for s in expected_stems if (stems_dir / s).exists()]

        if existing_stems:
            stems_exist = True
            LOG.info(f"Found existing stems ({len(existing_stems)} files) in {stems_dir}")
            LOG.info(f"Skipping stem separation - using existing stems: {existing_stems}")

    # Only separate if stems don't exist and skip_sep is False
    if not skip_sep and not stems_exist:
        LOG.info("Stems not found, running separation...")
        stems_dir = separate_stems_demucs(source, stems_dir)
        LOG.info(f"Separation completed. Stems at: {stems_dir}")
    elif skip_sep:
        LOG.info("Skipping stem separation due to --skip-separate flag")
        stems_dir = None if not stems_exist else stems_dir

    transcription = transcribe_whisperx(source, model_name=whisper_model)
    LOG.info("Transcription complete: %d segments", len(transcription.get("segments", [])))
    # Create your strategies
    selected_strategy = VocalPriorityStrategy()
    LOG.info(f"selecstrategy: {selected_strategy}")

    match strat:
        case "lyric":
            selected_strategy = LyricalImportanceStrategy()
        case "perc":
            selected_strategy = PercussionDrivenStrategy()
        case "balance":
            selected_strategy = BalancedDiversityStrategy()
        case "energy":
            selected_strategy = EnergyDominantStrategy()
        case "hybrid":
            vocal_priority = VocalPriorityStrategy()
            perc_driven = PercussionDrivenStrategy()
            # Blend them with weights
            selected_strategy = HybridScoringStrategy([
                (vocal_priority, 0.6),
                (perc_driven, 0.4)
            ])

    scored = analyze_and_score(
            source,
            stems_dir,
            transcription,
            topk,
            strategy=selected_strategy
    )

    LOG.info(f"scored strategy: {scored}")
   #  # Pass into your analyzer
   #  selected_hybrid = analyze_and_score(
   #      source,
   #      stems_dir,
   #      transcription,
   #      topk=topk,
   #      strategy=hybrid
   #  )

   #  LOG.info(f"ran hybrid strat: {selected_hybrid} ")

   #  selected = analyze_and_score(
   #          source,
   #          stems_dir,
   #          transcription,
   #          topk=topk,
   #          strategy=selected_hybrid
   #  )




    write_dir = write_chops_and_metadata(source, scored, project_dir)

    # NEW: save analysis + transcript for UI (minimal addition; no new args)
    save_analysis_and_transcript(project_dir, source, transcription, scored)

    LOG.info(f"Pipeline complete. Project at: {write_dir}")
