from __future__ import annotations

import os
import json
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, Form, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- Import your real pipeline/CLI (no stubs) ---
try:
    from archaeologist.pipeline import run_pipeline, write_chops_and_metadata, save_analysis_and_transcript
except Exception as e:
    raise RuntimeError(f"Failed to import pipeline.py: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("archaeologist")

app = FastAPI(title="Archaeologist API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./songs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_FILE = "analysis.json"
TRANSCRIPT_DIR = "transcript"
TRANSCRIPT_FILE = "transcript.json"

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

class Chop(BaseModel):
    start: float
    end: float

class SongMeta(BaseModel):
    id: str
    title: str
    duration: float
    audioUrl: str
    chops: List[Chop]

from urllib.parse import quote, unquote

def url_safe_id(project_dir: Path) -> str:
    return quote(project_dir.name, safe="")

def project_from_id(song_id: str) -> Path:
    name = unquote(song_id)
    proj = OUTPUT_DIR / name
    if not proj.exists() or not proj.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    return proj

def find_source_audio(proj: Path) -> Path:
    src_dir = proj / "source"
    if src_dir.exists():
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS and f.is_file():
                return f
    for f in sorted(proj.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            return f
    raise HTTPException(status_code=404, detail="Source audio not found")

def read_analysis(proj: Path) -> dict:
    af = proj / ANALYSIS_FILE
    if af.exists():
        with af.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def read_chops(proj: Path) -> List[Chop]:
    data = read_analysis(proj)
    chops_out: List[Chop] = []
    if "chops" in data and isinstance(data["chops"], list):
        for c in data["chops"]:
            s = float(c.get("start", 0.0))
            if "end" in c:
                e = float(c["end"])
            else:
                e = s + float(c.get("dur", 0.0))
            chops_out.append(Chop(start=s, end=e))
        return chops_out

    csv_path = proj / "chops.csv"
    if csv_path.exists():
        import csv as _csv
        with csv_path.open("r", encoding="utf-8") as cf:
            reader = _csv.DictReader(cf)
            for row in reader:
                s = float(row.get("start", "0") or 0)
                d = float(row.get("dur", "0") or 0)
                chops_out.append(Chop(start=s, end=s + d))
    return chops_out

def song_meta_from_project(proj: Path) -> SongMeta:
    data = read_analysis(proj)
    duration = float(data.get("duration", 0.0))
    chops = read_chops(proj)
    sid = url_safe_id(proj)
    return SongMeta(
        id=sid,
        title=proj.name,
        duration=duration,
        audioUrl=f"/songs/{sid}/audio",
        chops=chops,
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"--> {request.method} {request.url}")
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.body()
            if body:
                logger.info(f"Body: {body[:500]}{'...' if len(body) > 500 else ''}")
        except Exception as e:
            logger.warning(f"Could not read body: {e}")
    response = await call_next(request)
    logger.info(f"<-- {response.status_code} {request.url}")
    return response

@app.get("/songs", response_model=List[SongMeta])
def list_songs():
    projects = [p for p in OUTPUT_DIR.iterdir() if p.is_dir()]
    projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [song_meta_from_project(p) for p in projects]

@app.get("/songs/{song_id}", response_model=SongMeta)
def get_song(song_id: str):
    proj = project_from_id(song_id)
    return song_meta_from_project(proj)

@app.get("/songs/{song_id}/audio")
def get_song_audio(song_id: str):
    proj = project_from_id(song_id)
    src = find_source_audio(proj)
    return FileResponse(str(src), media_type="audio/*", filename=src.name)

@app.post("/songs/{song_id}/chops")
def save_chops(song_id: str, chops: List[Chop] = Body(...)):
    proj = project_from_id(song_id)
    src = find_source_audio(proj)

    selected = []
    for c in chops:
        s = float(c.start)
        e = float(c.end)
        d = max(0.0, e - s)
        selected.append({
            "start": s,
            "dur": d,
            "type": "manual",
            "label": "",
            "score": 1.0,
        })

    logger.info(f"Saving chops for {song_id}: {selected}")

    write_chops_and_metadata(src, selected, proj)

    transcript = {}
    tjson = proj / TRANSCRIPT_DIR / TRANSCRIPT_FILE
    if tjson.exists():
        with tjson.open("r", encoding="utf-8") as f:
            try:
                transcript = json.load(f)
            except Exception:
                transcript = {}

    save_analysis_and_transcript(proj, src, transcript, selected)

    return {"ok": True}

@app.post("/process", response_model=SongMeta)
def process_song(
    file: UploadFile,
    topk: Optional[int] = Form(12),
    num_chops: Optional[int] = Form(None),
    skip_sep: Optional[bool] = Form(True),
    acoustid_key: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form("small"),
):
    tmp_dir = OUTPUT_DIR / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
    with tmp_path.open("wb") as f:
        f.write(file.file.read())

    effective_topk = int(num_chops) if num_chops is not None else (int(topk) if topk is not None else 12)
    logger.info(f"Running pipeline on {tmp_path} with topk={effective_topk}, skip_sep={skip_sep}, whisper_model={whisper_model}")

    run_pipeline(
        source=tmp_path,
        out_base=OUTPUT_DIR,
        acoustid_key=acoustid_key,
        skip_sep=bool(skip_sep),
        topk=effective_topk,
        whisper_model=str(whisper_model or "small"),
    )

    projects = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name != "_uploads"]
    if not projects:
        raise HTTPException(status_code=500, detail="Pipeline produced no project")
    latest = max(projects, key=lambda p: p.stat().st_mtime)

    meta = song_meta_from_project(latest)
    logger.info(f"Processed new song: {meta}")
    return meta


#
#
#
#
#
#from __future__ import annotations
#
#import os
#import json
#import uuid
#from pathlib import Path
#from typing import List, Optional
#
#from fastapi import FastAPI, UploadFile, Form, HTTPException, Body
#from fastapi.middleware.cors import CORSMiddleware
#from fastapi.responses import FileResponse
#from pydantic import BaseModel
#
## --- Import your real pipeline/CLI (no stubs) ---
#try:
#    from archaeologist.pipeline import run_pipeline, write_chops_and_metadata, save_analysis_and_transcript
#except Exception as e:
#    raise RuntimeError(f"Failed to import pipeline.py: {e}")
#
#app = FastAPI(title="Archaeologist API", version="1.0")
#
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
#
#OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./songs")).resolve()
#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#ANALYSIS_FILE = "analysis.json"
#TRANSCRIPT_DIR = "transcript"
#TRANSCRIPT_FILE = "transcript.json"
#
#AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
#
#class Chop(BaseModel):
#    start: float
#    end: float
#
#class SongMeta(BaseModel):
#    id: str
#    title: str
#    duration: float
#    audioUrl: str
#    chops: List[Chop]
#
#from urllib.parse import quote, unquote
#
#def url_safe_id(project_dir: Path) -> str:
#    return quote(project_dir.name, safe="")
#
#def project_from_id(song_id: str) -> Path:
#    name = unquote(song_id)
#    proj = OUTPUT_DIR / name
#    if not proj.exists() or not proj.is_dir():
#        raise HTTPException(status_code=404, detail="Project not found")
#    return proj
#
#def find_source_audio(proj: Path) -> Path:
#    src_dir = proj / "source"
#    if src_dir.exists():
#        for f in sorted(src_dir.iterdir()):
#            if f.suffix.lower() in AUDIO_EXTS and f.is_file():
#                return f
#    for f in sorted(proj.rglob("*")):
#        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
#            return f
#    raise HTTPException(status_code=404, detail="Source audio not found")
#
#def read_analysis(proj: Path) -> dict:
#    af = proj / ANALYSIS_FILE
#    if af.exists():
#        with af.open("r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#def read_chops(proj: Path) -> List[Chop]:
#    data = read_analysis(proj)
#    chops_out: List[Chop] = []
#    if "chops" in data and isinstance(data["chops"], list):
#        for c in data["chops"]:
#            s = float(c.get("start", 0.0))
#            if "end" in c:
#                e = float(c["end"])
#            else:
#                e = s + float(c.get("dur", 0.0))
#            chops_out.append(Chop(start=s, end=e))
#        return chops_out
#
#    csv_path = proj / "chops.csv"
#    if csv_path.exists():
#        import csv as _csv
#        with csv_path.open("r", encoding="utf-8") as cf:
#            reader = _csv.DictReader(cf)
#            for row in reader:
#                s = float(row.get("start", "0") or 0)
#                d = float(row.get("dur", "0") or 0)
#                chops_out.append(Chop(start=s, end=s + d))
#    return chops_out
#
#def song_meta_from_project(proj: Path) -> SongMeta:
#    data = read_analysis(proj)
#    duration = float(data.get("duration", 0.0))
#    chops = read_chops(proj)
#    sid = url_safe_id(proj)
#    return SongMeta(
#        id=sid,
#        title=proj.name,
#        duration=duration,
#        audioUrl=f"/songs/{sid}/audio",
#        chops=chops,
#    )
#
#@app.get("/songs", response_model=List[SongMeta])
#def list_songs():
#    projects = [p for p in OUTPUT_DIR.iterdir() if p.is_dir()]
#    projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
#    return [song_meta_from_project(p) for p in projects]
#
#@app.get("/songs/{song_id}", response_model=SongMeta)
#def get_song(song_id: str):
#    proj = project_from_id(song_id)
#    return song_meta_from_project(proj)
#
#@app.get("/songs/{song_id}/audio")
#def get_song_audio(song_id: str):
#    proj = project_from_id(song_id)
#    src = find_source_audio(proj)
#    return FileResponse(str(src), media_type="audio/*", filename=src.name)
#
#@app.post("/songs/{song_id}/chops")
#def save_chops(song_id: str, chops: List[Chop] = Body(...)):
#    proj = project_from_id(song_id)
#    src = find_source_audio(proj)
#
#    selected = []
#    for c in chops:
#        s = float(c.start)
#        e = float(c.end)
#        d = max(0.0, e - s)
#        selected.append({
#            "start": s,
#            "dur": d,
#            "type": "manual",
#            "label": "",
#            "score": 1.0,
#        })
#
#    write_chops_and_metadata(src, selected, proj)
#
#    transcript = {}
#    tjson = proj / TRANSCRIPT_DIR / TRANSCRIPT_FILE
#    if tjson.exists():
#        with tjson.open("r", encoding="utf-8") as f:
#            try:
#                transcript = json.load(f)
#            except Exception:
#                transcript = {}
#
#    save_analysis_and_transcript(proj, src, transcript, selected)
#
#    return {"ok": True}
#
#@app.post("/process", response_model=SongMeta)
#def process_song(
#    file: UploadFile,
#    topk: Optional[int] = Form(12),
#    num_chops: Optional[int] = Form(None),
#    skip_sep: Optional[bool] = Form(True),
#    acoustid_key: Optional[str] = Form(None),
#    whisper_model: Optional[str] = Form("small"),
#):
#    tmp_dir = OUTPUT_DIR / "_uploads"
#    tmp_dir.mkdir(parents=True, exist_ok=True)
#    tmp_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
#    with tmp_path.open("wb") as f:
#        f.write(file.file.read())
#
#    effective_topk = int(num_chops) if num_chops is not None else (int(topk) if topk is not None else 12)
#    run_pipeline(
#        source=tmp_path,
#        out_base=OUTPUT_DIR,
#        acoustid_key=acoustid_key,
#        skip_sep=bool(skip_sep),
#        topk=effective_topk,
#        whisper_model=str(whisper_model or "small"),
#    )
#
#    projects = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name != "_uploads"]
#    if not projects:
#        raise HTTPException(status_code=500, detail="Pipeline produced no project")
#    latest = max(projects, key=lambda p: p.stat().st_mtime)
#
#    return song_meta_from_project(latest)
#
#
#
#from __future__ import annotations
#
#import os
#import json
#import uuid
#from pathlib import Path
#from typing import List, Optional
#
#from fastapi import FastAPI, UploadFile, Form, HTTPException
#from fastapi.middleware.cors import CORSMiddleware
#from fastapi.responses import FileResponse
#from pydantic import BaseModel
#
## --- Import your real pipeline/CLI (no stubs) ---
## We assume pipeline.py and cli.py sit on PYTHONPATH or next to this file.
## If they live elsewhere, adjust sys.path before importing.
#try:
#    from archaeologist.pipeline import run_pipeline, write_chops_and_metadata, save_analysis_and_transcript
#except Exception as e:
#    raise RuntimeError(f"Failed to import pipeline.py: {e}")
#
## -----------------------------------------------------------------------------
#
#app = FastAPI(title="Archaeologist API", version="1.0")
#
## CORS for local Next.js dev
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # lock down in prod
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
#
## Base output dir where pipeline writes projects (matches your CLI --out)
#OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./songs")).resolve()
#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#
#ANALYSIS_FILE = "analysis.json"
#TRANSCRIPT_DIR = "transcript"
#TRANSCRIPT_FILE = "transcript.json"
#
#AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
#
## ------------------ Models ------------------
#
#class Chop(BaseModel):
#    start: float
#    end: float
#
#class SongMeta(BaseModel):
#    id: str            # URL-safe id
#    title: str         # human name (project dir)
#    duration: float
#    audioUrl: str
#    chops: List[Chop]
#
## ------------------ Helpers ------------------
#
#from urllib.parse import quote, unquote
#
#def url_safe_id(project_dir: Path) -> str:
#    # encode folder name to be URL-safe (spaces etc.)
#    return quote(project_dir.name, safe="")
#
#def project_from_id(song_id: str) -> Path:
#    name = unquote(song_id)
#    proj = OUTPUT_DIR / name
#    if not proj.exists() or not proj.is_dir():
#        raise HTTPException(status_code=404, detail="Project not found")
#    return proj
#
#def find_source_audio(proj: Path) -> Path:
#    # Prefer project/source/<file>; fallback to any audio in project tree.
#    src_dir = proj / "source"
#    if src_dir.exists():
#        for f in sorted(src_dir.iterdir()):
#            if f.suffix.lower() in AUDIO_EXTS and f.is_file():
#                return f
#    # fallback
#    for f in sorted(proj.rglob("*")):
#        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
#            return f
#    raise HTTPException(status_code=404, detail="Source audio not found")
#
#def read_analysis(proj: Path) -> dict:
#    af = proj / ANALYSIS_FILE
#    if af.exists():
#        with af.open("r", encoding="utf-8") as f:
#            return json.load(f)
#    return {}
#
#def read_chops(proj: Path) -> List[Chop]:
#    # Prefer analysis.json -> "chops" (with start/end/dur fields)
#    data = read_analysis(proj)
#    chops_out: List[Chop] = []
#    if "chops" in data and isinstance(data["chops"], list):
#        for c in data["chops"]:
#            s = float(c.get("start", 0.0))
#            if "end" in c:
#                e = float(c["end"])
#            else:
#                e = s + float(c.get("dur", 0.0))
#            chops_out.append(Chop(start=s, end=e))
#        return chops_out
#
#    # Fallback: parse chops.csv if present
#    csv_path = proj / "chops.csv"
#    if csv_path.exists():
#        import csv as _csv
#        with csv_path.open("r", encoding="utf-8") as cf:
#            reader = _csv.DictReader(cf)
#            for row in reader:
#                s = float(row.get("start", "0") or 0)
#                d = float(row.get("dur", "0") or 0)
#                chops_out.append(Chop(start=s, end=s + d))
#    return chops_out
#
#def song_meta_from_project(proj: Path) -> SongMeta:
#    data = read_analysis(proj)
#    duration = float(data.get("duration", 0.0))
#    chops = read_chops(proj)
#    sid = url_safe_id(proj)
#    return SongMeta(
#        id=sid,
#        title=proj.name,
#        duration=duration,
#        audioUrl=f"/songs/{sid}/audio",
#        chops=chops,
#    )
#
## ------------------ Routes ------------------
#
#@app.get("/songs", response_model=List[SongMeta])
#def list_songs():
#    projects = [p for p in OUTPUT_DIR.iterdir() if p.is_dir()]
#    projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
#    return [song_meta_from_project(p) for p in projects]
#
#@app.get("/songs/{song_id}", response_model=SongMeta)
#def get_song(song_id: str):
#    proj = project_from_id(song_id)
#    return song_meta_from_project(proj)
#
#@app.get("/songs/{song_id}/audio")
#def get_song_audio(song_id: str):
#    proj = project_from_id(song_id)
#    src = find_source_audio(proj)
#    return FileResponse(str(src), media_type="audio/*", filename=src.name)
#
#@app.post("/songs/{song_id}/chops")
#def save_chops(song_id: str, chops: List[Chop]):
#    proj = project_from_id(song_id)
#    src = find_source_audio(proj)
#
#    # Convert incoming chops -> selected format expected by write_chops_and_metadata
#    selected = []
#    for c in chops:
#        s = float(c.start)
#        e = float(c.end)
#        d = max(0.0, e - s)
#        selected.append({
#            "start": s,
#            "dur": d,
#            "type": "manual",
#            "label": "",
#            "score": 1.0,
#        })
#
#    # Regenerate chops audio + CSV
#    write_chops_and_metadata(src, selected, proj)
#
#    # Update analysis.json (reuse existing transcript if available)
#    transcript = {}
#    tjson = proj / TRANSCRIPT_DIR / TRANSCRIPT_FILE
#    if tjson.exists():
#        with tjson.open("r", encoding="utf-8") as f:
#            try:
#                transcript = json.load(f)
#            except Exception:
#                transcript = {}
#    # This will recompute novelty/epicness and embed new chops
#    save_analysis_and_transcript(proj, src, transcript, selected)
#
#    return {"ok": True}
#
#@app.post("/process", response_model=SongMeta)
#def process_song(
#    file: UploadFile,
#    topk: Optional[int] = Form(12),                 # also accepts num_chops
#    num_chops: Optional[int] = Form(None),          # legacy UI field name
#    skip_sep: Optional[bool] = Form(True),
#    acoustid_key: Optional[str] = Form(None),
#    whisper_model: Optional[str] = Form("small"),
#):
#    """
#    Runs your real pipeline.run_pipeline on the uploaded file and registers the result.
#    Output project folder will be under OUTPUT_DIR named '<artist> - <title>'.
#    """
#    # persist the upload to a temp file
#    tmp_dir = OUTPUT_DIR / "_uploads"
#    tmp_dir.mkdir(parents=True, exist_ok=True)
#    tmp_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
#    with tmp_path.open("wb") as f:
#        f.write(file.file.read())
#
#    # Run real pipeline (no stubs)
#    # This will create OUTPUT_DIR/<artist - title>/...
#    effective_topk = int(num_chops) if num_chops is not None else (int(topk) if topk is not None else 12)
#    run_pipeline(
#        source=tmp_path,
#        out_base=OUTPUT_DIR,
#        acoustid_key=acoustid_key,
#        skip_sep=bool(skip_sep),
#        topk=effective_topk,
#        whisper_model=str(whisper_model or "small"),
#    )
#
#    # Find the newest project created
#    projects = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name != "_uploads"]
#    if not projects:
#        raise HTTPException(status_code=500, detail="Pipeline produced no project")
#    latest = max(projects, key=lambda p: p.stat().st_mtime)
#
#    # Return SongMeta for the new project
#    return song_meta_from_project(latest)
#
# 
# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List, Optional
# import os, uuid, json, subprocess, tempfile, shutil
# 
# app = FastAPI()
# 
# SONGS_DIR = "./songs"
# CHOPS_FILE = "chops.json"
# os.makedirs(SONGS_DIR, exist_ok=True)
# 
# class Chop(BaseModel):
#     start: float
#     end: float
# 
# class SongMeta(BaseModel):
#     id: str
#     title: str
#     duration: float
#     audioUrl: str
#     chops: List[Chop]
# 
# songs_index: dict[str, SongMeta] = {}
# 
# def load_index():
#     global songs_index
#     songs_index = {}
#     for song_id in os.listdir(SONGS_DIR):
#         song_path = os.path.join(SONGS_DIR, song_id)
#         if not os.path.isdir(song_path):
#             continue
#         chops_path = os.path.join(song_path, CHOPS_FILE)
#         audio_path = None
#         for f in os.listdir(song_path):
#             if f.lower().endswith((".wav",".mp3",".flac",".ogg")):
#                 audio_path = os.path.join(song_path, f)
#                 break
#         if not audio_path:
#             continue
#         if os.path.exists(chops_path):
#             with open(chops_path) as f:
#                 chops = [Chop(**c) for c in json.load(f)]
#         else:
#             chops = []
#         songs_index[song_id] = SongMeta(
#             id=song_id,
#             title=song_id,
#             duration=0.0,
#             audioUrl=f"/songs/{song_id}/audio",
#             chops=chops,
#         )
# 
# load_index()
# 
# @app.get("/songs", response_model=List[SongMeta])
# def list_songs():
#     return list(songs_index.values())
# 
# @app.get("/songs/{song_id}", response_model=SongMeta)
# def get_song(song_id: str):
#     return songs_index[song_id]
# 
# @app.get("/songs/{song_id}/audio")
# def get_song_audio(song_id: str):
#     song_path = os.path.join(SONGS_DIR, song_id)
#     for f in os.listdir(song_path):
#         if f.lower().endswith((".wav",".mp3",".flac",".ogg")):
#             return FileResponse(os.path.join(song_path, f))
#     return {"error": "No audio found"}
# 
# @app.post("/songs/{song_id}/chops")
# def save_chops(song_id: str, chops: List[Chop]):
#     song_path = os.path.join(SONGS_DIR, song_id)
#     os.makedirs(song_path, exist_ok=True)
#     with open(os.path.join(song_path, CHOPS_FILE), "w") as f:
#         json.dump([c.dict() for c in chops], f)
#     songs_index[song_id].chops = chops
#     return {"ok": True}
# 
# @app.post("/process")
# def process_song(file: UploadFile, output_dir: Optional[str] = Form(None), num_chops: Optional[int] = Form(None)):
#     song_id = str(uuid.uuid4())
#     song_path = os.path.join(SONGS_DIR, song_id)
#     os.makedirs(song_path, exist_ok=True)
#     out_audio_path = os.path.join(song_path, file.filename)
#     with open(out_audio_path, "wb") as f:
#         f.write(file.file.read())
# 
#     # Use CLI to process the file
#     # Assume cli.py is executable: `python cli.py --input <file> --output <dir> --num_chops <n>`
#     cmd = ["python", "cli.py", "--input", out_audio_path, "--output", song_path]
#     if num_chops:
#         cmd.extend(["--num_chops", str(num_chops)])
#     if output_dir:
#         # Save results to specified directory as well
#         os.makedirs(output_dir, exist_ok=True)
#         cmd.extend(["--extra_output", output_dir])
#     subprocess.run(cmd, check=True)
# 
#     # Load chops.json generated by pipeline/cli
#     chops_path = os.path.join(song_path, CHOPS_FILE)
#     if os.path.exists(chops_path):
#         with open(chops_path) as f:
#             chops = [Chop(**c) for c in json.load(f)]
#     else:
#         chops = []
# 
#     songs_index[song_id] = SongMeta(
#         id=song_id,
#         title=file.filename,
#         duration=0.0,
#         audioUrl=f"/songs/{song_id}/audio",
#         chops=chops,
#     )
#     return songs_index[song_id]
# 
