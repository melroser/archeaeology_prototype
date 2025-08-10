# from fastapi import FastAPI
# from fastapi.responses import HTMLResponse
# import uvicorn
# import logging
# 
# LOG = logging.getLogger("archaeologist.web_ui")
# app = FastAPI()
# 
# @app.get("/")
# def index():
#     return HTMLResponse("<h1>Archaeologist preview server</h1><p>Use the CLI to create a project and then open the project folder to preview clips.</p>")
# 
# def run_server(host="127.0.0.1", port=8000):
#     uvicorn.run(app, host=host, port=port)
# 

# 
# 
# def load_whisperx_model(model_name: str = "small", device: str = None):
#     """
#     Load WhisperX model with safe defaults for Apple Silicon and CPU/GPU.
#     Forces float32 on M-series to avoid unsupported fp16 ops.
#     """
#     import torch, whisperx
# 
#     if device is None:
#         if torch.backends.mps.is_available():
#             device = "mps"
#         elif torch.cuda.is_available():
#             device = "cuda"
#         else:
#             device = "cpu"
# 
#     compute_type = "float32"
#     if device == "cuda" and torch.cuda.is_available():
#         compute_type = "float16"
# 
#     print(f"[INFO] Loading WhisperX '{model_name}' on {device} ({compute_type})")
#     return whisperx.load_model(model_name, device=device, compute_type=compute_type)


# 
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from pathlib import Path
# import json
# 
# app = FastAPI()
# 
# CHOPS_DIR = Path(__file__).resolve().parent.parent / "output" / "chops"
# 
# @app.get("/api/chops")
# def list_chops():
#     return [
#         {"file": wav.name, "path": f"/chops/{wav.name}"}
#         for wav in CHOPS_DIR.glob("*.wav")
#     ]
# 
# # Serve audio files
# app.mount("/chops", StaticFiles(directory=CHOPS_DIR), name="chops")
# 
# # Serve frontend build
# frontend_path = Path(__file__).parent.parent / "frontend"
# app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")



from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn
import logging

LOG = logging.getLogger("archaeologist.web_ui")
app = FastAPI()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHOPS_DIR = BASE_DIR / "output" / "chops"
FRONTEND_BUILD = BASE_DIR / "frontend_build"

@app.get("/api/chops")
def list_chops():
    if not CHOPS_DIR.exists():
        return []
    return [
        {"file": wav.name, "path": f"/chops/{wav.name}"}
        for wav in CHOPS_DIR.glob("*.wav")
    ]

# Serve chops
app.mount("/chops", StaticFiles(directory=CHOPS_DIR), name="chops")

# Serve React frontend
if FRONTEND_BUILD.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")
else:
    @app.get("/")
    def index():
        return {"message": "Frontend not built yet. Run npm build in /frontend."}

def run_server(host="127.0.0.1", port=8000):
    uvicorn.run(app, host=host, port=port)
