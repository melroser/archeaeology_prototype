from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pathlib import Path
import uvicorn
import logging
import json

LOG = logging.getLogger("archaeologist.web_ui")
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

def list_projects():
    if not OUTPUT_DIR.exists():
        return []
    projects = []
    for p in OUTPUT_DIR.iterdir():
        if p.is_dir() and (p / "chops").exists():
            projects.append(p)
    projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return projects

@app.get("/api/projects")
def api_projects():
    return [{"name": p.name, "chops": len(list((p / "chops").glob("*.wav")))} for p in list_projects()]

@app.get("/api/chops")
def api_chops(project: str):
    proj = OUTPUT_DIR / project
    chops_dir = proj / "chops"
    if not chops_dir.exists():
        raise HTTPException(status_code=404, detail="Project or chops not found")
    files = sorted(chops_dir.glob("*.wav"))
    return [{"file": f.name, "path": f"/audio?project={project}&file={f.name}"} for f in files]

@app.get("/audio")
def audio(project: str, file: str):
    proj = OUTPUT_DIR / project
    path = (proj / "chops" / file).resolve()
    if OUTPUT_DIR not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="audio/wav")

@app.get("/source")
def source(project: str):
    proj = OUTPUT_DIR / project
    src_dir = proj / "source"
    if not src_dir.exists():
        raise HTTPException(status_code=404, detail="Source folder not found")
    candidates = sorted(src_dir.glob("*.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Source file not found")
    path = candidates[0].resolve()
    if OUTPUT_DIR not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    # naive media type; browser will handle common formats
    return FileResponse(str(path))

@app.get("/api/analysis")
def api_analysis(project: str):
    proj = OUTPUT_DIR / project
    j = proj / "analysis.json"
    if not j.exists():
        return JSONResponse({"duration": 0, "novelty": {"times": [], "values": []},
                             "vocal_activity": {"times": [], "values": []},
                             "epicness": {"times": [], "values": []},
                             "boundaries": [], "chops": []})
    with j.open("r", encoding="utf8") as f:
        return json.load(f)

@app.get("/api/library/top")
def api_library_top(limit: int = Query(50, ge=1, le=500)):
    rows = []
    for proj in list_projects():
        csvp = proj / "chops.csv"
        if not csvp.exists():
            continue
        try:
            import csv
            with csvp.open("r", encoding="utf8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        score = float(r.get("score", 0.0))
                    except Exception:
                        score = 0.0
                    rows.append({
                        "project": proj.name,
                        "file": r.get("file", ""),
                        "label": r.get("label", ""),
                        "type": r.get("type", ""),
                        "score": score,
                        "path": f"/audio?project={proj.name}&file={Path(r.get('file','')).name}",
                    })
        except Exception:
            continue
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]

@app.get("/api/search")
def api_search(q: str = Query(..., min_length=1)):
    ql = q.lower()
    hits = []

    for proj in list_projects():
        # transcript
        tjson = proj / "transcript" / "transcript.json"
        if tjson.exists():
            try:
                data = json.loads(tjson.read_text(encoding="utf8"))
                for s in data.get("segments", []):
                    text = (s.get("text", "") or s.get("word", "")).strip()
                    if ql in text.lower():
                        hits.append({
                            "project": proj.name,
                            "type": "transcript",
                            "text": text,
                            "start": float(s.get("start", 0.0)),
                            "end": float(s.get("end", 0.0)),
                            "jump": f"/source?project={proj.name}",
                        })
            except Exception:
                pass

        # chops
        csvp = proj / "chops.csv"
        if csvp.exists():
            try:
                import csv
                with csvp.open("r", encoding="utf8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        lab = (r.get("label", "") or "").lower()
                        if ql in lab:
                            hits.append({
                                "project": proj.name,
                                "type": "chop",
                                "text": r.get("label", ""),
                                "file": r.get("file", ""),
                                "path": f"/audio?project={proj.name}&file={Path(r.get('file','')).name}",
                            })
            except Exception:
                pass

    return hits[:200]

@app.get("/ui")
def ui():
    # Single-file UI: heat strip, boundary markers, chop list, search, library top
    return HTMLResponse("""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>ARCHAEOLOG1ST — Explorer</title>
<style>
  :root { --bg:#0a0a0a; --fg:#eaeaea; --hi:#00ff99; --hi2:#ff0066; --mut:#999; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--fg); }
  .wrap { max-width: 1100px; margin: 32px auto; padding: 0 16px; }
  h1 { font-weight:600; letter-spacing: .02em; margin: 0 0 16px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .col { flex:1 1 360px; min-width: 320px; }
  .panel { background:#101010; border:1px solid #191919; border-radius:12px; padding:12px; }
  select, input[type="text"], input[type="range"] {
    width:100%; background:#0f0f0f; border:1px solid #222; border-radius:8px; color:var(--fg); padding:8px;
  }
  button { background:transparent; color:var(--fg); border:1px solid var(--hi); border-radius:8px; padding:6px 10px; cursor:pointer; }
  button:hover { background:var(--hi); color:#000; }
  canvas { width:100%; height:100px; background:#0d0d0d; border:1px solid #1a1a1a; border-radius:8px; display:block; }
  .legend { display:flex; gap:12px; align-items:center; font-size:12px; color:var(--mut); margin-top:6px;}
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .list { max-height: 380px; overflow:auto; margin-top:8px; }
  .item { padding:6px 4px; border-bottom:1px solid #191919; display:flex; gap:8px; align-items:center; }
  .item small { color:var(--mut); }
  .pill { font-size:11px; border:1px solid #333; padding:2px 6px; border-radius:999px; color:#ccc; }
</style>
</head>
<body>
<div class="wrap">
  <h1>ARCHAEOLOG1ST — Explorer</h1>

  <div class="row">
    <div class="col panel">
      <div style="display:flex; gap:8px; align-items:center;">
        <select id="project"></select>
        <button id="refresh">Refresh</button>
      </div>
      <div style="margin:8px 0 6px; display:flex; gap:8px; align-items:center;">
        <label style="font-size:12px; color:#aaa;">Boundary density</label>
        <input type="range" id="density" min="0" max="100" value="60" />
      </div>
      <canvas id="heat"></canvas>
      <div class="legend">
        <span class="dot" style="background:var(--hi);"></span> epicness
        <span class="dot" style="background:var(--hi2);"></span> boundaries
      </div>
      <div style="margin-top:8px; display:flex; gap:8px; align-items:center;">
        <audio id="player" controls style="width:100%;"></audio>
      </div>
    </div>

    <div class="col panel">
      <input type="text" id="search" placeholder="Search keyword (e.g., wake up, suicide)" />
      <div class="list" id="results"></div>
    </div>

    <div class="col panel">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-weight:600;">Chops</div>
        <a href="/api/library/top" target="_blank" class="pill">Top library chops →</a>
      </div>
      <div class="list" id="chops"></div>
    </div>
  </div>
</div>

<script>
const $ = (s) => document.querySelector(s);
const projectSel = $("#project");
const heat = $("#heat");
const player = $("#player");
const density = $("#density");
const results = $("#results");
const chops = $("#chops");
const search = $("#search");

let analysis = null;
let currentProject = null;
let chopList = [];

async function loadProjects() {
  const res = await fetch("/api/projects");
  const data = await res.json();
  projectSel.innerHTML = data.map(p => `<option value="${p.name}">${p.name} (${p.chops})</option>`).join("");
  if (data.length) { currentProject = data[0].name; projectSel.value = currentProject; }
}

async function loadProject(name) {
  currentProject = name;
  player.src = `/source?project=${encodeURIComponent(name)}`;
  analysis = await (await fetch(`/api/analysis?project=${encodeURIComponent(name)}`)).json();
  chopList = await (await fetch(`/api/chops?project=${encodeURIComponent(name)}`)).json();
  drawHeat();
  renderChops();
}

function drawHeat() {
  const ctx = heat.getContext("2d");
  const w = heat.width = heat.clientWidth * devicePixelRatio;
  const h = heat.height = heat.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,w,h);

  if (!analysis || !analysis.epicness || !analysis.epicness.values.length) return;

  const times = analysis.epicness.times;
  const values = analysis.epicness.values;
  const dur = analysis.duration || (times.length ? times[times.length-1] : 0);

  // Density → threshold (higher density = more boundaries)
  const dens = Number(density.value) / 100;
  const threshold = 0.3 + 0.6 * (1 - dens);

  // Epicness bar
  for (let x=0; x<w; x++) {
    const t = (x / w) * dur;
    // nearest index
    let i = 0;
    // binary search-ish
    let lo=0, hi=times.length-1;
    while (lo < hi) {
      const mid = (lo+hi)>>1;
      if (times[mid] < t) lo = mid + 1; else hi = mid;
    }
    i = lo;
    const v = values[Math.max(0, Math.min(values.length-1, i))] || 0;
    const y = h - (v * h);
    ctx.strokeStyle = "#00ff99";
    ctx.beginPath();
    ctx.moveTo(x, h);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  // Boundaries
  const bounds = (analysis.boundaries || []).filter(t => t >= 0);
  ctx.strokeStyle = "#ff0066";
  bounds.forEach(t => {
    if (!dur) return;
    // only draw if above threshold region
    // find epicness at time t
    let i = 0;
    let lo=0, hi=times.length-1;
    while (lo < hi) {
      const mid = (lo+hi)>>1;
      if (times[mid] < t) lo = mid + 1; else hi = mid;
    }
    i = lo;
    const v = values[Math.max(0, Math.min(values.length-1, i))] || 0;
    if (v < threshold) return;

    const x = (t / dur) * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  });

  // Click-to-seek
  heat.onclick = (ev) => {
    const rect = heat.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * devicePixelRatio;
    const t = (x / w) * dur;
    if (player.src) {
      player.currentTime = t;
      player.play();
    }
  };
}

function renderChops() {
  chops.innerHTML = (chopList || []).map(c => `
    <div class="item">
      <button onclick="playChop('${c.path}')">▶</button>
      <div>
        <div>${c.file.replace('.wav','')}</div>
        <small>${currentProject}</small>
      </div>
    </div>
  `).join("");
}

window.playChop = function(url) {
  const a = new Audio(url);
  a.play();
};

async function doSearch(q) {
  if (!q.trim()) { results.innerHTML = ""; return; }
  const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
  const data = await res.json();
  results.innerHTML = data.map(hit => {
    if (hit.type === "chop") {
      return `
      <div class="item">
        <button onclick="playChop('${hit.path}')">▶</button>
        <div>
          <div>${hit.text || '(chop)'} <span class="pill">${hit.project}</span></div>
          <small>${hit.file}</small>
        </div>
      </div>`;
    } else {
      return `
      <div class="item">
        <button onclick="jumpTranscript('${hit.project}', ${hit.start || 0})">⇥</button>
        <div>
          <div>${hit.text} <span class="pill">${hit.project}</span></div>
          <small>${(hit.start||0).toFixed(2)}–${(hit.end||0).toFixed(2)}s</small>
        </div>
      </div>`;
    }
  }).join("");
}

window.jumpTranscript = async function(project, t) {
  if (currentProject !== project) {
    await loadProject(project);
  }
  if (player.src) {
    player.currentTime = t;
    player.play();
  }
};

$("#refresh").onclick = async () => {
  await loadProjects();
  if (projectSel.value) await loadProject(projectSel.value);
};
projectSel.onchange = (e) => loadProject(e.target.value);
density.oninput = drawHeat;
search.onkeydown = (e) => { if (e.key === "Enter") doSearch(search.value); };

(async function init(){
  await loadProjects();
  if (projectSel.value) await loadProject(projectSel.value);
})();
</script>
</body>
</html>
    """)

#from fastapi import FastAPI, HTTPException
#from fastapi.responses import FileResponse, HTMLResponse
#from pathlib import Path
#import uvicorn
#import logging
#
#LOG = logging.getLogger("archaeologist.web_ui")
#app = FastAPI()
#
#BASE_DIR = Path(__file__).resolve().parent.parent
#OUTPUT_DIR = BASE_DIR / "output"
#
#def _latest_chops_dir() -> Path:
#    if not OUTPUT_DIR.exists():
#        return OUTPUT_DIR / "chops"
#
#    candidates = []
#    for p in OUTPUT_DIR.iterdir():
#        if p.is_dir():
#            c = p / "chops"
#            if c.exists():
#                candidates.append(c)
#
#    if not candidates:
#        return OUTPUT_DIR / "chops"
#
#    # most recently modified project
#    return max(candidates, key=lambda d: d.stat().st_mtime)
#
#CHOPS_DIR = _latest_chops_dir()
#
#@app.get("/")
#def index():
#    return HTMLResponse("""
#      <h1>Archaeologist — Preview</h1>
#      <p>Use the CLI to process a track. Then query:</p>
#      <ul>
#        <li>/api/projects</li>
#        <li>/api/chops?project=Artist%20-%20Title</li>
#        <li>/audio?project=Artist%20-%20Title&file=0.000-0.750-vocal-word-wake_up-0.912.wav</li>
#      </ul>
#    """)
#
#@app.get("/api/projects")
#def api_projects():
#    return [{"name": p.name, "chops": len(list((p/"chops").glob("*.wav")))} for p in list_projects()]
#
#@app.get("/api/chops")
#def api_chops(project: str):
#    proj = OUTPUT_DIR / project
#    chops_dir = proj / "chops"
#    if not chops_dir.exists():
#        raise HTTPException(status_code=404, detail="Project or chops not found")
#    files = sorted(chops_dir.glob("*.wav"))
#    return [{"file": f.name,
#             "path": f"/audio?project={project}&file={f.name}"} for f in files]
#
#@app.get("/audio")
#def audio(project: str, file: str):
#    proj = OUTPUT_DIR / project
#    path = (proj / "chops" / file).resolve()
#    # prevent path traversal
#    if OUTPUT_DIR not in path.parents:
#        raise HTTPException(status_code=400, detail="Invalid path")
#    if not path.exists():
#        raise HTTPException(status_code=404, detail="File not found")
#    return FileResponse(str(path), media_type="audio/wav")
#
#def run_server(host="127.0.0.1", port=8000):
#    uvicorn.run(app, host=host, port=port)
#
#
#


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



#from fastapi import FastAPI
#from fastapi.staticfiles import StaticFiles
#from fastapi.responses import FileResponse
#from pathlib import Path
#import uvicorn
#import logging
#
#LOG = logging.getLogger("archaeologist.web_ui")
#app = FastAPI()
#
## Paths
#BASE_DIR = Path(__file__).resolve().parent.parent
#CHOPS_DIR = BASE_DIR / "output" / "chops"
#FRONTEND_BUILD = BASE_DIR / "frontend_build"
#
#@app.get("/api/chops")
#def list_chops():
#    if not CHOPS_DIR.exists():
#        return []
#    return [
#        {"file": wav.name, "path": f"/chops/{wav.name}"}
#        for wav in CHOPS_DIR.glob("*.wav")
#    ]
#
## Serve chops
#app.mount("/chops", StaticFiles(directory=CHOPS_DIR), name="chops")
#
## Serve React frontend
#if FRONTEND_BUILD.exists():
#    app.mount("/", StaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")
#else:
#    @app.get("/")
#    def index():
#        return {"message": "Frontend not built yet. Run npm build in /frontend."}
#
#def run_server(host="127.0.0.1", port=8000):
#    uvicorn.run(app, host=host, port=port)
