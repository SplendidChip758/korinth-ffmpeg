"""
Korinth ffmpeg service  (v4)
----------------------------
HTTP wrapper around ffmpeg for the Korinth Industries video pipelines.

SHORTS path (unchanged since v2):
  PUT  /clip/{job}/{index}        raw mp4 bytes
  PUT  /narration/{job}           whole-video voiceover, base64
  POST /concat/{job}              join + mux one narration track

LONG-FORM path (v4):
  PUT  /image64/{job}/{index}     base64 PNG + motion preset
  PUT  /clip64/{job}/{index}      base64 mp4
  POST /narrate/{job}/{index}     text -> Gemini TTS -> wav, returns MEASURED duration
  POST /assemble/{job}            per-segment timing, Ken Burns, concat, mux

  GET  /lastframe/{job}/{index}   last frame as base64 PNG (chaining, unused on GEAP)
  GET  /health · GET /jobs · DELETE /job/{job}

Why narration is generated here rather than in n8n
  /assemble needs each segment's REAL duration to time the Ken Burns move and
  trim the clips. Generating the audio here means the wav is written straight
  into the job directory and probed on local disk - the duration is measured,
  never estimated. That removes the words-per-second guesswork that has needed
  recalibrating twice on the shorts pipeline.

Auth
  Optional. If X_AUTH_TOKEN is set, every request except /health must send a
  matching X-Auth-Token header.

Version
  Reported by /health from KORINTH_VERSION / KORINTH_GIT_SHA, which install.sh
  writes into the env file on every deploy. Do not hardcode a version here -
  the deployed commit is the source of truth.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", "/var/lib/korinth-ffmpeg"))
# X_AUTH_TOKEN is what install.sh generates and what the n8n Header Auth
# credential is built around. SERVICE_TOKEN is accepted as a fallback so a box
# still running the old self-contained installer's env file keeps its auth
# instead of silently coming up wide open.
SERVICE_TOKEN = (os.environ.get("X_AUTH_TOKEN")
                 or os.environ.get("SERVICE_TOKEN", "")).strip()
MAX_JOB_AGE_SECONDS = int(os.environ.get("MAX_JOB_AGE_SECONDS", 6 * 60 * 60))
MAX_CLIP_BYTES = int(os.environ.get("MAX_CLIP_BYTES", 200 * 1024 * 1024))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", 1800))

AMBIENT_VOLUME = os.environ.get("AMBIENT_VOLUME", "0.30")
PAD_SECONDS = float(os.environ.get("PAD") or os.environ.get("PAD_SECONDS") or "0.4")
OUT_FPS = int(os.environ.get("OUT_FPS", "25"))
OUT_W, OUT_H = 1920, 1080
# Imagen caps at 2K, so stills arrive ~2048x1152. Upscaling to 4K before
# zoompan is what gives the Ken Burns move room to travel - cropping into a
# 2K source for a 1080p output leaves only ~1.07x and goes soft immediately.
WORK_W, WORK_H = 3840, 2160

# Vertex AI (service account), not the AI Studio API-key surface — GEAP's GCP
# project only issues service account keys, not GEMINI_API_KEY values. The
# project id lives in the key file itself, so GCP_PROJECT_ID is only needed to
# override it (e.g. calling Vertex in a different project than the one that
# issued the key) — leave it unset in the normal case.
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "").strip()
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1").strip()
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-3.1-flash-tts-preview")
TTS_VOICE = os.environ.get("TTS_VOICE", "Orus")
TTS_STYLE = os.environ.get(
    "TTS_STYLE",
    "Read this at a brisk, clipped pace, like a control-room log entry. "
    "Cold, flat, professional. Do not linger or pause between sentences: ",
)
TTS_RATE = 24000
TTS_CHANNELS = 1


def _version() -> str:
    v = os.environ.get("KORINTH_VERSION", "").strip()
    if v:
        return v
    try:
        return (BASE_DIR / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"


VERSION = _version()
GIT_SHA = os.environ.get("KORINTH_GIT_SHA", "").strip() or "unknown"

SAFE_JOB = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MOTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right", "kenburns")

TTS_CONFIGURED = bool(GOOGLE_APPLICATION_CREDENTIALS)

app = FastAPI(title="Korinth ffmpeg service", version=VERSION)

# Loaded lazily (not at import time) so a box with TTS unconfigured can still
# start and serve every other route; /narrate is what surfaces the error.
# /narrate is a sync endpoint, so FastAPI runs it in a threadpool and two
# requests really can land here at once - hence the lock.
_vertex_credentials = None
_vertex_project_id = GCP_PROJECT_ID or None
_vertex_lock = threading.Lock()


def _load_vertex_credentials() -> service_account.Credentials:
    """Parses the key file and caches it. No network call - safe to use from /health."""
    global _vertex_credentials, _vertex_project_id
    if not TTS_CONFIGURED:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_APPLICATION_CREDENTIALS must be set in /etc/korinth-ffmpeg.env")
    if _vertex_credentials is None:
        with _vertex_lock:
            if _vertex_credentials is None:
                try:
                    creds = service_account.Credentials.from_service_account_file(
                        GOOGLE_APPLICATION_CREDENTIALS,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"])
                except (OSError, ValueError) as e:
                    raise HTTPException(status_code=500,
                                        detail=f"failed to load service account credentials: {e}")
                _vertex_project_id = GCP_PROJECT_ID or creds.project_id
                _vertex_credentials = creds
    return _vertex_credentials


def _vertex_access_token() -> str:
    creds = _load_vertex_credentials()
    # Access tokens are short-lived; refresh() is a no-op if the cached one
    # still has enough lifetime left, so it's cheap to call on every request.
    if not creds.valid:
        with _vertex_lock:
            if not creds.valid:
                try:
                    creds.refresh(GoogleAuthRequest())
                except Exception as e:
                    raise HTTPException(status_code=502,
                                        detail=f"failed to refresh service account token: {e}")
    return creds.token


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def check_auth(request: Request) -> None:
    if not SERVICE_TOKEN:
        return
    if request.headers.get("x-auth-token", "") != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="bad or missing X-Auth-Token")


def job_dir(job: str) -> Path:
    if not SAFE_JOB.match(job):
        raise HTTPException(status_code=400, detail="invalid job id")
    return DATA_DIR / job


def sweep_old_jobs() -> None:
    now = time.time()
    if not DATA_DIR.exists():
        return
    for entry in DATA_DIR.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > MAX_JOB_AGE_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def run(cmd, cwd) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg/ffprobe not installed")


def ffmpeg_version() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        return "NOT FOUND"
    try:
        proc = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
        first = (proc.stdout or "").splitlines()[0]
        return first.split()[2] if len(first.split()) > 2 else first
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def probe_duration(path: Path) -> float:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", path.name], cwd=path.parent)
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return 0.0


def has_audio(path: Path) -> bool:
    proc = run(["ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", path.name],
               cwd=path.parent)
    return bool((proc.stdout or "").strip())


def decode_b64(payload: dict, field: str) -> bytes:
    b64 = payload.get(field) or ""
    if not b64:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    try:
        data = base64.b64decode(b64, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field} is not valid base64")
    if not data:
        raise HTTPException(status_code=400, detail=f"{field} decoded empty")
    if len(data) > MAX_CLIP_BYTES:
        raise HTTPException(status_code=413, detail=f"{field} too large")
    return data


def segment_indices(d: Path) -> list:
    """
    Segment indices present on disk. Only bare numeric stems count - a stray
    `000_last.png` orphaned by a failed /lastframe call would otherwise reach
    int() and take the whole assemble down with an uncaught ValueError.
    """
    found = set()
    for pattern in ("[0-9]*.mp4", "[0-9]*.png"):
        for p in d.glob(pattern):
            if p.stem.isdigit():
                found.add(int(p.stem))
    return sorted(found)


class NarrateBody(BaseModel):
    text: str = ""
    voice: Optional[str] = None
    style: Optional[str] = None


async def json_body(request: Request) -> dict:
    try:
        return json.loads(await request.body() or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="body must be JSON")


def motion_filter(motion: str, frames: int) -> str:
    """
    Ken Burns preset. Applied AFTER an upscale to 3840x2160, so the zoom always
    crops into an oversized source and never has to invent detail.
    `on` is the output frame number; d must equal the total frame count.
    """
    f = max(frames, 2)
    centre = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    tail = f":d={f}:s={OUT_W}x{OUT_H}:fps={OUT_FPS}"

    if motion == "zoom_out":
        return f"zoompan=z='max(1.12-0.12*on/{f},1.0)':{centre}{tail}"
    if motion == "pan_left":
        return (f"zoompan=z='1.08':x='(iw-iw/zoom)*(1-on/{f})'"
                f":y='ih/2-(ih/zoom/2)'{tail}")
    if motion == "pan_right":
        return (f"zoompan=z='1.08':x='(iw-iw/zoom)*(on/{f})'"
                f":y='ih/2-(ih/zoom/2)'{tail}")
    if motion == "kenburns":
        # slow zoom with a slight diagonal drift
        return (f"zoompan=z='min(1+0.10*on/{f},1.10)'"
                f":x='iw/2-(iw/zoom/2)+(iw*0.02)*(on/{f})'"
                f":y='ih/2-(ih/zoom/2)-(ih*0.02)*(on/{f})'{tail}")
    # zoom_in / default
    return f"zoompan=z='min(1+0.12*on/{f},1.12)':{centre}{tail}"


VIDEO_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-r", str(OUT_FPS), "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


# --------------------------------------------------------------------------
# health / listing
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    ff = ffmpeg_version()
    gcp_project = _vertex_project_id
    if gcp_project is None and TTS_CONFIGURED:
        try:
            gcp_project = _load_vertex_credentials().project_id
        except HTTPException:
            pass  # bad/missing key file - surfaced properly by /narrate
    return {
        "status": "ok" if shutil.which("ffmpeg") and shutil.which("ffprobe") else "degraded",
        "version": VERSION,
        "git_sha": GIT_SHA,
        "ffmpeg": ff,
        "ffprobe": "ok" if shutil.which("ffprobe") else "NOT FOUND",
        "auth": bool(SERVICE_TOKEN),
        "tts_configured": TTS_CONFIGURED,
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "gcp_project": gcp_project,
        "gcp_location": GCP_LOCATION,
    }


@app.get("/jobs")
def jobs(request: Request) -> dict:
    check_auth(request)
    if not DATA_DIR.exists():
        return {"jobs": []}
    out = []
    for entry in sorted(DATA_DIR.iterdir()):
        if entry.is_dir():
            out.append({
                "job": entry.name,
                "clips": sorted(p.name for p in entry.glob("[0-9]*.mp4")),
                "images": sorted(p.name for p in entry.glob("[0-9]*.png")),
                "narrations": sorted(p.name for p in entry.glob("narr_*.wav")),
                "narration": (entry / "narration.meta").exists(),
            })
    return {"jobs": out}


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

def _store(job: str, index: int, data: bytes, ext: str) -> dict:
    if index < 0 or index > 999:
        raise HTTPException(status_code=400, detail="index out of range")
    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    # Zero-padded so lexical order matches segment order.
    (d / f"{index:03d}.{ext}").write_bytes(data)
    return {"job": job, "index": index, "bytes": len(data)}


@app.put("/clip/{job}/{index}")
async def put_clip(job: str, index: int, request: Request) -> dict:
    check_auth(request)
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    if len(body) > MAX_CLIP_BYTES:
        raise HTTPException(status_code=413, detail="clip too large")
    return _store(job, index, body, "mp4")


@app.put("/clip64/{job}/{index}")
async def put_clip64(job: str, index: int, request: Request) -> dict:
    check_auth(request)
    return _store(job, index, decode_b64(await json_body(request), "video_base64"), "mp4")


@app.put("/image64/{job}/{index}")
async def put_image64(job: str, index: int, request: Request, motion: str = "kenburns") -> dict:
    """Body (JSON): {"image_base64": "..."} · query: ?motion=zoom_in"""
    check_auth(request)
    if motion not in MOTIONS:
        motion = "kenburns"
    res = _store(job, index, decode_b64(await json_body(request), "image_base64"), "png")
    (job_dir(job) / f"{index:03d}.motion").write_text(motion, encoding="utf-8")
    res["motion"] = motion
    return res


@app.get("/lastframe/{job}/{index}")
def last_frame(job: str, index: int, request: Request) -> dict:
    check_auth(request)
    d = job_dir(job)
    clip = d / f"{index:03d}.mp4"
    if not clip.exists():
        raise HTTPException(status_code=404, detail=f"clip {index} not found for job {job}")
    out = d / f"{index:03d}_last.png"
    proc = run(["ffmpeg", "-y", "-sseof", "-1", "-i", clip.name,
                "-update", "1", "-q:v", "2", out.name], cwd=d)
    if proc.returncode != 0 or not out.exists():
        raise HTTPException(status_code=500,
                            detail=f"frame extract failed: {(proc.stderr or '')[-1500:]}")
    data = out.read_bytes()
    out.unlink(missing_ok=True)
    return {"job": job, "index": index, "mime_type": "image/png",
            "bytes": len(data), "image_base64": base64.b64encode(data).decode("ascii")}


# --------------------------------------------------------------------------
# narration
# --------------------------------------------------------------------------

@app.post("/narrate/{job}/{index}")
def narrate(job: str, index: int, request: Request, body: NarrateBody) -> dict:
    """
    Body (JSON): {"text": "...", "voice": "Orus" (optional), "style": "..." (optional)}
    Generates the segment's voiceover via Gemini TTS, writes it as a wav in the
    job directory, and returns its MEASURED duration.

    Deliberately a sync def: this makes a blocking HTTP call and then shells out
    to ffmpeg. As `async def` both of those stall the event loop for the whole
    request, so nothing else the service is doing can proceed. Sync handlers run
    in FastAPI's threadpool instead, which is also what /assemble does.
    """
    check_auth(request)

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice = body.voice or TTS_VOICE
    style = body.style if body.style is not None else TTS_STYLE

    token = _vertex_access_token()

    body = {
        # Vertex AI's generateContent rejects a content entry with no role
        # ("Please use a valid role: user, model.") - the AI Studio endpoint
        # this was ported from defaults it, Vertex doesn't.
        "contents": [{"role": "user", "parts": [{"text": style + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    url = (f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{_vertex_project_id}"
           f"/locations/{GCP_LOCATION}/publishers/google/models/{TTS_MODEL}:generateContent")

    try:
        with httpx.Client(timeout=180.0) as client:
            r = client.post(url, json=body,
                            headers={"Authorization": f"Bearer {token}",
                                     "Content-Type": "application/json"})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"TTS request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"TTS returned {r.status_code}: {r.text[:400]}")

    try:
        b64 = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    except (KeyError, IndexError, ValueError):
        raise HTTPException(status_code=502,
                            detail=f"unexpected TTS response: {r.text[:400]}")

    pcm = base64.b64decode(b64)
    if not pcm:
        raise HTTPException(status_code=502, detail="TTS returned empty audio")

    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    raw = d / f"narr_{index:03d}.raw"
    wav = d / f"narr_{index:03d}.wav"
    raw.write_bytes(pcm)

    # Gemini returns headerless s16le PCM; wrap it so everything downstream
    # (and ffprobe) can treat it as an ordinary audio file.
    proc = run(["ffmpeg", "-y", "-f", "s16le", "-ar", str(TTS_RATE),
                "-ac", str(TTS_CHANNELS), "-i", raw.name,
                "-c:a", "pcm_s16le", wav.name], cwd=d)
    raw.unlink(missing_ok=True)
    if proc.returncode != 0 or not wav.exists():
        raise HTTPException(status_code=500,
                            detail=f"wav wrap failed: {(proc.stderr or '')[-1000:]}")

    dur = probe_duration(wav)
    return {"job": job, "index": index, "bytes": len(pcm),
            "voice": voice, "duration_seconds": round(dur, 3)}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

@app.post("/assemble/{job}")
def assemble(job: str, request: Request, allow_silent: int = 0,
             allow_gaps: int = 0, expect: int = 0) -> Response:
    check_auth(request)
    sweep_old_jobs()

    d = job_dir(job)
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"no such job {job}")

    indices = segment_indices(d)
    if not indices:
        raise HTTPException(status_code=404, detail=f"no segments stored for job {job}")

    # A missing segment is invisible in the output - the video just loses a
    # story beat and nothing errors. Both checks below exist to make that loud.
    if expect and len(indices) != expect:
        raise HTTPException(
            status_code=409,
            detail=f"expected {expect} segments, found {len(indices)}: {indices}. "
                   f"A generation step failed without stopping the run.")

    gaps = [n for n in range(indices[-1] + 1) if n not in indices]
    if gaps and not allow_gaps:
        raise HTTPException(
            status_code=409,
            detail=f"segment indices are not contiguous, missing: {gaps}. "
                   f"Re-run those segments, or pass ?allow_gaps=1 to render anyway.")

    missing = [i for i in indices if not (d / f"narr_{i:03d}.wav").exists()]
    if missing and not allow_silent:
        # A silent segment is worse than a failed run - it ships broken and
        # nothing errors. Fail here, before n8n reaches the upload node.
        raise HTTPException(
            status_code=409,
            detail=f"segments missing narration: {missing}. "
                   f"Re-run /narrate for those, or pass ?allow_silent=1 to inspect.")

    images = videos = narrations = 0
    parts = []

    for i in indices:
        png, mp4 = d / f"{i:03d}.png", d / f"{i:03d}.mp4"
        wav = d / f"narr_{i:03d}.wav"
        seg = f"seg_{i:03d}.mp4"

        if wav.exists():
            dur = probe_duration(wav) + PAD_SECONDS
            narrations += 1
        else:
            dur = 5.0
        dur = max(dur, 1.0)
        frames = int(round(dur * OUT_FPS))

        # Every audio path below is padded to the segment's full length. The
        # final concat is a stream copy, so an audio track even a fraction
        # shorter than its video makes the voiceover creep ahead of the
        # picture, once per segment, cumulatively.
        if png.exists():
            images += 1
            motion_file = d / f"{i:03d}.motion"
            motion = motion_file.read_text(encoding="utf-8").strip() if motion_file.exists() else "kenburns"
            vf = (f"[0:v]scale={WORK_W}:{WORK_H}:flags=lanczos,"
                  f"setsar=1,{motion_filter(motion, frames)}[v]")

            cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(OUT_FPS),
                   "-t", f"{dur:.3f}", "-i", png.name]
            if wav.exists():
                cmd += ["-i", wav.name,
                        "-filter_complex", f"{vf};[1:a]apad[a]",
                        "-map", "[v]", "-map", "[a]"]
            else:
                cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                        "-i", "anullsrc=r=48000:cl=stereo",
                        "-filter_complex", vf, "-map", "[v]", "-map", "1:a"]
            cmd += ["-t", f"{dur:.3f}"] + VIDEO_ARGS + [seg]

        elif mp4.exists():
            videos += 1
            clip_has_audio = has_audio(mp4)
            # tpad holds the last frame if the clip is shorter than its
            # narration; -t trims it if longer.
            chain = [f"[0:v]tpad=stop_mode=clone:stop_duration={dur:.3f},setsar=1[v]"]
            cmd = ["ffmpeg", "-y", "-i", mp4.name]

            if wav.exists() and clip_has_audio:
                chain.append(f"[0:a]volume={AMBIENT_VOLUME},apad[amb]")
                chain.append("[1:a]apad[vo]")
                chain.append("[amb][vo]amix=inputs=2:duration=longest:normalize=0[a]")
                cmd += ["-i", wav.name]
                maps = ["-map", "[v]", "-map", "[a]"]
            elif wav.exists():
                chain.append("[1:a]apad[a]")
                cmd += ["-i", wav.name]
                maps = ["-map", "[v]", "-map", "[a]"]
            elif clip_has_audio:
                chain.append(f"[0:a]volume={AMBIENT_VOLUME},apad[a]")
                maps = ["-map", "[v]", "-map", "[a]"]
            else:
                cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                        "-i", "anullsrc=r=48000:cl=stereo"]
                maps = ["-map", "[v]", "-map", "1:a"]

            cmd += ["-filter_complex", ";".join(chain)] + maps
            cmd += ["-t", f"{dur:.3f}"] + VIDEO_ARGS + [seg]
        else:
            continue

        proc = run(cmd, cwd=d)
        if proc.returncode != 0 or not (d / seg).exists():
            raise HTTPException(status_code=500,
                                detail=f"segment {i} render failed: {(proc.stderr or '')[-1500:]}")
        parts.append(seg)

    if not parts:
        raise HTTPException(status_code=500, detail="no segments rendered")

    (d / "concat.txt").write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
    final = "final.mp4"
    # Every segment was written with identical stream parameters above, so a
    # stream-copy concat is safe here and avoids a second full re-encode.
    proc = run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                "-c", "copy", "-movflags", "+faststart", final], cwd=d)
    if proc.returncode != 0 or not (d / final).exists():
        raise HTTPException(status_code=500,
                            detail=f"concat failed: {(proc.stderr or '')[-1500:]}")

    total = probe_duration(d / final)
    data = (d / final).read_bytes()
    shutil.rmtree(d, ignore_errors=True)

    return Response(
        content=data,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{job}.mp4"',
            "X-Segment-Count": str(len(parts)),
            "X-Image-Count": str(images),
            "X-Clip-Count": str(videos),
            "X-Narration-Count": str(narrations),
            "X-Duration": f"{total:.2f}",
        },
    )


# --------------------------------------------------------------------------
# shorts path (unchanged)
# --------------------------------------------------------------------------

@app.put("/narration/{job}")
async def put_narration(job: str, request: Request) -> dict:
    check_auth(request)
    payload = await json_body(request)
    audio = decode_b64(payload, "audio_base64")
    fmt = str(payload.get("format") or "pcm_s16le").lower()
    if fmt not in ("pcm_s16le", "wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be pcm_s16le, wav or mp3")
    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    ext = {"pcm_s16le": "raw", "wav": "wav", "mp3": "mp3"}[fmt]
    (d / f"narration.{ext}").write_bytes(audio)
    (d / "narration.meta").write_text(json.dumps({
        "format": fmt, "file": f"narration.{ext}",
        "sample_rate": int(payload.get("sample_rate") or 24000),
        "channels": int(payload.get("channels") or 1),
    }), encoding="utf-8")
    return {"job": job, "bytes": len(audio), "format": fmt}


@app.post("/concat/{job}")
def concat(job: str, request: Request) -> Response:
    check_auth(request)
    sweep_old_jobs()
    d = job_dir(job)
    clips = sorted(d.glob("[0-9]*.mp4")) if d.exists() else []
    if not clips:
        raise HTTPException(status_code=404, detail=f"no clips stored for job {job}")

    (d / "concat.txt").write_text("".join(f"file '{c.name}'\n" for c in clips), encoding="utf-8")
    joined = "joined.mp4"
    proc = run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", "-r", "24", "-pix_fmt", "yuv420p",
                joined], cwd=d)
    if proc.returncode != 0 or not (d / joined).exists():
        raise HTTPException(status_code=500, detail=f"concat failed: {(proc.stderr or '')[-2000:]}")

    meta_path = d / "narration.meta"
    final = "final.mp4"
    narration_used = False
    out_duration = probe_duration(d / joined)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pcm_args = []
        if meta["format"] == "pcm_s16le":
            pcm_args = ["-f", "s16le", "-ar", str(meta["sample_rate"]), "-ac", str(meta["channels"])]
        video_dur = probe_duration(d / joined)
        narr_probe = run(["ffprobe", "-v", "error"] + pcm_args +
                         ["-show_entries", "format=duration", "-of", "csv=p=0", meta["file"]], cwd=d)
        try:
            narr_dur = float((narr_probe.stdout or "").strip())
        except ValueError:
            narr_dur = 0.0
        target = max(video_dur, narr_dur) + 0.4
        pad = max(0.0, target - video_dur)
        filt = (f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.2f}[v];"
                f"[0:a]volume={AMBIENT_VOLUME}[amb];[1:a]apad[vo];"
                f"[amb][vo]amix=inputs=2:duration=longest:normalize=0[a]")
        proc = run(["ffmpeg", "-y", "-i", joined, *pcm_args, "-i", meta["file"],
                    "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
                    "-t", f"{target:.2f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", "-r", "24", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", final], cwd=d)
        if proc.returncode != 0 or not (d / final).exists():
            raise HTTPException(status_code=500, detail=f"mux failed: {(proc.stderr or '')[-2000:]}")
        narration_used = True
        out_duration = target
    else:
        proc = run(["ffmpeg", "-y", "-i", joined, "-c", "copy",
                    "-movflags", "+faststart", final], cwd=d)
        if proc.returncode != 0 or not (d / final).exists():
            raise HTTPException(status_code=500, detail=f"finalise failed: {(proc.stderr or '')[-2000:]}")

    data = (d / final).read_bytes()
    clip_count = len(clips)
    shutil.rmtree(d, ignore_errors=True)
    return Response(content=data, media_type="video/mp4", headers={
        "Content-Disposition": f'attachment; filename="{job}.mp4"',
        "X-Clip-Count": str(clip_count),
        "X-Narration": "1" if narration_used else "0",
        "X-Duration": f"{out_duration:.2f}",
    })


@app.delete("/job/{job}")
def delete_job(job: str, request: Request) -> dict:
    check_auth(request)
    shutil.rmtree(job_dir(job), ignore_errors=True)
    return {"job": job, "deleted": True}


# Systemd runs uvicorn directly; this is here so `python app.py` works for a
# quick manual run on the box.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                timeout_keep_alive=120)