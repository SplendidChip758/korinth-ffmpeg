"""
Korinth ffmpeg service  (v4)
----------------------------
HTTP wrapper around ffmpeg for the Korinth Industries video pipelines.

SHORTS path (unchanged since v2):
  PUT  /clip/{job}/{index}        raw mp4 bytes
  PUT  /narration/{job}           whole-video voiceover, base64
  POST /concat/{job}              join + mux one narration track
  POST /excerpt                    derive a 9:16 excerpt from an archived episode

LONG-FORM path (v4):
  PUT  /image64/{job}/{index}     base64 PNG + motion preset
  PUT  /clip64/{job}/{index}      base64 mp4
  POST /narrate/{job}/{index}     text -> Gemini TTS -> wav, returns MEASURED duration
  POST /assemble/{job}            per-segment timing, Ken Burns, concat, mux
  POST /assemble/{job}?background=1&deliver=path   -> 202, renders in a thread
  GET  /assemble/{job}/status     running | done | failed  (diagnostic only)

Why assemble can run in the background (v4.7.0)
  A full episode takes tens of minutes. As one long HTTP request, the caller's
  timeout - not the render - decides whether the episode exists: n8n gave up at
  30 minutes on a render that finished at 48 and reported a failure for a file
  that was already on the share. background=1 returns immediately and the
  caller polls, so no timeout anywhere is load-bearing.

Why the render calls back instead of being polled (v4.8.0)
  n8n no longer polls /assemble/{job}/status. When a background render reaches
  a terminal state, the service POSTs it to CALLBACK_URL (or callback_url in
  the submit body) and retries with backoff, from the state file rather than
  from the render thread, until it gets a 2xx. That file also survives a
  restart: at startup, any render still "running" under an instance that no
  longer exists is promoted to failed and its callback is fired, so an OOM
  mid-render is reported the moment the service comes back rather than on
  whatever poll would eventually have caught it. background=1 is refused at
  submit if no callback target is configured - a render nobody can hear from
  is worse than one that never started.

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
import contextvars
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import (http_exception_handler,
                                       request_validation_exception_handler)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

import korinth_tts_presets as tts_presets

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
# Base64 expands uploads by roughly one third. Reject oversized requests from
# Content-Length before FastAPI reads the complete body into memory.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", MAX_CLIP_BYTES * 4 // 3 + 1024 * 1024))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", 1800))

# Where a finished background render reports to. n8n no longer polls, so a
# render with no callback target finishes into silence - see the fail-closed
# check in assemble().
CALLBACK_URL = os.environ.get("CALLBACK_URL", "").strip()
CALLBACK_TOKEN = os.environ.get("CALLBACK_TOKEN", "").strip()
ALLOW_CALLBACK_OVERRIDE = os.environ.get("ALLOW_CALLBACK_OVERRIDE", "false").strip().lower() in ("1", "true", "yes", "on")
CALLBACK_ALLOWED_HOSTS = frozenset(host.strip().lower() for host in os.environ.get("CALLBACK_ALLOWED_HOSTS", "").split(",") if host.strip())
CALLBACK_TIMEOUT = float(os.environ.get("CALLBACK_TIMEOUT", "30"))
# The notifier wakes on this interval to retry whatever did not land. It is a
# stat() of one local directory, not a workflow execution - the thing that was
# actually costing something.
CALLBACK_TICK = float(os.environ.get("CALLBACK_TICK", "5"))
CALLBACK_BACKOFF_MAX = float(os.environ.get("CALLBACK_BACKOFF_MAX", "300"))
# After this long a callback stops being retried and starts being an alert.
CALLBACK_MAX_AGE_SECONDS = int(os.environ.get("CALLBACK_MAX_AGE_SECONDS", 24 * 60 * 60))
# Terminal state files are never swept today (sweep_old_jobs skips the whole
# underscore dir), so _assemble/ grows one file per episode forever.
STATE_RETENTION_SECONDS = int(os.environ.get("STATE_RETENTION_SECONDS", 7 * 24 * 60 * 60))
MAX_BACKGROUND_ASSEMBLIES = max(1, int(os.environ.get("MAX_BACKGROUND_ASSEMBLIES", "1")))

# Background /assemble bookkeeping. Underscore-prefixed so sweep_old_jobs()
# skips it; it has to outlive the job directory it describes, because
# deliver=path deletes that directory on success.
ASSEMBLE_STATE_DIR = DATA_DIR / "_assemble"
_assemble_lock = threading.Lock()
_assembly_slots = threading.BoundedSemaphore(MAX_BACKGROUND_ASSEMBLIES)
_active_assemblies: set = set()
_notify_inflight: set = set()
_notify_lock = threading.Lock()
_notify_stop = threading.Event()
# Identifies this process. A state file that still says "running" but carries a
# different instance was orphaned by a restart - see /assemble/{job}/status.
SERVICE_INSTANCE = f"{os.getpid()}-{int(time.time())}"

# Finished episodes are copied here instead of vanishing with the job folder.
# Unset = old behaviour (stream the bytes back, destroy everything).
_archive = os.environ.get("ARCHIVE_DIR", "").strip()
ARCHIVE_DIR = Path(_archive) if _archive else None

AMBIENT_VOLUME = os.environ.get("AMBIENT_VOLUME", "0.30")
# Gemini TTS loses gain across a long utterance. dynaudnorm rides the drift
# out inside a segment; NARR_LUFS pins every segment to the same absolute
# loudness so the film does not step up and down at each cut.
NARR_DYNAMICS = os.environ.get("NARR_DYNAMICS",
                               "dynaudnorm=f=150:g=51:p=0.95:m=30")
NARR_LUFS = os.environ.get("NARR_LUFS", "-16")
PAD_SECONDS = float(os.environ.get("PAD") or os.environ.get("PAD_SECONDS") or "0.4")
OUT_FPS = int(os.environ.get("OUT_FPS", "25"))

# ffmpeg parallelises its filtergraph across threads, and EACH in-flight thread
# holds its own copy of the frame. At the 8K work canvas that is ~100 MB per
# rgb24 frame in scale plus ~50 MB per yuv420p frame after it, so the default
# (one thread per core) is what turns a single segment into gigabytes.
#
# Measured on LXC 429, 2026-08-21: an OOM kill at 3.9 GB with ~13 ffmpeg worker
# threads live, on the FIRST segment of the render - the box could not fit one
# segment, never mind fourteen. Capping the filtergraph at a couple of threads
# cuts that roughly linearly.
#
# This does not change a single output pixel: it is how many frames are worked
# on at once, not how they are rendered. The 8K canvas and the 2x zoompan
# render are the 4.3.0 anti-stutter fix and are deliberately untouched.
#
# 0 = do not pass the flag, let ffmpeg decide (the pre-4.7 behaviour).
FILTER_THREADS = int(os.environ.get("FILTER_THREADS", "2"))
# Global options, so they belong before the inputs. Both spellings are set:
# -filter_threads covers -vf, -filter_complex_threads covers -filter_complex,
# and /assemble uses the latter.
FILTER_THREAD_ARGS = ([] if FILTER_THREADS <= 0 else
                      ["-filter_threads", str(FILTER_THREADS),
                       "-filter_complex_threads", str(FILTER_THREADS)])
OUT_W, OUT_H = 1920, 1080
# zoompan positions its crop window on whole SOURCE pixels, so the source has
# to be large enough that a slow move advances >1px per frame. At 3840 a 12%
# zoom over 15s advances 0.55px/frame - it stalls, then jumps, and that
# cadence is what reads as shake. 7680 puts it at ~1.1px/frame. (Imagen caps
# at 2K, so stills arrive ~2048x1152 and this is always an upscale.)
WORK_W, WORK_H = 7680, 4320
# zoompan renders at 2x output; the final downscale blends the residual
# whole-pixel step into a sub-pixel one.
ZOOM_W, ZOOM_H = 3840, 2160

# Service account, not the AI Studio API-key surface — GEAP's GCP project only
# issues service account keys, not GEMINI_API_KEY values. The project id lives
# in the key file itself, so GCP_PROJECT_ID is only needed to override it (e.g.
# billing a different project than the one that issued the key) — leave it
# unset in the normal case. GCP_PROJECT is accepted as an alias because that is
# the name the v4.5.0 rollout note used.
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
GCP_PROJECT_ID = (os.environ.get("GCP_PROJECT_ID")
                  or os.environ.get("GCP_PROJECT", "")).strip()
# Narration moved off Vertex AI's generateContent onto Cloud Text-to-Speech in
# v4.5.0, and text:synthesize is a global endpoint - so GCP_LOCATION no longer
# selects anything. Kept because /health reports it and the env file sets it.
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1").strip()
TTS_ENDPOINT = os.environ.get(
    "TTS_ENDPOINT", "https://texttospeech.googleapis.com/v1/text:synthesize")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "en-US")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-3.1-flash-tts-preview")
TTS_RATE = 24000
TTS_CHANNELS = 1

# v4.6.0: voice and style come from the named preset table on the share.
# korinth_tts_presets owns the whole decision - the file, the mtime reload, the
# env fallback and the hardcoded locked read - so nothing here reads TTS_VOICE
# or TTS_STYLE directly. Do not wrap `or` chains around resolve(); the
# precedence lives in one place on purpose.


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


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
# Everything goes to stdout, which systemd hands to journald:
#     journalctl -u korinth-ffmpeg -f
#     journalctl -u korinth-ffmpeg --since "10 min ago" | grep <request-id>
#
# INFO is one or two lines per pipeline step - enough to follow a render and see
# where it stopped. DEBUG adds every ffmpeg argv, every ffprobe result and the
# resolved style prompt, which is what you want for one reproduction run and not
# permanently: a fourteen-segment episode is a few hundred lines at DEBUG.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# A typo in LOG_LEVEL must not turn logging off, and it must not be silent
# either: an operator who set DEBGU and saw no debug lines would go looking in
# the wrong place. Fall back to INFO, and say so in the startup banner.
EFFECTIVE_LOG_LEVEL = LOG_LEVEL if LOG_LEVEL in _LEVELS else "INFO"
LOG_LEVEL_NOTE = "" if LOG_LEVEL in _LEVELS else \
    f" (LOG_LEVEL={LOG_LEVEL!r} is not one of {'/'.join(_LEVELS)})"

# n8n fires /narrate once per segment and /assemble logs a line per segment, so
# several renders interleave in the journal with nothing to tell them apart.
# Every record carries the id of the request that produced it, and the response
# carries it back in X-Request-Id, so an n8n execution points straight at its
# own log lines.
_request_id: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "korinth_request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """Attach the current request id to every record.

    Sync handlers run in FastAPI's threadpool rather than on the event loop, but
    anyio copies the caller's context into the worker thread, so an id set in
    the middleware is still visible from inside /narrate and /assemble.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = _request_id.get()
        return True


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("korinth")
    logger.setLevel(getattr(logging, EFFECTIVE_LOG_LEVEL))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s [%(rid)s] %(message)s",
            datefmt="%H:%M:%S"))
        handler.addFilter(_RequestIdFilter())
        logger.addHandler(handler)
    # No date in the format and no propagation: journald stamps its own date on
    # every line, and uvicorn owns the root logger's handlers - propagating
    # would print each line twice under systemd and, because uvicorn configures
    # logging only when it starts, not at all under `python app.py`.
    logger.propagate = False
    return logger


log = _setup_logging()


def _clip(text: str, limit: int = 160) -> str:
    """Shorten a value for a log line, marking that it was shortened."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit})"


def _log_startup() -> None:
    """One block, at boot, naming every setting a failed render can be traced to.

    A misconfigured service starts happily and fails on the first request that
    touches whatever is wrong - an unmounted share, an absent key file, a preset
    table that did not parse. Reading it back at startup means the journal
    already answers "what was this box actually running" without a /health poll
    that nobody made at the time.
    """
    log.info("korinth-ffmpeg %s+%s starting (log level %s%s)",
             VERSION, GIT_SHA, EFFECTIVE_LOG_LEVEL, LOG_LEVEL_NOTE)
    log.info("  ffmpeg=%s ffprobe=%s", ffmpeg_version(),
             "ok" if shutil.which("ffprobe") else "NOT FOUND")
    log.info("  data_dir=%s (writable=%s) max_job_age=%ss ffmpeg_timeout=%ss",
             DATA_DIR, os.access(DATA_DIR, os.W_OK) if DATA_DIR.exists() else "MISSING",
             MAX_JOB_AGE_SECONDS, FFMPEG_TIMEOUT)
    # ARCHIVE_DIR unwritable is the classic one: it only surfaces as EROFS at
    # the end of /assemble, after the whole render has already been paid for.
    log.info("  archive_dir=%s (writable=%s)", ARCHIVE_DIR,
             bool(ARCHIVE_DIR and os.access(ARCHIVE_DIR, os.W_OK)))
    log.info("  auth=%s", "on" if SERVICE_TOKEN else "OFF (no X_AUTH_TOKEN set)")
    log.info("  tts_configured=%s model=%s language=%s endpoint=%s",
             TTS_CONFIGURED, TTS_MODEL, TTS_LANGUAGE, TTS_ENDPOINT)
    if not TTS_CONFIGURED:
        log.warning("  GOOGLE_APPLICATION_CREDENTIALS is unset - /narrate will 500")

    presets = tts_presets.health()
    log.info("  tts_presets path=%s loaded=%s default=%s count=%d names=%s",
             presets["path"], presets["loaded"], presets["default"],
             presets["count"], presets["names"])
    if not presets["loaded"]:
        # Not fatal by design - resolve() falls through to TTS_VOICE/TTS_STYLE
        # and then to its own locked read - but it means edits to the table on
        # the share are having no effect, which is invisible from the audio.
        log.warning("  tts_presets NOT loaded (%s); narration will use "
                    "TTS_VOICE/TTS_STYLE or the hardcoded locked read",
                    presets["error"])
    log.info("  audio ambient=%s narr_lufs=%s dynamics=%s",
             AMBIENT_VOLUME, NARR_LUFS, NARR_DYNAMICS)
    log.info("  video out=%dx%d@%dfps work=%dx%d zoom=%dx%d pad=%ss",
             OUT_W, OUT_H, OUT_FPS, WORK_W, WORK_H, ZOOM_W, ZOOM_H, PAD_SECONDS)
    # The setting most likely to be the difference between a render and an OOM
    # on a small box, so it is read back with the rest of the video config.
    log.info("  filter_threads=%s%s", FILTER_THREADS or "unset (ffmpeg default)",
             "" if FILTER_THREADS else
             " - uncapped filtergraph threading at an 8K canvas is what OOM-killed this service on 2026-08-21")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _log_startup()
    promoted = _promote_orphans()
    log.info("  callback url=%s auth=%s tick=%.0fs timeout=%.0fs "
             "give_up_after=%.0fh (promoted %d orphaned render(s))",
             CALLBACK_URL or "NOT SET", bool(CALLBACK_TOKEN),
             CALLBACK_TICK, CALLBACK_TIMEOUT,
             CALLBACK_MAX_AGE_SECONDS / 3600, promoted)
    if not CALLBACK_URL:
        log.warning("  CALLBACK_URL is unset - background renders will be "
                    "refused at submit, because nothing polls this service")
    threading.Thread(target=_notifier_loop, name="callback-notifier",
                     daemon=True).start()
    yield
    _notify_stop.set()
    log.info("korinth-ffmpeg %s shutting down", VERSION)


app = FastAPI(title="Korinth ffmpeg service", version=VERSION, lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Tag every request with an id, then log how it ended and how long it took.

    Uvicorn's access log already records method, path and status. What it does
    not record is the elapsed time or an id, and both are the whole point here:
    /assemble runs for minutes, and "which of these four interleaved renders was
    the slow one" is unanswerable without them.
    """
    # An id supplied by the caller wins, so an n8n execution id can be carried
    # straight through into the journal.
    rid = (request.headers.get("x-request-id") or "").strip()[:32] or os.urandom(3).hex()
    token = _request_id.set(rid)
    started = time.monotonic()
    # /health is polled by install.sh and by monitoring; at INFO it would be the
    # only thing in the journal on an idle box.
    quiet = request.url.path == "/health"
    log.debug("-> %s %s", request.method, request.url.path)
    try:
        length = request.headers.get("content-length")
        if length:
            try:
                size = int(length)
            except ValueError:
                size = -1
            if size < 0:
                response = JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
            elif size > MAX_REQUEST_BYTES:
                response = JSONResponse({"detail": f"request exceeds maximum body size of {MAX_REQUEST_BYTES} bytes"}, status_code=413)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
    except Exception:
        # Uvicorn logs the traceback too, but without the id, so this is what
        # ties the crash to the request that caused it.
        log.exception("!! %s %s crashed after %.2fs",
                      request.method, request.url.path, time.monotonic() - started)
        _request_id.reset(token)
        raise
    elapsed = time.monotonic() - started
    log.log(logging.DEBUG if quiet else logging.INFO,
            "<- %s %s %s in %.2fs", request.method, request.url.path,
            response.status_code, elapsed)
    # Handed back so the caller can quote it. n8n stores response headers on the
    # execution, which makes this the link from a failed run to its log lines.
    response.headers["X-Request-Id"] = rid
    _request_id.reset(token)
    return response


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    """Log the 4xx/5xx bodies the service returns.

    Every deliberate refusal in here - 409 for a segment missing narration, 400
    for over-long text, 502 from the TTS call - travels back to n8n and nowhere
    else. Without this the journal shows a 409 with no reason, and the reason is
    the entire message.
    """
    log.log(logging.ERROR if exc.status_code >= 500 else logging.WARNING,
            "%s %s -> %s: %s", request.method, request.url.path,
            exc.status_code, _clip(exc.detail, 600))
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """422s, which are almost always an n8n expression that did not render.

    FastAPI returns a precise error and logs nothing, so a body that arrived as
    `{{ $json.text }}` instead of the text is invisible on this side.
    """
    log.warning("%s %s -> 422 validation: %s", request.method, request.url.path,
                _clip(exc.errors(), 600))
    return await request_validation_exception_handler(request, exc)


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

# Loaded lazily (not at import time) so a box with TTS unconfigured can still
# start and serve every other route; /narrate is what surfaces the error.
# /narrate is a sync endpoint, so FastAPI runs it in a threadpool and two
# requests really can land here at once - hence the lock.
# The _vertex_* names are historical: the credential is scoped to
# cloud-platform, so the same token authenticates Cloud Text-to-Speech now that
# narration no longer goes to Vertex AI.
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
                    log.error("service account key %s unusable: %s",
                              GOOGLE_APPLICATION_CREDENTIALS, e)
                    raise HTTPException(status_code=500,
                                        detail=f"failed to load service account credentials: {e}")
                _vertex_project_id = GCP_PROJECT_ID or creds.project_id
                _vertex_credentials = creds
                # client_email is not a secret and is the first thing you need
                # when a 403 arrives: the message names a permission but never
                # which identity was missing it. Logged once, on first load.
                log.info("service account loaded: %s (key project=%s, "
                         "billing project=%s%s)",
                         getattr(creds, "service_account_email", "unknown"),
                         creds.project_id, _vertex_project_id,
                         ", overridden by GCP_PROJECT_ID" if GCP_PROJECT_ID else "")
    return _vertex_credentials


def _vertex_access_token() -> str:
    creds = _load_vertex_credentials()
    # Access tokens are short-lived; refresh() is a no-op if the cached one
    # still has enough lifetime left, so it's cheap to call on every request.
    #
    # The validity check, the refresh and the token read all happen inside one
    # lock. Reading creds.token outside it let a thread sample the credential
    # while another was mid-refresh and come away with the expired value - the
    # exact race the lock was added for. Holding it across the refresh also
    # stops every waiting thread from firing its own refresh call.
    #
    # _load_vertex_credentials() takes this same lock, so it has to stay
    # outside the block: threading.Lock is not reentrant.
    with _vertex_lock:
        if not creds.valid:
            started = time.monotonic()
            try:
                creds.refresh(GoogleAuthRequest())
            except Exception as e:
                log.error("token refresh failed: %s", e)
                raise HTTPException(status_code=502,
                                    detail=f"failed to refresh service account token: {e}")
            # A refresh on nearly every request means the token is being thrown
            # away rather than cached, which is worth seeing in a sequence of
            # fourteen /narrate calls.
            log.debug("refreshed access token in %.2fs (expires %s)",
                      time.monotonic() - started, getattr(creds, "expiry", "?"))
        return creds.token


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def check_auth(request: Request) -> None:
    if not SERVICE_TOKEN:
        return
    if request.headers.get("x-auth-token", "") != SERVICE_TOKEN:
        # Whether the header was absent or merely wrong is the whole diagnosis:
        # absent means the n8n node lost its credential, wrong means the token
        # was regenerated by a re-install and the credential was not updated.
        # The value itself is never logged.
        log.warning("auth rejected on %s: X-Auth-Token %s", request.url.path,
                    "mismatched" if request.headers.get("x-auth-token")
                    else "not sent")
        raise HTTPException(status_code=401, detail="bad or missing X-Auth-Token")


def job_dir(job: str) -> Path:
    if not SAFE_JOB.match(job):
        raise HTTPException(status_code=400, detail="invalid job id")
    return DATA_DIR / job


def check_index(index: int) -> int:
    """Reject indices that would produce a file no other route can find.

    Filenames are zero-padded to three digits, and segment_indices() only
    matches bare numeric stems - so index 1000 writes `narr_1000.wav` and
    index -1 writes `narr_-01.wav`, and /assemble then reports the segment as
    unnarrated. Shared by every route that takes an index so the ingest and
    narration paths cannot drift apart on what they accept, which is how the
    asymmetry arose in the first place.
    """
    if index < 0 or index > 999:
        raise HTTPException(status_code=400, detail="index out of range")
    return index


def sweep_old_jobs() -> None:
    now = time.time()
    if not DATA_DIR.exists():
        return
    for entry in DATA_DIR.iterdir():
        try:
            # Underscore-prefixed directories are service state, not jobs.
            # _assemble/ holds background render status and outlives every job.
            if entry.name.startswith("_"):
                continue
            if entry.is_dir() and now - entry.stat().st_mtime > MAX_JOB_AGE_SECONDS:
                # Deleting someone's segments silently is how a slow pipeline
                # looks like a pipeline that dropped work. Say what went and how
                # old it was, so a MAX_JOB_AGE_SECONDS that is too low for the
                # run time is visible rather than inferred.
                age = now - entry.stat().st_mtime
                log.info("sweeping job %s (idle %.0fm, limit %.0fm)",
                         entry.name, age / 60, MAX_JOB_AGE_SECONDS / 60)
                shutil.rmtree(entry, ignore_errors=True)
        except OSError as e:
            log.debug("sweep skipped %s: %s", entry, e)


def archive_final(job: str, src: Path) -> Optional[str]:
    """Copy the finished mp4 onto the share.

    Written to a dotfile then renamed. os.replace is atomic within a
    directory, so a reader watching the share never sees a partial file -
    which matters because n8n starts reading the moment this call returns.
    """
    if ARCHIVE_DIR is None:
        log.debug("no ARCHIVE_DIR set; %s stays in the job dir", src.name)
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{job}.mp4"
    tmp = ARCHIVE_DIR / f".{job}.mp4.part"
    started = time.monotonic()
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    except OSError as exc:
        # The copy is the last thing /assemble does, so a share that went away
        # costs the entire render. Log it separately from the 500 body: the
        # errno here is the difference between not-mounted and full.
        log.error("archive copy to %s failed after %.1fs: %s",
                  dest, time.monotonic() - started, exc)
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"could not write to ARCHIVE_DIR {ARCHIVE_DIR}: {exc}. "
                   f"Check the share is mounted and writable by this service.")
    log.info("archived %s (%.1f MB) in %.1fs", dest,
             dest.stat().st_size / 1e6, time.monotonic() - started)
    return str(dest)


def run(cmd, cwd, label: str = "") -> subprocess.CompletedProcess:
    """Run ffmpeg/ffprobe and log what it did.

    Every ffmpeg failure in the service is logged here, once, with the FULL
    stderr - the HTTPException bodies raised by callers truncate to the last
    1000-2000 characters, and the line that explains the failure is regularly
    above the cut. The argv is logged too: pasting it into a shell on the box is
    the fastest way to reproduce a filter-graph error.
    """
    what = label or Path(str(cmd[0])).name
    log.debug("%s: %s", what, " ".join(str(c) for c in cmd))
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.error("%s timed out after %ss: %s", what, FFMPEG_TIMEOUT,
                  " ".join(str(c) for c in cmd))
        raise HTTPException(status_code=504, detail="ffmpeg timed out")
    except FileNotFoundError:
        log.error("%s not installed (cmd=%s)", cmd[0], what)
        raise HTTPException(status_code=500, detail="ffmpeg/ffprobe not installed")
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        log.warning("%s exited %d after %.1fs\n  cmd: %s\n  stderr: %s",
                    what, proc.returncode, elapsed,
                    " ".join(str(c) for c in cmd), (proc.stderr or "").strip())
    else:
        log.debug("%s ok in %.2fs", what, elapsed)
    return proc


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
                "-of", "csv=p=0", path.name], cwd=path.parent,
               label=f"ffprobe duration {path.name}")
    try:
        dur = float((proc.stdout or "").strip())
    except ValueError:
        # Returning 0.0 keeps /assemble running, but a zero-duration narration
        # silently collapses that segment to the 1.0s floor - the picture flashes
        # past and the voiceover is cut off. Never let that be inferred from the
        # output alone.
        log.warning("could not read a duration from %s (ffprobe stdout=%r "
                    "stderr=%s); treating it as 0.0s",
                    path.name, _clip(proc.stdout or "", 80),
                    _clip(proc.stderr or "", 200))
        return 0.0
    log.debug("%s duration %.3fs", path.name, dur)
    return dur


def normalise_narration(d: Path, src: str, dst: str) -> None:
    """Flatten TTS gain drift, then pin the file to a fixed loudness.

    Two passes on purpose. One-pass loudnorm runs in *dynamic* mode and
    reimposes a slow ramp of its own, so the decay survives at reduced depth.
    Measuring first and feeding the numbers back puts loudnorm in *linear*
    mode: one constant gain, dynaudnorm's flat envelope left intact.

    Measured on a synthetic 20 dB ramp, mean level at 0s / 10s / 20s:
        raw        -22.2  -25.7  -31.9   (the symptom)
        one-pass   -13.1  -14.6  -17.6   (still sliding)
        two-pass   -16.8  -14.1  -17.3   (no trend, just speech variation)
    """
    base = f"{NARR_DYNAMICS},loudnorm=I={NARR_LUFS}:TP=-1.5:LRA=11"

    probe = run(["ffmpeg", "-hide_banner", "-i", src,
                 "-af", base + ":print_format=json", "-f", "null", "-"], cwd=d,
                label=f"loudnorm measure {src}")
    stats = {}
    blob = re.search(r"\{[^{}]*input_i[^{}]*\}", probe.stderr or "", re.S)
    if blob:
        try:
            stats = json.loads(blob.group(0))
        except ValueError:
            log.warning("loudnorm printed a JSON block for %s that did not "
                        "parse: %s", src, _clip(blob.group(0), 300))
            stats = {}

    af = base
    keys = ("input_i", "input_tp", "input_lra", "input_thresh")
    if all(k in stats for k in keys):
        af += (f":measured_I={stats['input_i']}"
               f":measured_TP={stats['input_tp']}"
               f":measured_LRA={stats['input_lra']}"
               f":measured_thresh={stats['input_thresh']}:linear=true")
        # input_i is the loudness the TTS actually came back at. Logged on every
        # segment because a drift in that number across an episode is the
        # original v4.4.0 symptom, and the only place it is ever visible.
        log.info("normalise %s: two-pass linear, measured I=%s LRA=%s TP=%s "
                 "-> target %s LUFS", src, stats["input_i"], stats["input_lra"],
                 stats["input_tp"], NARR_LUFS)
    else:
        # Falls through to one-pass, which works but reimposes a gain ramp of
        # its own - the exact fade this function exists to remove. Silent, the
        # symptom comes back looking like a TTS regression.
        log.warning("normalise %s: loudnorm measurement unavailable (parsed "
                    "keys %s), falling back to ONE-PASS dynamic mode - gain "
                    "drift will be reduced but not removed", src, sorted(stats))

    proc = run(["ffmpeg", "-y", "-i", src, "-af", af,
                "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", dst], cwd=d,
               label=f"normalise {src} -> {dst}")
    if proc.returncode != 0 or not (d / dst).exists():
        raise HTTPException(
            status_code=500,
            detail=f"narration normalise failed: {(proc.stderr or '')[-1000:]}")


def has_audio(path: Path) -> bool:
    proc = run(["ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", path.name],
               cwd=path.parent, label=f"ffprobe audio {path.name}")
    found = bool((proc.stdout or "").strip())
    log.debug("%s has audio: %s", path.name, found)
    return found


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
    # A separate field on purpose. Inferring the preset by sniffing whether
    # `style` looks like a name would misresolve a one-word style prompt, which
    # is a legitimate thing to audition, and that bug is invisible until
    # playback.
    # An unknown name is not an error: korinth_tts_presets.resolve substitutes
    # default_preset and flags it, because failing a fourteen-segment render at
    # segment nine over a typo in a slug is worse than a consistent wrong
    # narrator.
    preset: Optional[str] = None
    voice: Optional[str] = None
    style: Optional[str] = None


class AssembleBody(BaseModel):
    """Optional. Present only to override the env-configured callback target.

    Optional with a None default so every existing caller that POSTs no body
    at all keeps working.
    """
    callback_url: Optional[str] = None
    callback_token: Optional[str] = None


class ExcerptBody(BaseModel):
    path: str
    start_seconds: float = 0.0
    duration_seconds: float = 45.0


async def json_body(request: Request) -> dict:
    raw = await request.body()
    try:
        return json.loads(raw or b"{}")
    except json.JSONDecodeError as e:
        # The first bytes are the diagnosis: `------WebKit` means the node sent
        # multipart, `{{ $json` means an expression did not render, and `<html`
        # means something in front of the service answered instead of it.
        log.warning("non-JSON body on %s (%d bytes, starts %r): %s",
                    request.url.path, len(raw), raw[:60], e)
        raise HTTPException(status_code=400, detail="body must be JSON")


def motion_filter(motion: str, frames: int) -> str:
    """
    Ken Burns preset. Applied AFTER an upscale to WORK_W x WORK_H, so the zoom
    always crops into an oversized source and never has to invent detail, and
    renders at ZOOM_W x ZOOM_H so the caller's downscale to output can turn the
    residual whole-pixel step into a sub-pixel one.
    `on` is the output frame number; d must equal the total frame count.
    """
    f = max(frames, 2)
    centre = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    tail = f":d={f}:s={ZOOM_W}x{ZOOM_H}:fps={OUT_FPS}"

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


VIDEO_ARGS = ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
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
        except HTTPException as e:
            # Surfaced properly by /narrate, but /health is what install.sh
            # polls, so the reason belongs in the journal at deploy time.
            log.warning("/health could not load credentials: %s", e.detail)
    return {
        "status": "ok" if shutil.which("ffmpeg") and shutil.which("ffprobe") else "degraded",
        "version": VERSION,
        "git_sha": GIT_SHA,
        "ffmpeg": ff,
        "ffprobe": "ok" if shutil.which("ffprobe") else "NOT FOUND",
        "auth": bool(SERVICE_TOKEN),
        "tts_configured": TTS_CONFIGURED,
        # So a broken credential is visible without a test render.
        "tts_auth": "service-account" if TTS_CONFIGURED else "MISSING",
        "tts_project": gcp_project or None,
        "tts_model": TTS_MODEL,
        # health() never raises, and its mtime is how you tell which table is
        # actually deployed - the same problem the version string solves for
        # app.py. A table that failed to load reports `error` with `loaded:
        # false` here rather than on the first render.
        "tts_presets": tts_presets.health(),
        "gcp_project": gcp_project,
        "gcp_location": GCP_LOCATION,
        "archive_dir": str(ARCHIVE_DIR) if ARCHIVE_DIR else None,
        "archive_writable": bool(ARCHIVE_DIR and os.access(ARCHIVE_DIR, os.W_OK)),
        "callback_configured": bool(CALLBACK_URL),
        "callback_authenticated": bool(CALLBACK_TOKEN),
        "callback_override_allowed": ALLOW_CALLBACK_OVERRIDE,
        "background_assemblies": {
            "active": len(_active_assemblies),
            "maximum": MAX_BACKGROUND_ASSEMBLIES,
        },
        "max_request_bytes": MAX_REQUEST_BYTES,
        "excerpt_supported": True,
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
    check_index(index)
    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    # Zero-padded so lexical order matches segment order.
    (d / f"{index:03d}.{ext}").write_bytes(data)
    # One line per stored asset is how you see the pipeline advancing, and how
    # you tell "n8n stopped sending" from "the service stopped accepting".
    log.info("stored %s/%03d.%s (%.2f MB)", job, index, ext, len(data) / 1e6)
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
        # Silently corrected so a typo cannot fail an ingest, but the correction
        # only becomes visible three steps later in /assemble's motion line.
        log.warning("unknown motion %r for %s/%d, using kenburns (valid: %s)",
                    motion, job, index, ", ".join(MOTIONS))
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
                "-update", "1", "-q:v", "2", out.name], cwd=d,
               label=f"lastframe {job}/{index}")
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
    Body (JSON):
      {"text":   "...",              required
       "preset": "narrator",         optional - name from tts-presets.json
       "voice":  "Umbriel",          optional - overrides the preset's voice
       "style":  "Read this as ..."} optional - overrides the preset's style

    Generates the segment's voiceover via Gemini TTS, writes it as a wav in the
    job directory, and returns its MEASURED duration.

    voice and style override the preset per field, which is what lets
    korinth-audition.sh drive this endpoint directly instead of calling Google
    itself, so an audition follows whatever auth the service is using.

    Deliberately a sync def: this makes a blocking HTTP call and then shells out
    to ffmpeg. As `async def` both of those stall the event loop for the whole
    request, so nothing else the service is doing can proceed. Sync handlers run
    in FastAPI's threadpool instead, which is also what /assemble does.
    """
    check_auth(request)
    check_index(index)

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    n_bytes = len(text.encode("utf-8"))
    if n_bytes > 4000:
        # The API's own cap. Failing here names the real problem; the API
        # returns a 400 that does not mention which field was too long.
        raise HTTPException(
            status_code=400,
            detail=f"segment text is {n_bytes} bytes; the synthesize API caps "
                   f"input.text at 4000. Raise the segment count in Split "
                   f"Narration so chunks are smaller.")

    # One call, one place. resolve() already implements the whole precedence
    # (explicit fields, then the named preset, then TTS_VOICE / TTS_STYLE, then
    # the hardcoded locked read), so there is deliberately no `or` chain here.
    sel = tts_presets.resolve(preset=body.preset, voice=body.voice,
                              style=body.style)
    voice, style = sel["voice"], sel["style"]

    log.info("narrate %s/%03d: %d chars / %d bytes, voice=%s preset=%s%s",
             job, index, len(text), n_bytes, voice, sel["preset"],
             "".join(f" {f}=explicit" for f, v in
                     (("voice", body.voice), ("style", body.style)) if v))
    if sel["preset_substituted"]:
        # Deliberate - a typo in a slug must not fail a fourteen-segment render
        # - but the whole episode is now narrated by something other than what
        # was asked for, so it is a warning rather than a note.
        log.warning("narrate %s/%03d: preset %r is not in the table; "
                    "substituted %r", job, index, sel["preset_requested"],
                    sel["preset"])
    # The style prompt is the other half of what the voice sounds like, so an
    # audition that came out wrong needs to show exactly what was sent.
    log.debug("narrate %s/%03d style: %s", job, index, _clip(style, 400))
    log.debug("narrate %s/%03d text: %s", job, index, _clip(text, 400))

    token = _vertex_access_token()

    payload = {
        # prompt and text are separate fields, so the style instruction cannot
        # be read aloud the way it could when it was concatenated on the front.
        "input": {"prompt": style, "text": text},
        "voice": {"languageCode": TTS_LANGUAGE, "name": voice,
                  "modelName": TTS_MODEL},
        "audioConfig": {"audioEncoding": "LINEAR16",
                        "sampleRateHertz": TTS_RATE},
    }
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    # Pins quota and billing to our project rather than to whichever project
    # the credential happens to belong to. Needs
    # roles/serviceusage.serviceUsageConsumer on that project, or the call 403s
    # with a message about the caller that reads like a TTS permission problem.
    if _vertex_project_id:
        headers["x-goog-user-project"] = _vertex_project_id

    started = time.monotonic()
    try:
        with httpx.Client(timeout=180.0) as client:
            r = client.post(TTS_ENDPOINT, json=payload, headers=headers)
    except httpx.HTTPError as e:
        # A connect timeout and a read timeout mean different things here (no
        # egress vs. a model that is slow on long text), and the exception type
        # is the only thing that distinguishes them.
        log.error("TTS transport error after %.1fs: %s: %s",
                  time.monotonic() - started, type(e).__name__, e)
        raise HTTPException(status_code=502, detail=f"TTS request failed: {e}")
    api_elapsed = time.monotonic() - started

    if r.status_code != 200:
        # The full body, not the 400 characters that fit in the HTTP detail. A
        # 403 here carries a long message naming a permission and a project, and
        # the useful half is usually past the cut.
        log.error("TTS %s in %.1fs for %s/%03d (voice=%s, project=%s): %s",
                  r.status_code, api_elapsed, job, index, voice,
                  _vertex_project_id, (r.text or "").strip())
        raise HTTPException(status_code=502,
                            detail=f"TTS returned {r.status_code}: {r.text[:400]}")

    try:
        b64 = r.json()["audioContent"]
    except (KeyError, ValueError):
        log.error("TTS 200 but no audioContent for %s/%03d: %s",
                  job, index, (r.text or "")[:2000])
        raise HTTPException(status_code=502,
                            detail=f"unexpected TTS response: {r.text[:400]}")

    pcm = base64.b64decode(b64)
    if not pcm:
        log.error("TTS returned an empty audioContent for %s/%03d", job, index)
        raise HTTPException(status_code=502, detail="TTS returned empty audio")
    log.info("narrate %s/%03d: TTS 200 in %.1fs, %.0f KB %s",
             job, index, api_elapsed, len(pcm) / 1024,
             "RIFF" if pcm[:4] == b"RIFF" else "bare PCM")

    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    wav = d / f"narr_{index:03d}.wav"
    tmp = d / f"narr_{index:03d}.tmp.wav"

    # LINEAR16 comes back as a RIFF/WAV file. Handle bare PCM too, so a change
    # of audioEncoding default cannot silently produce noise.
    if pcm[:4] == b"RIFF":
        tmp.write_bytes(pcm)
    else:
        raw = d / f"narr_{index:03d}.raw"
        raw.write_bytes(pcm)
        proc = run(["ffmpeg", "-y", "-f", "s16le", "-ar", str(TTS_RATE),
                    "-ac", str(TTS_CHANNELS), "-i", raw.name,
                    "-c:a", "pcm_s16le", tmp.name], cwd=d,
                   label=f"wav wrap {job}/{index}")
        raw.unlink(missing_ok=True)
        if proc.returncode != 0 or not tmp.exists():
            raise HTTPException(status_code=500,
                                detail=f"wav wrap failed: {(proc.stderr or '')[-1000:]}")

    normalise_narration(d, tmp.name, wav.name)
    tmp.unlink(missing_ok=True)

    dur = probe_duration(wav)
    if dur <= 0:
        # /assemble will floor this segment at 1.0s and the voiceover will be
        # cut off mid-sentence. The wav is on disk, so nothing downstream errors.
        log.error("narrate %s/%03d produced a %.3fs wav - the segment will be "
                  "clipped in /assemble", job, index, dur)
    # Total minus api_elapsed is the local cost: wav wrap plus the two
    # normalisation passes plus the probe. Worth separating, because "narration
    # got slow" is usually Google and occasionally this box.
    log.info("narrate %s/%03d done: %.3fs of audio, %.0f KB wav "
             "(%.1fs total, %.1fs local)", job, index, dur,
             wav.stat().st_size / 1024, time.monotonic() - started,
             time.monotonic() - started - api_elapsed)
    # The resolved voice goes back on every response so the caller can stamp it
    # on the plan and the story log. When episode 30 sounds different from
    # episode 12, this is what says whether the preset changed or the model
    # drifted - otherwise unrecoverable, and it costs three fields.
    #
    # preset_substituted is the one to alert on: it means a name was asked for
    # and something else was used. The substitution is deliberate, but it must
    # not be silent.
    #
    # Not `**sel` - the style prose would be echoed on every segment, and
    # sel["model"] is documentation, not what was sent. TTS_MODEL is what
    # actually populated modelName above.
    #
    # bytes is the normalised wav on disk, which is what the next stage reads.
    # It used to report len(pcm) - the raw API payload before wav wrapping and
    # two-pass normalisation - so the two numbers are kept apart by name.
    return {"job": job, "index": index,
            "bytes": wav.stat().st_size, "tts_bytes": len(pcm),
            "duration_seconds": round(dur, 3),
            "voice": sel["voice"], "preset": sel["preset"],
            "preset_requested": sel["preset_requested"],
            "preset_substituted": sel["preset_substituted"]}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _assemble_core(job: str, allow_silent: int, allow_gaps: int, expect: int):
    """Render every segment, concat, archive.

    Returns (job_dir, final_name, payload, stats_headers). Delivery and cleanup
    are the caller's problem, so the same code serves both the synchronous
    endpoint and the background worker.
    """
    assemble_started = time.monotonic()

    d = job_dir(job)
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"no such job {job}")

    indices = segment_indices(d)
    if not indices:
        raise HTTPException(status_code=404, detail=f"no segments stored for job {job}")

    log.info("assemble %s: %d segments %s (allow_silent=%d allow_gaps=%d "
             "expect=%d)", job, len(indices), indices,
             allow_silent, allow_gaps, expect)

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

    if gaps and allow_gaps:
        log.warning("assemble %s: rendering with gaps at %s (allow_gaps=1)",
                    job, gaps)

    missing = [i for i in indices if not (d / f"narr_{i:03d}.wav").exists()]
    if missing and allow_silent:
        # Explicitly permitted, but it produces a video that ships broken
        # without erroring, so it should never be the state nobody noticed.
        log.warning("assemble %s: segments %s have no narration and will be "
                    "SILENT (allow_silent=1)", job, missing)
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
        seg_started = time.monotonic()

        if wav.exists():
            dur = probe_duration(wav) + PAD_SECONDS
            narrations += 1
        else:
            # The 5s default is arbitrary and only reachable with allow_silent.
            dur = 5.0
        dur = max(dur, 1.0)
        frames = int(round(dur * OUT_FPS))

        # Every audio path below is padded to the segment's full length. The
        # final concat is a stream copy, so an audio track even a fraction
        # shorter than its video makes the voiceover creep ahead of the
        # picture, once per segment, cumulatively.
        # A video segment keeps its PNG keyframe as a generation/reference
        # artifact, so both files can legitimately exist. Prefer the finished
        # MP4 and use the PNG only when no clip was stored.
        if png.exists() and not mp4.exists():
            images += 1
            motion_file = d / f"{i:03d}.motion"
            motion = motion_file.read_text(encoding="utf-8").strip() if motion_file.exists() else "kenburns"
            # Preserve the source aspect ratio. Portrait and square inputs are
            # centered on a neutral canvas instead of being stretched to 16:9.
            vf = (f"[0:v]scale={WORK_W}:{WORK_H}:"
                  f"force_original_aspect_ratio=decrease:flags=lanczos,"
                  f"pad={WORK_W}:{WORK_H}:(ow-iw)/2:(oh-ih)/2:"
                  f"color=0x090d12,setsar=1,"
                  f"{motion_filter(motion, frames)},"
                  f"scale={OUT_W}:{OUT_H}:flags=lanczos[v]")
            # The motion is read back from disk, so a `?motion=` that n8n sent
            # wrong was silently replaced with kenburns at ingest and this is
            # where that becomes visible.
            log.info("assemble %s seg %03d: still, %s, %.2fs (%d frames)%s",
                     job, i, motion, dur, frames,
                     "" if wav.exists() else ", SILENT")

            cmd = (["ffmpeg", "-y"] + FILTER_THREAD_ARGS +
                   ["-loop", "1", "-framerate", str(OUT_FPS),
                    "-t", f"{dur:.3f}", "-i", png.name])
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
            cmd = ["ffmpeg", "-y"] + FILTER_THREAD_ARGS + ["-i", mp4.name]

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
            # Which of the four audio shapes was chosen decides whether the
            # ambience is ducked, and it is not recoverable from the output.
            log.info("assemble %s seg %03d: clip, %.2fs, narration=%s "
                     "clip_audio=%s%s", job, i, dur, wav.exists(),
                     clip_has_audio,
                     f", ambience ducked to {AMBIENT_VOLUME}" if clip_has_audio else "")
        else:
            # segment_indices() matched a stem that is neither, which means a
            # file was deleted between the listing and here.
            log.warning("assemble %s seg %03d: neither %s nor %s on disk, "
                        "skipping", job, i, png.name, mp4.name)
            continue

        proc = run(cmd, cwd=d, label=f"render {job} seg {i:03d}")
        if proc.returncode != 0 or not (d / seg).exists():
            raise HTTPException(status_code=500,
                                detail=f"segment {i} render failed: {(proc.stderr or '')[-1500:]}")
        log.info("assemble %s seg %03d rendered in %.1fs (%.1f MB)", job, i,
                 time.monotonic() - seg_started, (d / seg).stat().st_size / 1e6)
        parts.append(seg)

    if not parts:
        raise HTTPException(status_code=500, detail="no segments rendered")

    (d / "concat.txt").write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
    final = "final.mp4"
    # Every segment was written with identical stream parameters above, so a
    # stream-copy concat is safe here and avoids a second full re-encode.
    log.info("assemble %s: concatenating %d segments", job, len(parts))
    proc = run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                "-c", "copy", "-movflags", "+faststart", final], cwd=d,
               label=f"concat {job}")
    if proc.returncode != 0 or not (d / final).exists():
        raise HTTPException(status_code=500,
                            detail=f"concat failed: {(proc.stderr or '')[-1500:]}")

    total = probe_duration(d / final)
    log.info("assemble %s complete: %d segments (%d stills, %d clips, "
             "%d narrated), %.1fs of video, %.1f MB, rendered in %.0fs",
             job, len(parts), images, videos, narrations, total,
             (d / final).stat().st_size / 1e6,
             time.monotonic() - assemble_started)
    archived = archive_final(job, d / final)

    stats = {
        "X-Segment-Count": str(len(parts)),
        "X-Image-Count": str(images),
        "X-Clip-Count": str(videos),
        "X-Narration-Count": str(narrations),
        "X-Duration": f"{total:.2f}",
    }
    if archived:
        stats["X-Archive-Path"] = archived

    payload = {
        "job": job,
        "path": archived,
        "file": f"{job}.mp4",
        "segments": len(parts),
        "images": images,
        "clips": videos,
        "narrations": narrations,
        "duration_seconds": round(total, 2),
    }
    return d, final, payload, stats


# --------------------------------------------------------------------------
# background assembly
# --------------------------------------------------------------------------
# A 14-segment render takes tens of minutes of wall clock. Held open as one
# HTTP request, the CALLER'S timeout decides whether the episode exists: n8n
# has given up at its 30 minute limit on a render that completed fine later,
# reported a failure, and skipped the upload for a file that was already
# sitting on the share. Nothing downstream could tell that apart from a real
# render bug.
#
# Submit-and-poll removes the whole class: no request is open longer than it
# takes to spawn a thread, and the caller asks about state instead of holding
# a socket. Same shape as the Veo path in korinth-produce, which already
# submits an operation and polls it.
#
# State is a file, not a dict, because a dict cannot tell you that the service
# restarted mid-render - it just comes back empty and the render looks like it
# never happened. The instance stamp turns that into an explicit failure.


def _assemble_state_path(job: str) -> Path:
    if not SAFE_JOB.match(job):
        raise HTTPException(status_code=400, detail="invalid job id")
    return ASSEMBLE_STATE_DIR / f"{job}.json"


def _write_assemble_state(job: str, state: dict) -> None:
    """Write atomically - a poller must never read a half-written state file."""
    ASSEMBLE_STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(ASSEMBLE_STATE_DIR, 0o700)
    dest = _assemble_state_path(job)
    tmp = dest.with_name(dest.name + ".part")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(state))
    os.chmod(tmp, 0o600)
    os.replace(tmp, dest)


def _read_assemble_state(job: str) -> Optional[dict]:
    dest = _assemble_state_path(job)
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# callback delivery
# --------------------------------------------------------------------------
# n8n does not poll. A background render that finishes and cannot tell anyone
# is an episode stuck in `assembling` with nothing watching it, so delivery is
# retried from the state file rather than attempted once from the render
# thread. The file is the outbox: it survives the thread that wrote it, the
# restart that killed the thread, and the box being down while n8n is up.


def _callback_payload(job: str, state: dict) -> dict:
    """Exactly what GET /assemble/{job}/status returns, plus an attempt count.

    Deliberately the same shape: the eight tail nodes were written against the
    poll response, and moving them to the webhook should not mean rewriting
    their expressions.
    """
    cb = state.get("callback") or {}
    payload = {
        "job": job,
        "state": state.get("state"),
        "instance": state.get("instance"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        # n8n will see this >1 whenever a response was lost on the way back.
        # It is the only signal that a duplicate is a duplicate.
        "attempt": cb.get("attempts", 0) + 1,
    }
    if state.get("state") == "done":
        payload["result"] = state.get("result")
    else:
        payload["status_code"] = state.get("status_code")
        payload["error"] = state.get("error")
    return payload


def _deliver_callback(job: str) -> None:
    """One delivery attempt. Records the outcome on the state file either way.

    Never raises: called from a daemon thread and from the notifier loop, and
    a callback failure must not change the render's own recorded outcome.
    """
    with _notify_lock:
        if job in _notify_inflight:
            return
        _notify_inflight.add(job)
    try:
        state = _read_assemble_state(job)
        if not state:
            return
        cb = state.get("callback") or {}
        url = cb.get("url")
        if (not url or cb.get("delivered") or cb.get("abandoned")
                or state.get("state") not in ("done", "failed")):
            return

        payload = _callback_payload(job, state)
        attempt = payload["attempt"]
        headers = {"Content-Type": "application/json"}
        token = cb.get("token") or CALLBACK_TOKEN
        if token:
            headers["X-Korinth-Token"] = token
        if cb.get("request_id"):
            # Carries the submit's request id into n8n, and back into this
            # journal on the retry lines. One id spans the whole render.
            headers["X-Request-Id"] = cb["request_id"]

        started = time.monotonic()
        status = None
        err = None
        try:
            with httpx.Client(timeout=CALLBACK_TIMEOUT) as client:
                r = client.post(url, json=payload, headers=headers)
            status = r.status_code
            ok = 200 <= status < 300
            if not ok:
                err = _clip(r.text or "", 300)
        except httpx.HTTPError as e:
            ok = False
            err = f"{type(e).__name__}: {e}"

        # Re-read rather than reusing the copy from before the POST: the record
        # may have been rewritten while the request was open.
        state = _read_assemble_state(job) or state
        cb = state.get("callback") or cb
        cb["attempts"] = attempt
        cb["last_status"] = status
        cb["last_error"] = err

        if ok:
            cb["delivered"] = True
            cb["delivered_at"] = time.time()
            cb.pop("next_attempt_at", None)
            log.info("callback %s -> %s delivered on attempt %d in %.1fs",
                     job, url, attempt, time.monotonic() - started)
        else:
            # Every non-2xx is retried, including 404. A 404 here usually means
            # the workflow is not published yet, which is a state that gets
            # fixed - refusing to retry it would strand the episode.
            delay = min(CALLBACK_BACKOFF_MAX, 2.0 ** min(attempt, 8))
            cb["next_attempt_at"] = time.time() + delay
            log.warning("callback %s -> %s attempt %d failed after %.1fs "
                        "(%s: %s); retrying in %.0fs", job, url, attempt,
                        time.monotonic() - started, status or "no response",
                        _clip(err or "", 200), delay)

        state["callback"] = cb
        _write_assemble_state(job, state)
    except Exception:  # noqa: BLE001 - the notifier must survive anything
        log.exception("callback %s: delivery attempt raised", job)
    finally:
        with _notify_lock:
            _notify_inflight.discard(job)


def _queue_callback(job: str) -> None:
    threading.Thread(target=_deliver_callback, args=(job,),
                     name=f"callback-{job}", daemon=True).start()


def _notify_scan() -> None:
    """Retry undelivered callbacks; retire state files nobody needs.

    Also the startup rescan - it is the same scan, so there is one code path
    for "n8n was down for an hour" and "this service was down for an hour".
    """
    now = time.time()
    if not ASSEMBLE_STATE_DIR.exists():
        return
    for path in sorted(ASSEMBLE_STATE_DIR.glob("*.json")):
        job = path.stem
        state = _read_assemble_state(job)
        if not state or state.get("state") not in ("done", "failed"):
            continue
        cb = state.get("callback") or {}

        if not cb.get("url") or cb.get("delivered") or cb.get("abandoned"):
            finished = state.get("finished_at") or 0
            if finished and now - finished > STATE_RETENTION_SECONDS:
                path.unlink(missing_ok=True)
                log.info("retired assemble state for %s (%.1f days old)",
                         job, (now - finished) / 86400)
            continue

        if now < cb.get("next_attempt_at", 0):
            continue

        queued = cb.get("first_queued_at") or state.get("finished_at") or now
        if now - queued > CALLBACK_MAX_AGE_SECONDS:
            cb["abandoned"] = True
            state["callback"] = cb
            _write_assemble_state(job, state)
            # Loud on purpose. Nothing polls, so this is the only place the
            # stuck row is ever going to be mentioned.
            log.error("callback %s -> %s ABANDONED after %.1f hours and %d "
                      "attempts (last: %s %s). The episode is still "
                      "'assembling' in n8n and needs a hand: POST the payload "
                      "from %s to the webhook by hand, or resubmit the render.",
                      job, cb.get("url"), (now - queued) / 3600,
                      cb.get("attempts", 0), cb.get("last_status"),
                      _clip(cb.get("last_error") or "", 200), path)
            continue

        _queue_callback(job)


def _notifier_loop() -> None:
    while not _notify_stop.wait(CALLBACK_TICK):
        try:
            _notify_scan()
        except Exception:  # noqa: BLE001
            log.exception("callback notifier scan failed")


def _promote_orphans() -> int:
    """Fail every render owned by a process that is no longer here, and tell n8n.

    /assemble/{job}/status already detects this, but only when something asks.
    After the poller is deleted nothing asks, so the detection has to move to
    startup - which is also strictly better: an OOM kill is reported the moment
    the service comes back, rather than on whatever tick would have caught it.
    """
    if not ASSEMBLE_STATE_DIR.exists():
        return 0
    promoted = 0
    for path in sorted(ASSEMBLE_STATE_DIR.glob("*.json")):
        job = path.stem
        state = _read_assemble_state(job)
        if not state or state.get("state") != "running":
            continue
        if state.get("instance") == SERVICE_INSTANCE:
            continue  # unreachable at startup; do not guess about it
        log.error("assemble %s was running under instance %s when the service "
                  "stopped; promoting to failed", job, state.get("instance"))
        state["state"] = "failed"
        state["status_code"] = 503
        state["error"] = ("the service restarted while this assemble was "
                          "running; resubmit it")
        state["finished_at"] = time.time()
        cb = state.get("callback")
        if cb:
            cb["first_queued_at"] = time.time()
            cb["next_attempt_at"] = 0
        _write_assemble_state(job, state)
        promoted += 1
    return promoted


def _write_terminal_state(job: str, patch: dict) -> None:
    """Write a terminal record, preserving the callback block from submit.

    The three call sites in _assemble_worker used to build a fresh dict, which
    would silently discard the callback target and leave the render finished
    and unreported.
    """
    prior = _read_assemble_state(job) or {}
    state = {
        "instance": SERVICE_INSTANCE,
        "started_at": prior.get("started_at"),
        "finished_at": time.time(),
        **patch,
    }
    cb = prior.get("callback")
    if cb:
        cb["first_queued_at"] = time.time()
        cb["next_attempt_at"] = 0
        state["callback"] = cb
    _write_assemble_state(job, state)
    if cb and cb.get("url"):
        _queue_callback(job)


def _assemble_worker(job: str, allow_silent: int, allow_gaps: int, expect: int) -> None:
    started = time.monotonic()
    try:
        d, _final, payload, _stats = _assemble_core(job, allow_silent, allow_gaps, expect)
        # The only log line naming the job dir before it goes: after this,
        # the archived copy is the only one left.
        log.info("assemble %s (background): returning a path pointer and "
                 "deleting %s", job, d)
        shutil.rmtree(d, ignore_errors=True)
        _write_terminal_state(job, {"state": "done", "result": payload})
        log.info("assemble %s (background) done in %.0fs", job,
                 time.monotonic() - started)
    except HTTPException as exc:
        # The same 409/500 the synchronous route would have returned, kept as
        # data so the callback payload carries the real reason instead of a
        # bare "failed".
        log.warning("assemble %s (background) failed after %.0fs: %s %s",
                    job, time.monotonic() - started, exc.status_code, exc.detail)
        _write_terminal_state(job, {"state": "failed",
                                    "status_code": exc.status_code,
                                    "error": str(exc.detail)})
    except Exception as exc:  # noqa: BLE001 - a worker thread must not die silently
        log.exception("assemble %s (background) crashed after %.0fs",
                      job, time.monotonic() - started)
        _write_terminal_state(job, {"state": "failed", "status_code": 500,
                                    "error": f"{type(exc).__name__}: {exc}"})
    finally:
        with _assemble_lock:
            _active_assemblies.discard(job)
        _assembly_slots.release()


@app.post("/assemble/{job}")
def assemble(job: str, request: Request, body: Optional[AssembleBody] = None,
             allow_silent: int = 0, allow_gaps: int = 0, expect: int = 0,
             deliver: str = "file", background: int = 0) -> Response:
    """Assemble a job's segments into the finished episode.

    background=0 (default) renders inline and returns the result, which is the
    v4.6 behaviour and is fine for short jobs.

    background=1 returns 202 immediately and renders in a thread. The service
    POSTs the terminal state (done or failed) to CALLBACK_URL (or the
    callback_url in the JSON body) when the render finishes, retrying with
    backoff until it gets a 2xx. GET /assemble/{job}/status still exists but is
    diagnostic only now - nothing in production polls it.
    """
    check_auth(request)
    sweep_old_jobs()

    if background:
        if deliver != "path":
            raise HTTPException(
                status_code=400,
                detail="background=1 requires deliver=path - there is no open "
                       "request to stream the file back on")
        if ARCHIVE_DIR is None:
            raise HTTPException(
                status_code=400,
                detail="background=1 needs ARCHIVE_DIR set in the service env")
        override_url = body.callback_url if body else None
        override_token = body.callback_token if body else None
        if (override_url or override_token) and not ALLOW_CALLBACK_OVERRIDE:
            raise HTTPException(status_code=403, detail="callback overrides are disabled")
        if override_url:
            parsed = urlsplit(override_url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise HTTPException(status_code=400, detail="callback override must be an HTTPS URL")
            allowed = CALLBACK_ALLOWED_HOSTS or ({urlsplit(CALLBACK_URL).hostname.lower()} if CALLBACK_URL and urlsplit(CALLBACK_URL).hostname else set())
            if not allowed or parsed.hostname.lower() not in allowed:
                raise HTTPException(status_code=403, detail="callback override host is not allowed")
        cb_url = override_url or CALLBACK_URL
        if not cb_url:
            # Fail closed. Nothing polls this service, so a background render
            # with no callback target renders for forty minutes, archives the
            # file and tells nobody - the exact failure the poller used to
            # cover for.
            raise HTTPException(
                status_code=400,
                detail="background=1 needs a callback target: set CALLBACK_URL "
                       "in /etc/korinth-ffmpeg.env or send callback_url in the "
                       "body. Nothing polls this service, so a render with no "
                       "callback would finish and never be collected.")
        # Cheap up-front checks so an obviously broken job fails on submit
        # rather than looking healthy for 40 minutes.
        d = job_dir(job)
        if not d.exists():
            raise HTTPException(status_code=404, detail=f"no such job {job}")

        with _assemble_lock:
            cur = _read_assemble_state(job)
            if (cur and cur.get("state") == "running"
                    and cur.get("instance") == SERVICE_INSTANCE):
                log.info("assemble %s: background render already running, "
                         "ignoring duplicate submit", job)
                return JSONResponse(
                    {"job": job, "state": "running",
                     "detail": "an assemble is already running for this job",
                     "status_url": f"/assemble/{job}/status"},
                    status_code=202)
            if not _assembly_slots.acquire(blocking=False):
                raise HTTPException(status_code=429, detail=f"background assembly capacity reached ({MAX_BACKGROUND_ASSEMBLIES}); retry without regenerating media", headers={"Retry-After": "60"})
            _active_assemblies.add(job)
            try:
                _write_assemble_state(job, {
                "state": "running",
                "instance": SERVICE_INSTANCE,
                "started_at": time.time(),
                "callback": {
                    "url": cb_url,
                    "token": (body.callback_token if body else None) or None,
                    "request_id": _request_id.get(),
                    "delivered": False,
                    "attempts": 0,
                    "next_attempt_at": 0,
                },
                })
            except Exception:
                _active_assemblies.discard(job)
                _assembly_slots.release()
                raise

        log.info("assemble %s: submitted for background render "
                 "(allow_silent=%d allow_gaps=%d expect=%d callback=%s)", job,
                 allow_silent, allow_gaps, expect, cb_url)
        try:
            threading.Thread(
                target=_assemble_worker,
                args=(job, allow_silent, allow_gaps, expect),
                name=f"assemble-{job}",
                daemon=True,
            ).start()
        except Exception:
            with _assemble_lock:
                _active_assemblies.discard(job)
            _assembly_slots.release()
            _write_terminal_state(job, {"state": "failed", "status_code": 503,
                                        "error": "background worker could not start; resubmit the job"})
            raise

        return JSONResponse(
            {"job": job, "state": "running", "callback_url": cb_url,
             "status_url": f"/assemble/{job}/status"},
            status_code=202)

    d, final, payload, stats = _assemble_core(job, allow_silent, allow_gaps, expect)

    if deliver == "path":
        if payload["path"] is None:
            raise HTTPException(
                status_code=400,
                detail="?deliver=path needs ARCHIVE_DIR set in the service env")
        # Both branches delete the job dir, so this is the last log line that
        # can name the files. After it, the only copy is the archived one.
        log.info("assemble %s: returning a path pointer and deleting %s",
                 job, d)
        shutil.rmtree(d, ignore_errors=True)
        return JSONResponse(payload, headers=stats)

    data = (d / final).read_bytes()
    log.info("assemble %s: streaming %.1f MB back and deleting %s",
             job, len(data) / 1e6, d)
    shutil.rmtree(d, ignore_errors=True)
    return Response(
        content=data,
        media_type="video/mp4",
        headers={**stats,
                 "Content-Disposition": f'attachment; filename="{job}.mp4"'},
    )


@app.get("/assemble/{job}/status")
def assemble_status(job: str, request: Request) -> dict:
    """Report a background assemble. Diagnostic only - nothing in production
    polls this any more; the terminal state is delivered by callback instead.

    states: running | done | failed. `done` carries the same payload the
    synchronous deliver=path route returns, so a caller can use it unchanged.
    Includes the `callback` block, so this also answers "did n8n ever get
    told" for a job stuck in n8n's `assembling` state.
    """
    check_auth(request)
    state = _read_assemble_state(job)
    if state is None:
        raise HTTPException(status_code=404,
                            detail=f"no assemble has been submitted for job {job}")

    # A "running" record stamped by a previous process means the service went
    # down mid-render. Nothing is going to finish it, so say so rather than
    # leaving the caller polling a job that no longer has a worker.
    if state.get("state") == "running" and state.get("instance") != SERVICE_INSTANCE:
        log.warning("assemble %s: status polled but the owning instance is "
                    "gone (state instance=%s, current=%s)",
                    job, state.get("instance"), SERVICE_INSTANCE)
        return {
            "job": job,
            "state": "failed",
            "status_code": 503,
            "error": "the service restarted while this assemble was running; "
                     "resubmit it",
            "started_at": state.get("started_at"),
        }

    return {"job": job, **state}


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
    meta = {
        "format": fmt, "file": f"narration.{ext}",
        "sample_rate": int(payload.get("sample_rate") or 24000),
        "channels": int(payload.get("channels") or 1),
    }
    (d / "narration.meta").write_text(json.dumps(meta), encoding="utf-8")
    # Raw pcm carries no header, so /concat can only probe it correctly if the
    # rate and channel count sent here are right. A wrong pair produces audio
    # that plays at the wrong speed rather than an error.
    log.info("stored %s narration: %.0f KB %s @ %dHz x%d", job,
             len(audio) / 1024, fmt, meta["sample_rate"], meta["channels"])
    return {"job": job, "bytes": len(audio), "format": fmt}


@app.post("/excerpt")
def excerpt(body: ExcerptBody, request: Request) -> Response:
    """Create a vertical excerpt from an archived long-form episode.

    The complete 16:9 source remains visible at its original aspect ratio over
    a blurred fill layer. This avoids stretching portraits or cropping away
    characters merely to fill a 9:16 canvas.
    """
    check_auth(request)
    if ARCHIVE_DIR is None:
        raise HTTPException(status_code=503, detail="ARCHIVE_DIR is not configured")
    if not (0 <= body.start_seconds):
        raise HTTPException(status_code=400, detail="start_seconds must be non-negative")
    if not (5 <= body.duration_seconds <= 60):
        raise HTTPException(status_code=400, detail="duration_seconds must be between 5 and 60")

    archive_root = ARCHIVE_DIR.resolve()
    supplied = Path(body.path)
    source = (supplied if supplied.is_absolute() else archive_root / supplied).resolve()
    try:
        source.relative_to(archive_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="excerpt path must be inside ARCHIVE_DIR")
    if not source.is_file():
        raise HTTPException(status_code=404, detail="archived episode not found")

    source_duration = probe_duration(source)
    if body.start_seconds >= source_duration:
        raise HTTPException(status_code=400, detail="start_seconds is beyond the source duration")
    duration = min(body.duration_seconds, source_duration - body.start_seconds)
    excerpt_dir = DATA_DIR / "_excerpt"
    excerpt_dir.mkdir(parents=True, exist_ok=True)
    output = excerpt_dir / f"{source.stem}-{time.time_ns()}.mp4"
    video_filter = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:10[bg];"
        "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    proc = run([
        "ffmpeg", "-y", "-ss", f"{body.start_seconds:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}", "-filter_complex", video_filter,
        "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-r", "30", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output)
    ], cwd=excerpt_dir, label=f"excerpt {source.name}")
    if proc.returncode != 0 or not output.exists():
        output.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"excerpt failed: {(proc.stderr or '')[-2000:]}")
    data = output.read_bytes()
    output.unlink(missing_ok=True)
    return Response(content=data, media_type="video/mp4", headers={
        "Content-Disposition": f'attachment; filename="{source.stem}-short.mp4"',
        "X-Source-Duration": f"{source_duration:.2f}",
        "X-Excerpt-Start": f"{body.start_seconds:.2f}",
        "X-Excerpt-Duration": f"{duration:.2f}",
        "X-Canvas": "1080x1920",
    })


@app.post("/concat/{job}")
def concat(job: str, request: Request) -> Response:
    check_auth(request)
    sweep_old_jobs()
    d = job_dir(job)
    clips = sorted(d.glob("[0-9]*.mp4")) if d.exists() else []
    if not clips:
        raise HTTPException(status_code=404, detail=f"no clips stored for job {job}")

    concat_started = time.monotonic()
    log.info("concat %s: %d clips %s", job, len(clips),
             [c.name for c in clips])

    (d / "concat.txt").write_text("".join(f"file '{c.name}'\n" for c in clips), encoding="utf-8")
    joined = "joined.mp4"
    proc = run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", "-r", "24", "-pix_fmt", "yuv420p",
                joined], cwd=d, label=f"concat {job}")
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
                         ["-show_entries", "format=duration", "-of", "csv=p=0", meta["file"]],
                         cwd=d, label=f"ffprobe narration {job}")
        try:
            narr_dur = float((narr_probe.stdout or "").strip())
        except ValueError:
            # 0.0 means the pad below is computed from the video alone, so a
            # voiceover longer than the picture gets cut off at the end.
            log.warning("concat %s: could not read a duration from %s "
                        "(format=%s, args=%s); treating it as 0.0s", job,
                        meta["file"], meta["format"], pcm_args or "none")
            narr_dur = 0.0
        target = max(video_dur, narr_dur) + 0.4
        pad = max(0.0, target - video_dur)
        log.info("concat %s: muxing narration, video=%.2fs narration=%.2fs "
                 "-> %.2fs (pad %.2fs, ambience %s)", job, video_dur, narr_dur,
                 target, pad, AMBIENT_VOLUME)
        filt = (f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.2f}[v];"
                f"[0:a]volume={AMBIENT_VOLUME}[amb];[1:a]apad[vo];"
                f"[amb][vo]amix=inputs=2:duration=longest:normalize=0[a]")
        proc = run(["ffmpeg", "-y", "-i", joined, *pcm_args, "-i", meta["file"],
                    "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
                    "-t", f"{target:.2f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", "-r", "24", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", final], cwd=d,
                   label=f"mux {job}")
        if proc.returncode != 0 or not (d / final).exists():
            raise HTTPException(status_code=500, detail=f"mux failed: {(proc.stderr or '')[-2000:]}")
        narration_used = True
        out_duration = target
    else:
        # No narration.meta, so /narration was never called for this job.
        log.info("concat %s: no narration stored, finalising video only", job)
        proc = run(["ffmpeg", "-y", "-i", joined, "-c", "copy",
                    "-movflags", "+faststart", final], cwd=d,
                   label=f"finalise {job}")
        if proc.returncode != 0 or not (d / final).exists():
            raise HTTPException(status_code=500, detail=f"finalise failed: {(proc.stderr or '')[-2000:]}")

    data = (d / final).read_bytes()
    clip_count = len(clips)
    log.info("concat %s complete: %d clips, narration=%s, %.1fs, %.1f MB in "
             "%.0fs; deleting %s", job, clip_count, narration_used,
             out_duration, len(data) / 1e6,
             time.monotonic() - concat_started, d)
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
    d = job_dir(job)
    # Logged whether or not it existed: this is the only record that the segments
    # were deleted on purpose rather than swept or never stored.
    log.info("deleting job %s (existed=%s)", job, d.exists())
    shutil.rmtree(d, ignore_errors=True)
    return {"job": job, "deleted": True}


# Systemd runs uvicorn directly; this is here so `python app.py` works for a
# quick manual run on the box.
if __name__ == "__main__":
    import uvicorn

    # LOG_LEVEL drives uvicorn too, so LOG_LEVEL=DEBUG on a manual run also turns
    # on its access log. EFFECTIVE_LOG_LEVEL is already validated, so a typo
    # cannot stop uvicorn from starting.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                timeout_keep_alive=120,
                log_level=EFFECTIVE_LOG_LEVEL.lower())
