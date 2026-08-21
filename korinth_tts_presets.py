"""
korinth-ffmpeg v4.6.0 - named narration presets.

ONE narrator for the whole channel, whoever an episode is an account of.

Decided 2026-08-21. Per-character voices were considered and set aside: the
format is a recovered transmission archive recounted by one voice, and a
narrator who changes voice partway through a series is the one continuity
error an audience notices without being told to look for it. The live preset
is 'narrator' - Umbriel plus the locked style prompt.

Five per-character voices are kept in tts-presets.json under _candidates so
the thinking is not lost. This module reads 'presets' only, so nothing there
can be selected by accident, and none of them has been auditioned.

Why a file on the share rather than two environment variables, when there is
only one voice: PROVENANCE. Nothing previously recorded which voice produced
which episode, so if episode 30 sounds different from episode 12 there is no
way to tell whether the voice changed or the read drifted.
korinth_story_log.tts_preset now records it, the same way lookClause records
the visual style. The style prompt is also a paragraph of prose that matters,
and an env file is a bad place to keep, diff, or review one.

Editing a voice should not need a systemctl restart, so this reloads by mtime
and keeps serving the last good copy while the file is unreadable or mid-edit.
A typo saved into a voice prompt must not take narration down in the middle of
a fourteen-segment render.

Standard library only, and it imports nothing from app.py, so it can be
unit-tested and exercised from a shell without the service running.
"""

import json
import os
import threading

# compiled/ is what the service already reads at generation time. This file is
# hand-edited rather than generated, so if a canon-build step ever populates
# compiled/ from canon/, it must copy this through rather than overwrite it.
DEFAULT_PATH = os.environ.get(
    "TTS_PRESETS_FILE",
    "/mnt/korinth-industries/compiled/tts-presets.json",
)

# Last resort, used only when the share is unreachable AND the environment is
# empty. Duplicated from the file on purpose: an unmounted share should still
# narrate in the channel's own voice rather than fall through to whatever
# Google defaults to that week.
_FALLBACK_VOICE = "Umbriel"
_FALLBACK_STYLE = (
    "Read this as someone recounting a case they worked. Deliberate and "
    "grounded, quietly gripping, the pacing of a story being told rather than "
    "a report being read. Serious throughout, never theatrical. "
)

_lock = threading.Lock()
_cache = {"mtime": None, "path": None, "data": None, "error": None}


def _parse(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    presets = data.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError("no 'presets' object")

    for name, preset in presets.items():
        if not isinstance(preset, dict):
            raise ValueError("preset %r is not an object" % name)
        if not str(preset.get("voice") or "").strip():
            raise ValueError("preset %r has no voice" % name)

    default = data.get("default_preset")
    if default and default not in presets:
        raise ValueError("default_preset %r is not a preset" % default)

    return data


def load(path=None, force=False):
    """Return the presets document, re-reading only when the file has changed.

    Returns None if the file has never loaded successfully. Returns the last
    good copy if the file exists but is currently unparseable.
    """
    path = path or DEFAULT_PATH

    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        with _lock:
            _cache.update(path=path, error="unreadable: %s" % exc)
            return _cache["data"]

    with _lock:
        fresh = (
            not force
            and _cache["data"] is not None
            and _cache["path"] == path
            and _cache["mtime"] == mtime
        )
        if fresh:
            return _cache["data"]

    try:
        data = _parse(path)
    except Exception as exc:
        with _lock:
            _cache.update(path=path, error="%s: %s" % (type(exc).__name__, exc))
            return _cache["data"]

    with _lock:
        _cache.update(mtime=mtime, path=path, data=data, error=None)
        return data


def resolve(preset=None, voice=None, style=None, path=None):
    """Decide the voice and style for one /narrate call.

    Precedence, highest first:

      1. explicit voice / style in the request body. korinth-audition.sh drives
         /narrate directly and has to be able to override everything, which is
         the whole reason it follows the service's auth instead of calling
         Google itself.
      2. the named preset from tts-presets.json.
      3. TTS_VOICE / TTS_STYLE from the environment - the pre-4.6.0 behaviour,
         so a service whose share is down still sounds correct.
      4. the hardcoded locked read above.

    An unknown preset name falls back to default_preset rather than failing.
    A render is expensive and half-finished; a wrong-but-consistent narrator is
    a better outcome than a 400 fourteen segments in. The response reports
    preset_requested alongside preset, so the substitution is visible in the
    execution rather than silent.
    """
    doc = load(path)
    chosen = None
    body = {}

    # "default" means "no preference", not a preset that went missing.
    # korinth-produce sends it whenever the pending row has no tts_preset, so
    # treating it as a miss would raise preset_substituted on ordinary calls
    # and the flag would stop meaning anything.
    wanted = str(preset or "").strip()
    if wanted.lower() == "default":
        wanted = ""

    if doc:
        presets = doc.get("presets") or {}

        if wanted and wanted in presets:
            chosen = wanted
        else:
            fallback = doc.get("default_preset")
            if fallback in presets:
                chosen = fallback

        if chosen:
            body = presets.get(chosen) or {}

    out_voice = voice or body.get("voice") or os.environ.get("TTS_VOICE") or _FALLBACK_VOICE

    if style is not None:
        out_style = style
    elif body.get("style") is not None:
        out_style = body["style"]
    else:
        out_style = os.environ.get("TTS_STYLE", _FALLBACK_STYLE)

    requested = wanted or None

    return {
        "voice": out_voice,
        "style": out_style,
        "preset": chosen,
        "preset_requested": requested,
        "preset_substituted": bool(requested) and requested != chosen,
        "model": (doc or {}).get("model") or os.environ.get("TTS_MODEL") or None,
    }


def health(path=None):
    """Presets status for /health. Never raises."""
    doc = load(path)
    with _lock:
        error = _cache["error"]
        mtime = _cache["mtime"]
        where = _cache["path"] or (path or DEFAULT_PATH)

    return {
        "path": where,
        "loaded": doc is not None,
        "mtime": mtime,
        "error": error,
        "count": len(((doc or {}).get("presets") or {})),
        "names": sorted(((doc or {}).get("presets") or {}).keys()),
        "default": (doc or {}).get("default_preset"),
        "schema_version": (doc or {}).get("schema_version"),
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps(health(target), indent=2))
    for name in sorted(((load(target) or {}).get("presets") or {})):
        r = resolve(name, path=target)
        print("\n%-16s %s" % (name, r["voice"]))
        print("  %s" % r["style"].strip())
    print("\nunknown name falls back to: %s" % resolve("no-such-person", path=target)["preset"])