# korinth-ffmpeg

Storage / stitch / TTS service for the Korinth Industries n8n pipelines.
Runs in Proxmox LXC 429 (node 3), fronted by `X-Auth-Token` header auth.

## Routes

Shorts path (unchanged since v2):

| | |
|---|---|
| `PUT /clip/{job}/{index}` | raw mp4 bytes |
| `PUT /narration/{job}` | whole-video voiceover, base64 |
| `POST /concat/{job}` | join + mux one narration track |

Long-form path (v4):

| | |
|---|---|
| `PUT /image64/{job}/{index}` | base64 PNG, `?motion=zoom_in\|zoom_out\|pan_left\|pan_right\|kenburns` |
| `PUT /clip64/{job}/{index}` | base64 mp4 |
| `POST /narrate/{job}/{index}` | text → Gemini TTS → wav, returns the **measured** duration |
| `POST /assemble/{job}` | per-segment timing, Ken Burns, concat, mux |

Plus `GET /lastframe/{job}/{index}`, `GET /health`, `GET /jobs`,
`DELETE /job/{job}`. Everything except `/health` requires the token header.

`/narrate` generates the voiceover on this box rather than in n8n so
`/assemble` can probe each wav on local disk. That is what lets segment
durations be measured instead of estimated from a words-per-second constant —
the constant on the shorts pipeline has needed recalibrating twice.

`/assemble` returns 409 if any segment is missing its narration; pass
`?allow_silent=1` to render anyway for inspection. A silent segment ships
broken without erroring, so the default is to fail before n8n reaches the
upload node. `/assemble` also checks segment indices are contiguous (409 with
the gap list unless `?allow_gaps=1`) and, if `?expect=N` is passed, that
exactly `N` segments showed up — both cover a generation step that failed
silently rather than one that errored loudly.

If `ARCHIVE_DIR` is set, `/assemble` copies the finished mp4 there (as
`{job}.mp4`, written atomically) before deleting the job folder — pass
`?deliver=path` to get back a small JSON pointer (`{"path": ...}`) instead of
the full video bytes over HTTP, useful when n8n just needs to hand the file to
a downstream step rather than hold it in memory. Without `ARCHIVE_DIR`,
`?deliver=path` 400s, and the default behaviour is unchanged: the bytes stream
back and nothing is kept on disk after the response.

## Deploy / update

```bash
# first time
curl -fsSL https://raw.githubusercontent.com/youruser/korinth-ffmpeg/main/install.sh | bash
# or, if the repo is already checked out at /opt/korinth-ffmpeg:
cd /opt/korinth-ffmpeg && ./install.sh
```

`install.sh` is idempotent — install system deps, create the `korinth` user and
`/var/lib/korinth-ffmpeg`, clone-or-pull, rebuild the venv, reinstall the
systemd unit, restart, confirm via `/health`. It never touches
`/etc/korinth-ffmpeg.env` after the token exists, so re-running never breaks
the n8n Header Auth credential. Run it as root; a bare LXC needs nothing done
to it by hand first.

`POST /narrate` calls Gemini TTS on **Cloud Text-to-Speech**, authenticated as
a service account — GEAP's GCP project issues service account keys, not AI
Studio `GEMINI_API_KEY` values. It needs two things: the API enabled
(`gcloud services enable texttospeech.googleapis.com`) and a service account
JSON key at `/etc/korinth-ffmpeg/service-account.json`, placed by hand — the
installer has no way to generate one. The account needs
`roles/serviceusage.serviceUsageConsumer`, which is what makes the
`x-goog-user-project` billing header legal. The project id is read straight out
of the key file's own `project_id` field, so there's no separate project config
to keep in sync.

The style instruction goes in `input.prompt`, a field of its own, so it is
never read aloud as part of the narration.

### TTS presets

Voice and style come from a named preset table on the share rather than from
env vars — a preset carries both fields together, so the pair cannot be
half-changed into a read that does one job and not the other, and the rejected
auditions keep their reasons. `korinth_tts_presets.py` owns the whole decision:
the file, the mtime reload, the env fallback and the last-resort read. It is
standard library only and imports nothing from `app.py`, so it can be exercised
without the service running:

```bash
python korinth_tts_presets.py tts-presets.json
```

`tts-presets.json` in this repo is the master; copy it to
`/mnt/korinth-industries/compiled/tts-presets.json` (or point
`TTS_PRESETS_FILE` elsewhere). It is hand-written — a canon build populating
`compiled/` must copy it through, not overwrite it. The service only reads it,
re-reads it whenever the mtime changes, and keeps serving the last good copy if
the file is mid-edit or briefly malformed: a typo in a voice prompt must not
take narration down halfway through a fourteen-segment render.

**One narrator, every episode, whoever the episode is about.** That is the
`narrator` preset — Umbriel plus the locked style prompt. It is the only entry
the loader can select; the five per-character candidates live under
`_candidates`, which nothing reads, so none of them can go live by accident.
Promoting one means moving it into `presets` *and* auditioning it first.

Precedence, highest first:

1. explicit `voice` / `style` in the request body — **per field**, which is what
   makes auditioning cheap: hold the voice, vary the prompt
2. the preset named by `preset` in the body, or the table's `default_preset`
3. `TTS_VOICE` / `TTS_STYLE` from the env file — the pre-4.6.0 behaviour, so a
   box whose share is down still sounds correct
4. the locked read hardcoded in `korinth_tts_presets.py`

An unknown preset name does **not** fail the request; it falls back to
`default_preset`. Failing segment nine of fourteen because of a typo in a slug
is a worse outcome than a consistent wrong narrator. But the substitution is
visible rather than silent — `preset_substituted` is the field to alert on:

```json
{"duration_seconds": 11.84, "voice": "Umbriel", "preset": "narrator",
 "preset_requested": null, "preset_substituted": false}
```

`preset_requested` is null when the caller expressed no preference, which
includes the literal `"default"` that `korinth-produce` sends when the pending
row has no `tts_preset` — treating that as a miss would raise the flag on
ordinary calls and it would stop meaning anything.

Switching the voice is the audio equivalent of model drift, and once episodes
have aired it is worse than that: a narrator who changes voice mid-series is the
one continuity error an audience notices without being told to look for it. Do
it at an arc boundary or not at all.

To roll back: delete `tts-presets.json` from the share. Resolution falls through
to `TTS_VOICE` / `TTS_STYLE`, and failing those to the hardcoded locked read, so
narration keeps working and keeps sounding the same. Nothing needs reverting in
code.

See `korinth-ffmpeg.env.example` for the exact setup steps. Until the key is
in place the route returns 500; `/health` reports `tts_auth`, `tts_project` and
`tts_configured` so this is visible without a test render.

To deploy a specific tag/commit instead of `main`:

```bash
KORINTH_REF=v4.1.0 ./install.sh
```

## Versioning

`VERSION` is a plain semver bumped by hand on meaningful changes (new routes,
breaking response format). `install.sh` also captures the exact git SHA and
writes both into the env file; `app.py` reads `KORINTH_VERSION` /
`KORINTH_GIT_SHA` and reports them on `/health`:

```json
{"status": "ok", "version": "4.6.0", "git_sha": "a1b2c3d", "ffmpeg": "6.1",
 "auth": true, "tts_configured": true, "tts_auth": "service-account",
 "tts_project": "korinth-industries",
 "tts_presets": {"path": "/mnt/korinth-industries/compiled/tts-presets.json",
                 "loaded": true, "mtime": 1755734400.0, "error": null,
                 "count": 1, "names": ["narrator"], "default": "narrator",
                 "schema_version": 1}}
```

`tts_presets.loaded: false` with a non-null `error` means the table is not being
used and narration is coming from `TTS_VOICE` / `TTS_STYLE` or the hardcoded
read. That is a working service, not a broken one — but it is not the state you
want to discover after a voice edit that appeared to do nothing.

If the env vars are absent (running `app.py` by hand outside systemd) the
version falls back to the `VERSION` file and the SHA reads `unknown`.

This replaces manually bumping the version string in code — the deployed
commit is always unambiguous, which matters when a run's `/health` check is
the only signal for which build is live.

## Repo layout

- `app.py` — the service itself
- `korinth_tts_presets.py` — narration voice resolution; stdlib only, no
  dependency on `app.py`
- `tts-presets.json` — master copy of the preset table; copied to the share
- `requirements.txt`
- `VERSION`
- `korinth-ffmpeg.env.example` — template, copied once on first install
- `systemd/korinth-ffmpeg.service`
- `install.sh`

## Ops

```bash
systemctl status korinth-ffmpeg
journalctl -u korinth-ffmpeg -f
curl -H "X-Auth-Token: $TOKEN" http://192.168.20.129:8080/health
```

## Known gotchas

`apt-get update -qq` on this box hides network errors and looks like a hang, so
`install.sh` runs it unquieted. If a fresh LXC's install seems stuck on
venv/pip setup, check DNS/routing first — the real error is in that output.

`DATA_DIR` and `ARCHIVE_DIR` must both stay in step with `ReadWritePaths=` in
`systemd/korinth-ffmpeg.service`. `ProtectSystem=strict` means a mismatch shows
up as read-only-filesystem errors on the first `PUT` or `/assemble`, not at
startup.

Long-form stills are upscaled to 7680x4320 before `zoompan`. Imagen caps at 2K,
and cropping a 2K source down to a 1080p output leaves only ~1.07x of travel —
the move goes soft immediately without the upscale.
