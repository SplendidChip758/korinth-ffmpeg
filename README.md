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

`POST /narrate` calls Gemini on **Vertex AI**, authenticated as a service
account — GEAP's GCP project issues service account keys, not AI Studio
`GEMINI_API_KEY` values. It needs one thing: a service account JSON key
(role: Vertex AI User) at `/etc/korinth-ffmpeg/service-account.json`, placed
by hand — the installer has no way to generate one. The project id is read
straight out of the key file's own `project_id` field, so there's no separate
project config to keep in sync.

See `korinth-ffmpeg.env.example` for the exact setup steps. Until the key is
in place the route returns 500; `/health` reports `tts_configured` and
`gcp_project` so this is visible without a test render.

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
{"status": "ok", "version": "4.0.0", "git_sha": "a1b2c3d", "ffmpeg": "6.1",
 "auth": true, "tts_configured": true, "tts_voice": "Orus"}
```

If the env vars are absent (running `app.py` by hand outside systemd) the
version falls back to the `VERSION` file and the SHA reads `unknown`.

This replaces manually bumping the version string in code — the deployed
commit is always unambiguous, which matters when a run's `/health` check is
the only signal for which build is live.

## Repo layout

- `app.py` — the service itself
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
