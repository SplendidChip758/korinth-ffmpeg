#!/usr/bin/env bash
# korinth-ffmpeg installer/updater — safe to re-run.
# Clones on first run, fetches+resets on every run after. Preserves the
# auth token and any local env overrides. Run as root inside the LXC.

set -euo pipefail

REPO_URL="${KORINTH_REPO_URL:-git@github.com:SplendidChip758/korinth-ffmpeg.git}"
REF="${KORINTH_REF:-main}"                 # branch, tag, or commit to deploy
INSTALL_DIR="/opt/korinth-ffmpeg"
ENV_FILE="/etc/korinth-ffmpeg.env"
SERVICE_NAME="korinth-ffmpeg"
VENV_DIR="${INSTALL_DIR}/venv"
APP_USER="korinth"
DATA_DIR="/var/lib/korinth-ffmpeg"

echo "== korinth-ffmpeg install/update =="

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this as root inside the LXC." >&2
  exit 1
fi

# --- 0. system deps, service user, data dir ----------------------------------
# Cheap no-ops on an already-provisioned box, but they mean a bare LXC needs
# nothing done to it by hand before this script runs.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-pip ffmpeg curl ca-certificates openssl

id -u "${APP_USER}" >/dev/null 2>&1 \
  || useradd --system --home-dir "${INSTALL_DIR}" --shell /usr/sbin/nologin "${APP_USER}"

mkdir -p "${INSTALL_DIR}" "${DATA_DIR}" /etc/korinth-ffmpeg

# The service account key for Cloud Text-to-Speech (POST /narrate) is dropped here by
# hand — install.sh has no way to obtain it. If it's already present, keep its
# permissions correct on every re-run.
if [ -f /etc/korinth-ffmpeg/service-account.json ]; then
  chown root:"${APP_USER}" /etc/korinth-ffmpeg/service-account.json
  chmod 0640 /etc/korinth-ffmpeg/service-account.json
fi

# --- 1. get/refresh the code -------------------------------------------------
if [ -d "${INSTALL_DIR}/.git" ]; then
  echo "-> existing checkout found, updating"
  git -C "${INSTALL_DIR}" fetch --all --tags --prune
  git -C "${INSTALL_DIR}" reset --hard "origin/${REF}" 2>/dev/null \
    || git -C "${INSTALL_DIR}" reset --hard "${REF}"
else
  echo "-> no checkout found, cloning"
  git clone --branch "${REF}" "${REPO_URL}" "${INSTALL_DIR}" 2>/dev/null \
    || { git clone "${REPO_URL}" "${INSTALL_DIR}" && git -C "${INSTALL_DIR}" checkout "${REF}"; }
fi

DEPLOYED_SHA="$(git -C "${INSTALL_DIR}" rev-parse --short HEAD)"
DEPLOYED_VERSION="$(cat "${INSTALL_DIR}/VERSION" 2>/dev/null || echo "0.0.0")"
echo "-> deploying ${DEPLOYED_VERSION}+${DEPLOYED_SHA}"

# Expose these to app.py so /health can report them without a code change
export KORINTH_VERSION="${DEPLOYED_VERSION}"
export KORINTH_GIT_SHA="${DEPLOYED_SHA}"

# --- 2. env file (created once, never overwritten) ---------------------------
if [ ! -f "${ENV_FILE}" ]; then
  echo "-> no env file found, creating from template"
  cp "${INSTALL_DIR}/korinth-ffmpeg.env.example" "${ENV_FILE}"
  TOKEN="$(openssl rand -hex 32)"
  sed -i "s/^X_AUTH_TOKEN=.*/X_AUTH_TOKEN=${TOKEN}/" "${ENV_FILE}"
  echo "   generated new X-Auth-Token — copy it into your n8n Header Auth credential:"
  echo "   ${TOKEN}"
  TOKEN_STATUS="generated above"
else
  echo "-> env file exists, leaving token untouched"
  TOKEN_STATUS="preserved (retrieve with root-only access to ${ENV_FILE})"
fi

# v4.6.0 deliberately does NOT migrate TTS_VOICE / TTS_STYLE out of the env
# file. They are no longer where the voice normally comes from, but they are
# still the layer that answers when the share is down, and deleting them is
# what would make an unreachable table sound wrong instead of sounding
# unchanged. Leaving them also keeps the documented rollback (remove
# tts-presets.json from the share) a one-step operation.

# Persist version info alongside the rest of the env so app.py can read it
# without needing systemd Environment= edits on every deploy
sed -i '/^KORINTH_VERSION=/d;/^KORINTH_GIT_SHA=/d' "${ENV_FILE}"
{
  echo "KORINTH_VERSION=${DEPLOYED_VERSION}"
  echo "KORINTH_GIT_SHA=${DEPLOYED_SHA}"
} >> "${ENV_FILE}"

# The token lives in here, so the service account may read it and nobody else.
chmod 0640 "${ENV_FILE}"
chown root:"${APP_USER}" "${ENV_FILE}"

# --- 3. python env ------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
  echo "-> creating venv"
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${INSTALL_DIR}/requirements.txt"

# Only the data dir changes hands. The checkout stays root-owned — the service
# only needs to read it, and chowning it away from root makes every later
# `git -C` re-run fail git's dubious-ownership check.
chown -R "${APP_USER}:${APP_USER}" "${DATA_DIR}"

# --- 4. systemd unit -----------------------------------------------------------
install -m 644 "${INSTALL_DIR}/systemd/${SERVICE_NAME}.service" \
  "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}" >/dev/null
systemctl restart "${SERVICE_NAME}"

# --- 5. confirm ----------------------------------------------------------------
sleep 3
if ! systemctl is-active --quiet "${SERVICE_NAME}"; then
  echo "!! service failed to start. Last 40 log lines:" >&2
  journalctl -u "${SERVICE_NAME}" --no-pager --lines=40 >&2
  exit 1
fi

echo "-> checking /health"
HEALTH="unchecked (curl not installed)"
if command -v curl >/dev/null; then
  HEALTH="$(curl -fsS "http://127.0.0.1:8080/health" || echo 'HEALTH CHECK FAILED')"
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "======================================================================"
echo " korinth-ffmpeg ${DEPLOYED_VERSION}+${DEPLOYED_SHA} deployed and running."
echo
echo "   Health      ${HEALTH}"
echo "   Base URL    http://${IP:-<this-lxc-ip>}:8080"
echo "   Header      X-Auth-Token"
echo "   Token       ${TOKEN_STATUS}"
if ! grep -q '^CALLBACK_URL=.' "${ENV_FILE}"; then
  echo "   WARNING     CALLBACK_URL is unset; background assembly will be refused."
fi
if ! grep -q '^CALLBACK_TOKEN=.' "${ENV_FILE}"; then
  echo "   WARNING     CALLBACK_TOKEN is unset; callbacks will be unauthenticated."
fi
echo
echo " POST /narrate needs a service account key at"
echo " /etc/korinth-ffmpeg/service-account.json (Cloud Text-to-Speech), and"
echo " reads its narration voice from"
echo " /mnt/korinth-industries/compiled/tts-presets.json — copy"
echo " tts-presets.json from this repo if the share has none yet. Without it,"
echo " narration falls back to TTS_VOICE / TTS_STYLE and then to the locked"
echo " read compiled into korinth_tts_presets.py, so it still sounds right."
echo " See korinth-ffmpeg.env.example for setup steps."
echo "======================================================================"
