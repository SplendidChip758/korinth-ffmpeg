#!/usr/bin/env bash
# korinth-ffmpeg updater — pulls the latest code, refreshes deps, restarts
# the service. Assumes install.sh has already run at least once (system
# deps, service user, venv, env file, systemd unit all in place). Safe to
# re-run. Run as root inside the LXC.
#
# For first-time setup, or if anything above is missing, use install.sh
# instead — it does everything this script does plus the one-time
# provisioning steps.

set -euo pipefail

REF="${KORINTH_REF:-main}"                 # branch, tag, or commit to deploy
INSTALL_DIR="/opt/korinth-ffmpeg"
ENV_FILE="/etc/korinth-ffmpeg.env"
SERVICE_NAME="korinth-ffmpeg"
VENV_DIR="${INSTALL_DIR}/venv"

echo "== korinth-ffmpeg update =="

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this as root inside the LXC." >&2
  exit 1
fi

if [ ! -d "${INSTALL_DIR}/.git" ] || [ ! -f "${ENV_FILE}" ]; then
  echo "No existing install found at ${INSTALL_DIR} / ${ENV_FILE}." >&2
  echo "Run install.sh first (curl -fsSL .../install.sh | bash)." >&2
  exit 1
fi

# --- 1. refresh the code ------------------------------------------------------
echo "-> fetching ${REF}"
git -C "${INSTALL_DIR}" fetch --all --tags --prune
git -C "${INSTALL_DIR}" reset --hard "origin/${REF}" 2>/dev/null \
  || git -C "${INSTALL_DIR}" reset --hard "${REF}"

DEPLOYED_SHA="$(git -C "${INSTALL_DIR}" rev-parse --short HEAD)"
DEPLOYED_VERSION="$(cat "${INSTALL_DIR}/VERSION" 2>/dev/null || echo "0.0.0")"
echo "-> deploying ${DEPLOYED_VERSION}+${DEPLOYED_SHA}"

# Keep the env file's version stamp in sync so /health reports the right thing.
sed -i '/^KORINTH_VERSION=/d;/^KORINTH_GIT_SHA=/d' "${ENV_FILE}"
{
  echo "KORINTH_VERSION=${DEPLOYED_VERSION}"
  echo "KORINTH_GIT_SHA=${DEPLOYED_SHA}"
} >> "${ENV_FILE}"

# --- 2. python deps ------------------------------------------------------------
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${INSTALL_DIR}/requirements.txt"

# --- 3. systemd unit + restart --------------------------------------------------
install -m 644 "${INSTALL_DIR}/systemd/${SERVICE_NAME}.service" \
  "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload
systemctl restart "${SERVICE_NAME}"

# --- 4. confirm ------------------------------------------------------------------
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

echo
echo "======================================================================"
echo " korinth-ffmpeg ${DEPLOYED_VERSION}+${DEPLOYED_SHA} updated and running."
echo "   Health      ${HEALTH}"
echo "======================================================================"
