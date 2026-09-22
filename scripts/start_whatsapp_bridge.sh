#!/usr/bin/env bash
# Start the WhatsApp webhook server + a Cloudflare Tunnel pointing at it.
# Prints the public HTTPS URL you need to paste into the Meta Developer App's
# "Callback URL" field (append "/webhook").
#
# Usage:
#   chmod +x scripts/start_whatsapp_bridge.sh
#   scripts/start_whatsapp_bridge.sh
#
# Stop with Ctrl+C (both processes exit cleanly).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PORT="${WHATSAPP_WEBHOOK_PORT:-8090}"

cleanup() {
    echo
    echo "Shutting down..."
    [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
    [ -n "${WEBHOOK_PID:-}" ] && kill "$WEBHOOK_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[1/2] Starting WhatsApp webhook server on :$PORT ..."
PYTHONPATH=src .venv/bin/python scripts/whatsapp_webhook.py &
WEBHOOK_PID=$!

# Wait for the webhook to listen.
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
        echo "    ✅ webhook ready (PID $WEBHOOK_PID)"
        break
    fi
    sleep 1
done

echo "[2/2] Starting Cloudflare Tunnel ..."
echo "      (free, no account needed — gives you a *.trycloudflare.com URL)"
echo
cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" 2>&1 | tee /tmp/cf-tunnel.log &
TUNNEL_PID=$!

# Watch the log for the public URL and print a callout.
( tail -F /tmp/cf-tunnel.log 2>/dev/null | while read -r line; do
    case "$line" in
        *trycloudflare.com*)
            url=$(printf '%s' "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
            if [ -n "$url" ]; then
                echo
                echo "═══════════════════════════════════════════════════════════════"
                echo " 📡  PUBLIC WEBHOOK URL ready:"
                echo
                echo "     $url/webhook"
                echo
                echo " Paste that into Meta Developer App →"
                echo "   WhatsApp → Configuration → Callback URL"
                echo " Verify token: \$WHATSAPP_VERIFY_TOKEN (see your .env)"
                echo "═══════════════════════════════════════════════════════════════"
                echo
                break
            fi
            ;;
    esac
done ) &

wait
