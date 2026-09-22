#!/usr/bin/env bash
# wire_twilio_to_hostel.sh
#
# One-shot script that wires a Twilio US trial number to the Hashim Girls
# Hostel LiveKit voice agent. Run this once you have a Twilio trial account
# with a US number purchased.
#
# Usage:
#   chmod +x wire_twilio_to_hostel.sh
#   ./wire_twilio_to_hostel.sh \
#       AC...your_account_sid... \
#       your_auth_token... \
#       +1XXXXXXXXXX \
#       +92XXXXXXXXXX
#
# Args:
#   $1  TWILIO_ACCOUNT_SID  — starts with "AC"
#   $2  TWILIO_AUTH_TOKEN
#   $3  TWILIO_PHONE_NUMBER — your trial US number in E.164, e.g. +14155551234
#   $4  VERIFIED_CALLER     — your +92 number Twilio verified at signup
#
# What it does (idempotent — safe to re-run):
#   1.  Creates a Twilio Elastic SIP trunk targeting LiveKit
#   2.  Adds the LiveKit SIP URI as the trunk's origination URI
#   3.  Associates your trial phone number with the trunk
#   4.  Cleans up the stale LiveKit SIP trunk + dispatch rule from the
#       previous tenant (so dispatch lands on hostel agent only)
#   5.  Creates a LiveKit inbound SIP trunk for that phone number
#   6.  Creates a LiveKit dispatch rule targeting hashim-girls-hostel-agent
#   7.  Updates configs/tenants/hashim-girls-hostel.yaml with the number
#   8.  Restarts the agent
#
# Requires: gh / curl / lk (LiveKit CLI), all already installed.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <TWILIO_ACCOUNT_SID> <TWILIO_AUTH_TOKEN> <TWILIO_PHONE_E164> <VERIFIED_CALLER_E164>" >&2
    exit 2
fi

TWILIO_SID="$1"
TWILIO_TOKEN="$2"
TWILIO_NUMBER="$3"
VERIFIED_CALLER="$4"

REPO_DIR="${REPO_DIR:-$(pwd)}"
: "${LIVEKIT_URL:?Set LIVEKIT_URL in the environment}"
: "${LIVEKIT_API_KEY:?Set LIVEKIT_API_KEY in the environment}"
: "${LIVEKIT_API_SECRET:?Set LIVEKIT_API_SECRET in the environment}"
: "${LIVEKIT_SIP_HOST:?Set LIVEKIT_SIP_HOST in the environment}"
AGENT_NAME="hashim-girls-hostel-agent"
TRUNK_FRIENDLY_NAME="HashimGirlsHostel"
TRUNK_DOMAIN="hashim-girls-hostel-$(echo "$TWILIO_SID" | tr 'A-Z' 'a-z' | tail -c 9).pstn.twilio.com"

export PATH="/opt/homebrew/bin:$PATH"
export LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET

cyan() { printf "\033[1;36m%s\033[0m\n" "$*"; }
red()  { printf "\033[1;31m%s\033[0m\n" "$*" >&2; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }

twilio_api() {
    # twilio_api METHOD PATH [data-fields...]
    local method="$1" path="$2"
    shift 2
    local args=(-sS -u "${TWILIO_SID}:${TWILIO_TOKEN}" -X "$method")
    for f in "$@"; do args+=(--data-urlencode "$f"); done
    curl "${args[@]}" "https://${path}"
}

# 0. Sanity-check the Twilio credentials.
cyan "[0/8] Verifying Twilio credentials..."
auth_check=$(twilio_api GET "api.twilio.com/2010-04-01/Accounts/${TWILIO_SID}.json")
if echo "$auth_check" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='active' else 1)" 2>/dev/null; then
    ok "Twilio account active"
else
    red "Twilio auth failed. Response:"
    echo "$auth_check" >&2
    exit 1
fi

# 1. Create the Twilio SIP trunk.
cyan "[1/8] Creating Twilio Elastic SIP trunk..."
trunk_json=$(twilio_api POST "trunking.twilio.com/v1/Trunks" \
    "FriendlyName=${TRUNK_FRIENDLY_NAME}" \
    "DomainName=${TRUNK_DOMAIN}")
TWILIO_TRUNK_SID=$(echo "$trunk_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['sid'])")
ok "Twilio trunk SID: $TWILIO_TRUNK_SID  (domain: $TRUNK_DOMAIN)"

# 2. Add LiveKit SIP origination URI to the trunk (this routes inbound calls).
cyan "[2/8] Adding LiveKit origination URI to Twilio trunk..."
twilio_api POST "trunking.twilio.com/v1/Trunks/${TWILIO_TRUNK_SID}/OriginationUrls" \
    "FriendlyName=LiveKit SIP URI" \
    "SipUrl=sip:${LIVEKIT_SIP_HOST};transport=tcp" \
    "Weight=1" "Priority=1" "Enabled=true" > /dev/null
ok "LiveKit URI attached (sip:${LIVEKIT_SIP_HOST};transport=tcp)"

# 3. Find the phone number SID and associate it with the trunk.
cyan "[3/8] Associating phone number ${TWILIO_NUMBER} with the trunk..."
PN_SID=$(twilio_api GET "api.twilio.com/2010-04-01/Accounts/${TWILIO_SID}/IncomingPhoneNumbers.json?PhoneNumber=${TWILIO_NUMBER}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['incoming_phone_numbers'][0]['sid']) if r['incoming_phone_numbers'] else sys.exit('phone not found')")
twilio_api POST "trunking.twilio.com/v1/Trunks/${TWILIO_TRUNK_SID}/PhoneNumbers" \
    "PhoneNumberSid=${PN_SID}" > /dev/null
ok "Phone ${TWILIO_NUMBER} routed through trunk"

# 4. Clean up the stale LiveKit trunk + dispatch rule from the previous tenant.
cyan "[4/8] Cleaning up stale LiveKit trunks / dispatch rules..."
for sdr in $(lk sip dispatch list -o json 2>/dev/null | python3 -c "import sys,json;[print(d['sip_dispatch_rule_id']) for d in json.load(sys.stdin).get('items',[])]" 2>/dev/null || true); do
    lk sip dispatch delete "$sdr" 2>/dev/null && ok "deleted dispatch $sdr" || true
done
for st in $(lk sip inbound list -o json 2>/dev/null | python3 -c "import sys,json;[print(t['sip_trunk_id']) for t in json.load(sys.stdin).get('items',[])]" 2>/dev/null || true); do
    lk sip inbound delete "$st" 2>/dev/null && ok "deleted inbound trunk $st" || true
done

# 5. Create the LiveKit inbound trunk for the new Twilio number.
cyan "[5/8] Creating LiveKit inbound SIP trunk..."
cat > /tmp/lk_inbound.json <<JSON
{
  "name": "Hashim Girls Hostel — Twilio inbound",
  "numbers": ["${TWILIO_NUMBER}"]
}
JSON
INBOUND_OUT=$(lk sip inbound create /tmp/lk_inbound.json 2>&1)
echo "$INBOUND_OUT"
INBOUND_SID=$(echo "$INBOUND_OUT" | grep -oE 'ST_[a-zA-Z0-9]+' | head -1)
ok "LiveKit inbound trunk: $INBOUND_SID"

# 6. Create the LiveKit dispatch rule targeting the hostel agent.
cyan "[6/8] Creating LiveKit dispatch rule for ${AGENT_NAME}..."
cat > /tmp/lk_dispatch.json <<JSON
{
  "name": "Hashim Girls Hostel dispatch",
  "trunk_ids": ["${INBOUND_SID}"],
  "rule": {
    "dispatchRuleIndividual": {
      "roomPrefix": "hostel-call-"
    }
  },
  "room_config": {
    "agents": [
      { "agent_name": "${AGENT_NAME}" }
    ]
  }
}
JSON
DISPATCH_OUT=$(lk sip dispatch create /tmp/lk_dispatch.json 2>&1)
echo "$DISPATCH_OUT"
ok "Dispatch rule created"

# 7. Update the hostel tenant YAML with the Twilio number.
cyan "[7/8] Updating tenant YAML with phone number..."
TENANT_YAML="${REPO_DIR}/configs/tenants/hashim-girls-hostel.yaml"
python3 - <<PY
import re, pathlib
p = pathlib.Path("${TENANT_YAML}")
text = p.read_text()
text = re.sub(r"phone_numbers:\s*\[\s*\]", 'phone_numbers:\n    - "${TWILIO_NUMBER}"', text)
p.write_text(text)
print("Updated phone_numbers in", p)
PY
ok "Tenant YAML updated"

# 8. Restart the agent if it's running.
cyan "[8/8] Restarting agent..."
if PID=$(lsof -ti :8088 -P -n 2>/dev/null | head -1) && [ -n "$PID" ]; then
    kill "$PID"; sleep 4
    ok "killed old agent PID $PID"
fi
ok "Done. Bring the agent back up with:"
echo
echo "    cd \"$REPO_DIR\" && docker compose up qdrant -d"
echo "    PYTHONPATH=src .venv/bin/python scripts/populate_qdrant.py \\"
echo "        --source knowledge/hashim-girls-hostel/faq_source.json \\"
echo "        --collection hashim_girls_hostel_faq"
echo "    PYTHONPATH=src nohup .venv/bin/python src/main.py start > /tmp/agent.log 2>&1 &"
echo
echo "Then dial ${TWILIO_NUMBER} from your verified number ${VERIFIED_CALLER}."
echo "Mom should call from her +92 PTCL / Jazz / Zong — standard intl rates apply."
