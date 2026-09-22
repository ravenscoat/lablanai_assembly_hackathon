#!/usr/bin/env python3
"""
Mock backend server for testing agent ↔ backend API integration.

Implements all 8 endpoints from agent-backend-api.md with in-memory storage.
Run this instead of the real backend to test the agent locally.

Usage:
    python scripts/mock_backend.py          # starts on port 3000
    python scripts/mock_backend.py 4000     # starts on port 4000

Then in .env:
    PLATFORM_API_URL=http://localhost:3000
    INTERNAL_API_KEY=test-key
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# In-memory storage
calls: dict[str, dict] = {}
murojatlar: list[dict] = []
caller_histories: dict[str, dict] = {}  # phone -> history

# Seed some caller history for testing
caller_histories["+92301234567"] = {
    "total_calls": 3,
    "last_topic": "business hours",
    "last_summary": "User asked about business hours",
    "last_call_status": "completed",
}


class MockBackendHandler(BaseHTTPRequestHandler):

    def _json_response(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _check_auth(self) -> bool:
        api_key = self.headers.get("X-API-Key", "")
        if not api_key:
            self._json_response(401, {"error": "Missing X-API-Key"})
            return False
        return True

    # ---- Routes ----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # GET /health
        if path == "/health":
            self._json_response(200, {"status": "ok"})
            return

        # GET /api/v1/internal/config
        if path == "/api/v1/internal/config":
            if not self._check_auth():
                return
            phone = params.get("phone", [""])[0]
            print(f"  [config] phone={phone}")
            # Return a minimal config — the agent uses YAML as primary source anyway
            self._json_response(200, {
                "success": True,
                "data": {
                    "tenant_id": "example-tenant",
                    "tenant_slug": "example-tenant",
                    "name": "Example Tenant",
                    "config": {},
                },
            })
            return

        # GET /api/v1/internal/caller-history
        if path == "/api/v1/internal/caller-history":
            if not self._check_auth():
                return
            phone = params.get("phone", [""])[0]
            tenant_id = params.get("tenant_id", [""])[0]
            print(f"  [caller-history] phone={phone} tenant={tenant_id}")

            history = caller_histories.get(phone)
            if history:
                self._json_response(200, {"success": True, "data": history})
            else:
                self._json_response(404, {"success": False, "error": "No history"})
            return

        # GET /api/v1/internal/calls/:id/operator/status
        if "/operator/status" in path:
            if not self._check_auth():
                return
            call_id = path.split("/calls/")[1].split("/operator")[0]
            call = calls.get(call_id, {})
            status = call.get("operator_status", "in_queue")
            print(f"  [operator-status] call={call_id} status={status}")
            self._json_response(200, {
                "success": True,
                "data": {
                    "status": status,
                    "operator_id": call.get("assigned_operator_id"),
                    "operator_name": call.get("operator_name"),
                },
            })
            return

        self._json_response(404, {"error": f"Not found: {path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # POST /api/v1/internal/calls
        if path == "/api/v1/internal/calls":
            if not self._check_auth():
                return
            body = self._read_body()
            call_db_id = str(uuid.uuid4())[:8]
            calls[call_db_id] = {
                **body,
                "id": call_db_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  [create-call] id={call_db_id} tenant={body.get('tenant_id')} "
                  f"caller={body.get('caller_phone')} agent={body.get('agent_phone')}")
            self._json_response(201, {"success": True, "data": {"id": call_db_id}})
            return

        # POST /api/v1/internal/calls/:id/operator
        if "/operator" in path and "/status" not in path and "/calls/" in path:
            if not self._check_auth():
                return
            call_id = path.split("/calls/")[1].split("/operator")[0]
            body = self._read_body()
            if call_id in calls:
                calls[call_id]["operator_status"] = "in_queue"
                calls[call_id]["transfer_reason"] = body.get("transfer_reason", "")
                calls[call_id]["ai_summary"] = body.get("ai_summary", "")
                # Simulate: operator accepts after 6 seconds (3 poll cycles)
                import threading
                def accept_operator():
                    import time
                    time.sleep(2)
                    if call_id in calls:
                        calls[call_id]["operator_status"] = "ringing"
                        print(f"  [operator] call={call_id} -> ringing")
                    time.sleep(4)
                    if call_id in calls:
                        calls[call_id]["operator_status"] = "transferred"
                        calls[call_id]["assigned_operator_id"] = "op-001"
                        calls[call_id]["operator_name"] = "Jasur"
                        print(f"  [operator] call={call_id} -> transferred (Jasur)")
                threading.Thread(target=accept_operator, daemon=True).start()

                print(f"  [request-operator] call={call_id} reason={body.get('transfer_reason')}")
                self._json_response(200, {"success": True})
            else:
                print(f"  [request-operator] call={call_id} NOT FOUND")
                self._json_response(404, {"error": "Call not found"})
            return

        # POST /api/v1/murojatlar/voice-assistant
        if path == "/api/v1/murojatlar/voice-assistant":
            body = self._read_body()
            murojaat_id = str(uuid.uuid4())[:8]
            murojatlar.append({"id": murojaat_id, **body})
            print(f"  [murojaat] id={murojaat_id} tenant={body.get('tenant_id')} "
                  f"from={body.get('murojaat_qiluvchi', {}).get('full_name', '?')}")
            self._json_response(201, {"success": True, "data": {"id": murojaat_id}})
            return

        self._json_response(404, {"error": f"Not found: {path}"})

    def do_PATCH(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # PATCH /api/v1/internal/calls/:id
        if path.startswith("/api/v1/internal/calls/"):
            if not self._check_auth():
                return
            call_id = path.split("/calls/")[1].rstrip("/")
            body = self._read_body()
            if call_id in calls:
                calls[call_id].update(body)
                status = body.get("status", "?")
                duration = body.get("duration_seconds", "?")
                summary = body.get("ai_summary", "")[:50]
                print(f"  [update-call] id={call_id} status={status} "
                      f"duration={duration}s summary=\"{summary}...\"")
                self._json_response(200, {"success": True})
            else:
                print(f"  [update-call] id={call_id} NOT FOUND")
                self._json_response(404, {"error": "Call not found"})
            return

        self._json_response(404, {"error": f"Not found: {path}"})

    def log_message(self, format, *args):
        """Override to add emoji prefix for readability."""
        method = args[0].split()[0] if args else "?"
        path = args[0].split()[1] if args and len(args[0].split()) > 1 else "?"
        status = args[1] if len(args) > 1 else "?"
        print(f"[{method}] {path} -> {status}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    server = HTTPServer(("0.0.0.0", port), MockBackendHandler)
    print(f"Mock backend running on http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  GET  /health")
    print(f"  GET  /api/v1/internal/config?phone=...")
    print(f"  GET  /api/v1/internal/caller-history?phone=...&tenant_id=...")
    print(f"  POST /api/v1/internal/calls")
    print(f"  PATCH /api/v1/internal/calls/:id")
    print(f"  POST /api/v1/internal/calls/:id/operator")
    print(f"  GET  /api/v1/internal/calls/:id/operator/status")
    print(f"  POST /api/v1/murojatlar/voice-assistant")
    print(f"")
    print(f"Seeded caller history for +998901234567 (3 previous calls)")
    print(f"Operator handoff simulates: in_queue -> ringing (2s) -> transferred (6s)")
    print(f"---")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down mock backend")


if __name__ == "__main__":
    main()
