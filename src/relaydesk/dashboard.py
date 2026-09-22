"""Local-only live dashboard for RelayDesk voice-agent operations."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv

from . import create_store

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RelayDesk Command Center</title><style>
*{box-sizing:border-box}body{margin:0;background:#070b12;color:#f4f7fb;font:15px Inter,system-ui,sans-serif}.shell{max-width:1400px;margin:auto;padding:32px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px}.brand{font-size:25px;font-weight:800}.brand span{color:#7c9cff}.live{padding:8px 12px;border:1px solid #244936;border-radius:99px;color:#72e3a6;background:#0b2018}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:linear-gradient(145deg,#111827,#0c121e);border:1px solid #202b3c;border-radius:18px}.card{padding:20px}.label{color:#8190a7;font-size:12px;text-transform:uppercase;letter-spacing:.09em}.value{font-size:30px;font-weight:800;margin-top:8px}.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:16px;margin-top:16px}.panel{padding:22px;min-height:460px}h2{font-size:17px;margin:0 0 18px}.event{display:grid;grid-template-columns:12px 1fr auto;gap:12px;padding:14px 0;border-bottom:1px solid #1d2736}.dot{width:9px;height:9px;background:#7c9cff;border-radius:50%;margin-top:6px}.actor{font-weight:700}.detail{color:#9eabc0;margin-top:5px;white-space:pre-wrap}.time{color:#67758b;font-size:12px}.case{padding:16px;border:1px solid #263247;border-radius:14px;margin-bottom:12px}.route{color:#86a5ff;font-weight:700}.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.chip{font-size:11px;background:#182338;color:#adbee0;padding:5px 8px;border-radius:99px}.empty{color:#718096;padding:35px 0;text-align:center}@media(max-width:850px){.metrics{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.shell{padding:18px}}</style></head><body><main class="shell"><header class="top"><div><div class="brand">Relay<span>Desk</span> Command Center</div><div class="label" style="margin-top:6px">Live voice support orchestration</div></div><div class="live">● Live</div></header><section class="metrics"><div class="card"><div class="label">Sessions</div><div id="sessions" class="value">0</div></div><div class="card"><div class="label">Cases</div><div id="casesCount" class="value">0</div></div><div class="card"><div class="label">Tool calls</div><div id="tools" class="value">0</div></div><div class="card"><div class="label">Verified actions</div><div id="verified" class="value">0</div></div></section><section class="grid"><div class="panel"><h2>Live activity</h2><div id="events"></div></div><div class="panel"><h2>Shared case memory</h2><div id="cases"></div></div></section></main><script>
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));async function refresh(){const d=await fetch('/api/snapshot').then(r=>r.json());sessions.textContent=new Set(d.events.map(e=>e.session_id)).size;casesCount.textContent=d.cases.length;tools.textContent=d.events.filter(e=>e.event_type==='tool_call').length;verified.textContent=d.cases.reduce((n,c)=>n+c.completed_actions.length,0);events.innerHTML=d.events.length?d.events.map(e=>`<div class="event"><i class="dot"></i><div><div class="actor">${esc(e.actor)} · ${esc(e.event_type)}</div><div class="detail">${esc(e.payload.text||e.payload.result?.error||JSON.stringify(e.payload.result||''))}</div></div><div class="time">${new Date(e.created_at).toLocaleTimeString()}</div></div>`).join(''):'<div class="empty">Waiting for the first call…</div>';cases.innerHTML=d.cases.length?d.cases.map(c=>`<div class="case"><div><b>${esc(c.case_id)}</b></div><div class="route">${esc(c.current_specialist)} specialist</div><div class="chips"><span class="chip">${c.confirmed_facts.length} facts</span><span class="chip">${c.completed_actions.length} actions</span><span class="chip">${c.evidence.length} evidence</span><span class="chip">${c.unresolved_tasks.length} open</span></div></div>`).join(''):'<div class="empty">No cases yet</div>'}refresh();setInterval(refresh,2000);
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    store = None

    def do_GET(self) -> None:
        if self.path == "/api/snapshot":
            body = json.dumps(
                {"events": self.store.list_runtime_events(100), "cases": self.store.list_cases(50)},
                default=str,
            ).encode()
            content_type = "application/json"
        elif self.path in {"/", "/dashboard"}:
            body, content_type = HTML.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


def main() -> None:
    load_dotenv()
    DashboardHandler.store = create_store()
    port = int(os.getenv("RELAYDESK_DASHBOARD_PORT", "8090"))
    print(f"RelayDesk dashboard: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler).serve_forever()


if __name__ == "__main__":
    main()
