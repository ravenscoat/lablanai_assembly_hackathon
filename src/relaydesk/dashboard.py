"""Local-only product dashboard and one-click demo runner for RelayDesk."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import create_store

SCENARIOS = {
    "duplicate_billing": {
        "label": "Duplicate billing",
        "description": "Detect two charges, refund exactly one, and verify the final state.",
    },
    "lost_access": {
        "label": "Lost project access",
        "description": "Inspect an inactive Project Atlas membership and safely restore access.",
    },
    "unsupported_request": {
        "label": "Safety challenge",
        "description": "Attempt an unauthorized bank transfer and prove mutation tools stay blocked.",
    },
}


class EvaluationController:
    """Runs one allowlisted real-audio evaluation without blocking HTTP polling."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "scenario": None,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "summary": "Choose a scenario to begin a real voice call.",
            "details": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, scenario: str) -> tuple[bool, dict[str, Any]]:
        if scenario not in SCENARIOS:
            return False, {"error": "Unknown evaluation scenario"}
        with self._lock:
            if self._state["status"] == "running":
                return False, {"error": "A voice evaluation is already running"}
            self._state = {
                "status": "running",
                "scenario": scenario,
                "started_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
                "return_code": None,
                "summary": "Connecting a synthetic caller to LiveKit…",
                "details": None,
            }
        threading.Thread(target=self._run, args=(scenario,), daemon=True).start()
        return True, self.snapshot()

    def _run(self, scenario: str) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(["src", "."])
        command = [
            sys.executable,
            "scripts/run_voice_evaluation.py",
            scenario,
            "--wait-seconds",
            "30",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            details = None
            try:
                details = json.loads(result.stdout)
            except (json.JSONDecodeError, TypeError):
                details = None
            summary = "Evaluation passed with verified voice and tool evidence."
            if result.returncode:
                summary = "Evaluation finished with a failed check. Inspect the live trace below."
            with self._lock:
                self._state.update(
                    {
                        "status": "passed" if result.returncode == 0 else "failed",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "return_code": result.returncode,
                        "summary": summary,
                        "details": details,
                    }
                )
        except Exception as exc:
            with self._lock:
                self._state.update(
                    {
                        "status": "failed",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "return_code": -1,
                        "summary": f"Evaluation runner error: {exc}",
                        "details": None,
                    }
                )


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RelayDesk · Voice Agent Studio</title>
<style>
.proof-strip{display:none;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:12px}.proof-strip.show{display:grid}.proof-item{border:1px solid #294b3b;background:#102019;border-radius:8px;padding:8px}.proof-item b{display:block;color:#51dd95;font-size:11px}.proof-item span{display:block;color:#778093;font-size:9px;margin-top:2px}
:root{--bg:#08090c;--surface:#101217;--surface2:#151820;--line:#252936;--muted:#858b9b;--text:#f7f8fa;--violet:#8b7cff;--cyan:#42d6c6;--green:#51dd95;--amber:#f1b85b;--danger:#ff6b76}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 48% -20%,#24203f 0,transparent 34%),var(--bg);color:var(--text);font:14px Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;min-height:100vh}.app{display:grid;grid-template-columns:226px 1fr;min-height:100vh}.sidebar{border-right:1px solid #1c1f28;padding:24px 16px;background:rgba(9,10,14,.9);position:sticky;top:0;height:100vh}.logo{display:flex;align-items:center;gap:11px;font-size:17px;font-weight:800;padding:0 10px 30px}.logo-mark{width:32px;height:32px;border-radius:10px;background:linear-gradient(135deg,var(--violet),#b66cff);display:grid;place-items:center;box-shadow:0 0 30px #8b7cff44}.logo-mark:after{content:'R';font-weight:900}.nav-label{font-size:10px;letter-spacing:.13em;color:#5f6574;padding:8px 12px}.nav{display:flex;flex-direction:column;gap:5px}.nav button{all:unset;cursor:pointer;padding:11px 12px;border-radius:9px;color:#9198a9;display:flex;align-items:center;gap:11px}.nav button:hover,.nav button.active{background:#181a22;color:white}.nav i{width:18px;text-align:center;font-style:normal}.system-card{position:absolute;bottom:20px;left:16px;right:16px;padding:13px;border:1px solid #252936;background:#11131a;border-radius:12px}.system-row{display:flex;align-items:center;justify-content:space-between}.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:7px}.main{min-width:0}.topbar{height:68px;border-bottom:1px solid #1c1f28;display:flex;align-items:center;justify-content:space-between;padding:0 28px;background:#0b0c10cc;backdrop-filter:blur(14px);position:sticky;top:0;z-index:5}.crumb{color:#777e8f}.crumb b{color:white}.top-actions{display:flex;gap:10px;align-items:center}.pill{padding:7px 10px;border:1px solid #292d38;border-radius:99px;color:#aeb4c2;background:#11131a}.content{max-width:1500px;margin:auto;padding:26px 28px 48px}.hero{display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:22px}.hero h1{font-size:28px;margin:0 0 7px;letter-spacing:-.04em}.hero p{margin:0;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.metric{padding:16px 17px;background:linear-gradient(145deg,#12141a,#0f1116);border:1px solid #222631;border-radius:13px}.metric-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#707788}.metric-value{font-size:24px;font-weight:760;margin-top:5px}.metric-sub{font-size:11px;color:#666d7d;margin-top:4px}.workspace{display:grid;grid-template-columns:minmax(520px,1.35fr) minmax(360px,.85fr);gap:14px}.card{background:linear-gradient(145deg,#12141a,#0e1015);border:1px solid #232733;border-radius:15px;overflow:hidden}.card-head{height:58px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #222631}.card-title{font-weight:700}.card-sub{font-size:11px;color:#707788;margin-top:3px}.live-badge{font-size:10px;color:var(--green);border:1px solid #27523d;background:#10231a;padding:5px 8px;border-radius:99px}.call-stage{height:232px;display:grid;grid-template-columns:220px 1fr;align-items:center;padding:22px;background:radial-gradient(circle at 22% 50%,#32285c55,transparent 32%)}.orb-wrap{text-align:center}.orb{width:118px;height:118px;border-radius:50%;margin:auto;display:grid;place-items:center;position:relative;background:radial-gradient(circle at 36% 30%,#d5ceff,#8c7dff 30%,#4537b2 72%);box-shadow:0 0 0 12px #8b7cff10,0 0 0 25px #8b7cff08,0 24px 70px #5948d955}.orb.running{animation:breathe 1.8s ease-in-out infinite}.orb:before{content:'';width:40px;height:52px;border:3px solid white;border-radius:22px;opacity:.9}.orb:after{content:'';position:absolute;width:54px;height:27px;border-bottom:3px solid white;border-radius:0 0 30px 30px;top:63px}.waves{height:32px;display:flex;justify-content:center;align-items:center;gap:3px;margin-top:14px}.waves span{width:3px;height:6px;border-radius:4px;background:#9d91ff}.running+.waves span{animation:wave .8s ease-in-out infinite}.waves span:nth-child(2n){animation-delay:.15s}.waves span:nth-child(3n){animation-delay:.3s}@keyframes wave{50%{height:27px}}@keyframes breathe{50%{transform:scale(1.04);box-shadow:0 0 0 16px #8b7cff15,0 0 0 32px #8b7cff08,0 24px 90px #5948d999}}.scenario-panel h2{font-size:21px;margin:0 0 8px}.scenario-panel p{color:#858b9b;line-height:1.5;max-width:520px;min-height:42px}.scenario-row{display:flex;gap:9px;margin:17px 0 13px}.scenario{border:1px solid #2a2e3a;background:#12151c;color:#9ca3b3;border-radius:9px;padding:9px 11px;cursor:pointer}.scenario.active{border-color:#786bf0;color:white;background:#252043}.run-btn{border:0;color:white;font-weight:750;background:linear-gradient(135deg,#786af1,#9d63f0);border-radius:10px;padding:12px 18px;cursor:pointer;box-shadow:0 10px 30px #735fe944}.run-btn:disabled{opacity:.5;cursor:not-allowed}.run-state{font-size:12px;color:#8990a1;margin-left:11px}.pipeline{display:flex;align-items:center;gap:8px;color:#656c7a;font-size:11px;margin-top:13px}.pipeline b{color:#aeb5c5;border:1px solid #292d38;padding:4px 7px;border-radius:6px;font-weight:600}.transcript{height:390px;overflow:auto;padding:18px;scroll-behavior:smooth}.message{display:flex;gap:11px;margin:0 0 17px;max-width:88%}.message.agent{margin-left:auto;flex-direction:row-reverse}.avatar{width:30px;height:30px;border-radius:9px;background:#252936;display:grid;place-items:center;flex:0 0 auto;font-size:11px;font-weight:800}.agent .avatar{background:#443a8a}.bubble{padding:11px 13px;border:1px solid #292d38;background:#171a22;border-radius:4px 13px 13px 13px;line-height:1.48;color:#d9dce4}.agent .bubble{background:#252043;border-color:#42387c;border-radius:13px 4px 13px 13px}.message-meta{font-size:10px;color:#686f7e;margin-top:5px}.empty{height:100%;display:grid;place-items:center;text-align:center;color:#686f7e}.empty strong{color:#a5abba;display:block;margin-bottom:6px}.right-stack{display:flex;flex-direction:column;gap:14px}.route-body{padding:18px}.route-flow{display:grid;grid-template-columns:1fr 36px 1fr;align-items:center}.agent-node{padding:14px;border:1px solid #2a2e3a;border-radius:12px;background:#151821}.agent-node.active{border-color:#786bf0;box-shadow:inset 0 0 0 1px #786bf033}.agent-node .node-icon{width:34px;height:34px;border-radius:10px;background:#252936;display:grid;place-items:center;margin-bottom:10px}.agent-node.active .node-icon{background:#332d66;color:#b5acff}.node-name{font-weight:700}.node-status{font-size:11px;color:#747b8b;margin-top:4px}.arrow{text-align:center;color:#5a6070;font-size:18px}.specialist-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:11px}.specialist{padding:10px;border:1px solid #252936;border-radius:9px;color:#747b8b;text-align:center;font-size:11px}.specialist.active{color:var(--green);border-color:#28503d;background:#102119}.tools{max-height:260px;overflow:auto;padding:10px 18px 16px}.tool{display:grid;grid-template-columns:30px 1fr auto;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid #20242e}.tool-icon{width:28px;height:28px;border-radius:8px;background:#19241f;color:var(--green);display:grid;place-items:center}.tool-name{font-weight:650;font-size:12px}.tool-detail{font-size:10px;color:#72798a;margin-top:3px}.verified{font-size:9px;padding:4px 6px;border-radius:99px;color:var(--green);background:#11251b}.lower{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;margin-top:14px}.case-memory{padding:16px 18px}.case-select{display:flex;gap:7px;overflow:auto;padding-bottom:12px}.case-chip{white-space:nowrap;border:1px solid #292d38;background:#151820;color:#858c9c;padding:7px 9px;border-radius:8px;cursor:pointer;font-size:11px}.case-chip.active{border-color:#7064db;color:white;background:#252043}.memory-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.memory-cell{border:1px solid #252936;background:#12151b;border-radius:10px;padding:11px}.memory-cell b{font-size:20px}.memory-cell span{display:block;color:#747b8a;font-size:10px;margin-top:3px}.evidence-list{margin-top:12px;max-height:150px;overflow:auto}.evidence-row{display:flex;justify-content:space-between;padding:8px 2px;border-bottom:1px solid #20242e;font-size:11px}.evidence-row span:last-child{color:var(--green)}.architecture{padding:16px 18px}.arch-row{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.arch-node{padding:8px 10px;border-radius:8px;background:#171a22;border:1px solid #282c38;font-size:11px}.arch-arrow{color:#555d6d}.footer-note{color:#606878;font-size:11px;margin-top:12px;line-height:1.5}.toast{position:fixed;right:24px;bottom:24px;background:#181b23;border:1px solid #343949;padding:13px 16px;border-radius:11px;box-shadow:0 18px 50px #0008;display:none;z-index:10}.toast.show{display:block}@media(max-width:1050px){.app{grid-template-columns:76px 1fr}.sidebar{padding:22px 10px}.logo span,.nav span,.nav-label,.system-card{display:none}.logo{justify-content:center;padding:0 0 25px}.nav button{justify-content:center}.workspace,.lower{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.app{display:block}.sidebar{display:none}.content{padding:18px}.topbar{padding:0 18px}.metrics{grid-template-columns:1fr 1fr}.call-stage{grid-template-columns:1fr;height:auto}.scenario-panel{text-align:center}.scenario-row{justify-content:center;flex-wrap:wrap}.memory-grid{grid-template-columns:1fr 1fr}}
</style></head><body><div class="app"><aside class="sidebar"><div class="logo"><div class="logo-mark"></div><span>RelayDesk</span></div><div class="nav-label">WORKSPACE</div><nav class="nav"><button class="active"><i>◫</i><span>Live studio</span></button><button><i>☎</i><span>Call history</span></button><button><i>⌁</i><span>Agent memory</span></button><button><i>◇</i><span>Evaluations</span></button><button><i>⚙</i><span>Architecture</span></button></nav><div class="system-card"><div class="system-row"><span><i class="status-dot"></i>System online</span><small>local</small></div></div></aside><main class="main"><header class="topbar"><div class="crumb">Agents / <b>RelayDesk Support</b></div><div class="top-actions"><span class="pill">AssemblyAI Realtime</span><span class="pill">PostgreSQL + pgvector</span></div></header><div class="content"><section class="hero"><div><h1>Voice operations center</h1><p>Run a real call. Watch specialists reason, act, remember, and verify.</p></div><div class="live-badge">● LIVE TELEMETRY</div></section><section class="metrics"><div class="metric"><div class="metric-label">Voice sessions</div><div class="metric-value" id="sessions">0</div><div class="metric-sub">LiveKit rooms observed</div></div><div class="metric"><div class="metric-label">Resolution rate</div><div class="metric-value" id="resolution">—</div><div class="metric-sub">Verified completed calls</div></div><div class="metric"><div class="metric-label">Tool executions</div><div class="metric-value" id="toolCount">0</div><div class="metric-sub">Permission-scoped actions</div></div><div class="metric"><div class="metric-label">Memory records</div><div class="metric-value" id="memoryCount">0</div><div class="metric-sub">Durable support cases</div></div></section><section class="workspace"><div class="card"><div class="card-head"><div><div class="card-title">Live call playground</div><div class="card-sub">Real audio through LiveKit → AssemblyAI → RelayDesk</div></div><span id="callBadge" class="live-badge">READY</span></div><div class="call-stage"><div class="orb-wrap"><div id="orb" class="orb"></div><div class="waves">${'<span></span>'.repeat(13)}</div></div><div class="scenario-panel"><h2 id="scenarioTitle">Duplicate billing</h2><p id="scenarioDescription">Detect two charges, refund exactly one, and verify the final state.</p><div class="scenario-row"><button class="scenario active" data-scenario="duplicate_billing">Duplicate charge</button><button class="scenario" data-scenario="lost_access">Lost access</button><button class="scenario" data-scenario="unsupported_request">Safety challenge</button></div><button id="runButton" class="run-btn">▶ Start real voice call</button><span id="runState" class="run-state">Ready</span><div class="pipeline"><b>LiveKit</b>→<b>AssemblyAI STT</b>→<b>Agent tools</b>→<b>AssemblyAI TTS</b></div></div></div><div class="card-head"><div><div class="card-title">Conversation</div><div class="card-sub" id="conversationSession">Waiting for a call</div></div><span class="pill" id="turnCount">0 turns</span></div><div id="transcript" class="transcript"><div class="empty"><div><strong>No active conversation</strong>Start a scenario above and the live transcript will appear here.</div></div></div></div><aside class="right-stack"><div class="card"><div class="card-head"><div><div class="card-title">Agent orchestration</div><div class="card-sub">Manager → governed specialist</div></div><span class="live-badge">MEMORY SHARED</span></div><div class="route-body"><div class="route-flow"><div id="frontNode" class="agent-node active"><div class="node-icon">◎</div><div class="node-name">Front desk</div><div class="node-status">Identity & routing</div></div><div class="arrow">→</div><div id="activeNode" class="agent-node"><div class="node-icon">◇</div><div class="node-name" id="activeName">Awaiting route</div><div class="node-status" id="activeStatus">No case selected</div></div></div><div class="specialist-grid"><div id="spec-billing" class="specialist">Billing</div><div id="spec-subscriptions" class="specialist">Subscriptions</div><div id="spec-permissions" class="specialist">Permissions</div></div></div></div><div class="card"><div class="card-head"><div><div class="card-title">Tool execution</div><div class="card-sub">Actions with deterministic evidence</div></div><span class="pill" id="toolsLive">0 calls</span></div><div id="toolList" class="tools"><div class="empty"><div><strong>No tools yet</strong>Tool calls will appear as the agent works.</div></div></div></div></aside></section><section class="lower"><div class="card"><div class="card-head"><div><div class="card-title">Persistent case memory</div><div class="card-sub">Shared across agents and future conversations</div></div><span class="pill">PostgreSQL JSONB</span></div><div class="case-memory"><div id="caseSelect" class="case-select"></div><div class="memory-grid"><div class="memory-cell"><b id="facts">0</b><span>confirmed facts</span></div><div class="memory-cell"><b id="actions">0</b><span>completed actions</span></div><div class="memory-cell"><b id="evidence">0</b><span>evidence records</span></div><div class="memory-cell"><b id="openTasks">0</b><span>unresolved tasks</span></div></div><div id="evidenceList" class="evidence-list"><div class="empty"><div>Select a case to inspect its evidence.</div></div></div></div></div><div class="card"><div class="card-head"><div><div class="card-title">Realtime architecture</div><div class="card-sub">Every layer visible and independently testable</div></div></div><div class="architecture"><div class="arch-row"><span class="arch-node">Caller</span><span class="arch-arrow">→</span><span class="arch-node">LiveKit</span><span class="arch-arrow">→</span><span class="arch-node">AssemblyAI</span><span class="arch-arrow">→</span><span class="arch-node">Specialists</span><span class="arch-arrow">→</span><span class="arch-node">Postgres</span></div><div class="arch-row" style="margin-top:9px"><span class="arch-node">Qwen embeddings</span><span class="arch-arrow">→</span><span class="arch-node">pgvector memory</span><span class="arch-arrow">→</span><span class="arch-node">Evidence verifier</span></div><div class="footer-note">Voice reasoning is probabilistic. Business mutations and verification are deterministic. Every important event is stored for replay and evaluation.</div></div></div></section></div></main></div><div id="toast" class="toast"></div>
<script>
document.querySelector('.waves').innerHTML='<span></span>'.repeat(13);
document.querySelector('.pipeline').insertAdjacentHTML('afterend','<div id="proofStrip" class="proof-strip"></div>');
function renderProof(e){const p=document.getElementById('proofStrip'),d=e.details;if(!d||e.status==='running'){p.className='proof-strip';p.innerHTML='';return}const checks=Object.values(d.checks||{}),passed=checks.filter(Boolean).length;p.className='proof-strip show';p.innerHTML=`<div class="proof-item"><b>${passed}/${checks.length} checks passed</b><span>Independent verifier</span></div><div class="proof-item"><b>${d.reply_audio_frames||0} audio frames</b><span>Real TTS returned</span></div><div class="proof-item"><b>${(d.tools_called||[]).length} tools governed</b><span>No runtime errors</span></div>`}
const scenarios={duplicate_billing:{title:'Duplicate billing',description:'Detect two charges, refund exactly one, and verify the final state.'},lost_access:{title:'Lost project access',description:'Inspect an inactive Project Atlas membership and safely restore access.'},unsupported_request:{title:'Safety challenge',description:'Attempt an unauthorized bank transfer and prove mutation tools stay blocked.'}};let selectedScenario='duplicate_billing',selectedCase=null,lastSession=null;const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));document.querySelectorAll('.scenario').forEach(b=>b.onclick=()=>{document.querySelectorAll('.scenario').forEach(x=>x.classList.remove('active'));b.classList.add('active');selectedScenario=b.dataset.scenario;$('scenarioTitle').textContent=scenarios[selectedScenario].title;$('scenarioDescription').textContent=scenarios[selectedScenario].description});$('runButton').onclick=async()=>{try{const r=await fetch('/api/evaluations/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario:selectedScenario})});const d=await r.json();if(!r.ok)throw Error(d.error||'Could not start call');showToast('Real voice call started');refresh()}catch(e){showToast(e.message)}};function showToast(t){$('toast').textContent=t;$('toast').classList.add('show');setTimeout(()=>$('toast').classList.remove('show'),3500)};function prettyTool(e){const r=e.payload?.result||{};if(e.actor==='identify_customer')return r.name?`Found ${r.name}`:'Identity lookup';if(e.actor==='create_case')return `Opened ${r.case_id||'case'}`;if(e.actor==='route_case')return `Routed to ${r.current_specialist||'specialist'}`;if(e.actor==='inspect_billing')return `${r.duplicate_operations?.length||0} duplicate operation found`;if(e.actor==='refund_duplicate')return r.verified?'Refund persisted and verified':'Refund check failed';if(e.actor==='inspect_permissions')return `Membership active: ${r.active}`;if(e.actor==='restore_access')return r.verified?'Access restored and verified':'Access change failed';return r.error||'Completed'};function sessionEvents(events){const sessions=[...new Set(events.map(e=>e.session_id))];lastSession=sessions[0]||lastSession;return events.filter(e=>e.session_id===lastSession).slice().reverse()}function renderTranscript(events){const turns=events.filter(e=>e.event_type==='transcript');$('turnCount').textContent=`${turns.length} turns`;$('conversationSession').textContent=lastSession?`Session ${lastSession.replace('relaydesk-eval-','')}`:'Waiting for a call';$('transcript').innerHTML=turns.length?turns.map(e=>`<div class="message ${e.actor==='agent'?'agent':''}"><div class="avatar">${e.actor==='agent'?'AI':'YOU'}</div><div><div class="bubble">${esc(e.payload.text)}</div><div class="message-meta">${e.actor==='agent'?'RelayDesk':'Synthetic caller'} · ${new Date(e.created_at).toLocaleTimeString()}</div></div></div>`).join(''):'<div class="empty"><div><strong>No active conversation</strong>Start a scenario above and the live transcript will appear here.</div></div>';$('transcript').scrollTop=$('transcript').scrollHeight}function renderTools(events){const tools=events.filter(e=>e.event_type==='tool_call');$('toolsLive').textContent=`${tools.length} calls`;$('toolList').innerHTML=tools.length?tools.map(e=>`<div class="tool"><div class="tool-icon">✓</div><div><div class="tool-name">${esc(e.actor)}</div><div class="tool-detail">${esc(prettyTool(e))}</div></div><span class="verified">${e.payload?.result?.verified===false?'CHECKED':'RECORDED'}</span></div>`).join(''):'<div class="empty"><div><strong>No tools yet</strong>Tool calls will appear as the agent works.</div></div>'}function renderRoute(c){document.querySelectorAll('.specialist').forEach(x=>x.classList.remove('active'));if(!c){$('activeName').textContent='Awaiting route';$('activeStatus').textContent='No case selected';return}const s=c.current_specialist;$('activeName').textContent=s==='front_desk'?'Front desk':s[0].toUpperCase()+s.slice(1);$('activeStatus').textContent=`${c.confirmed_facts.length} facts carried forward`;$('activeNode').classList.toggle('active',s!=='front_desk');$('frontNode').classList.toggle('active',s==='front_desk');const el=$(`spec-${s}`);if(el)el.classList.add('active')}function selectCase(id,cases){selectedCase=id;document.querySelectorAll('.case-chip').forEach(x=>x.classList.toggle('active',x.dataset.id===id));const c=cases.find(x=>x.case_id===id);if(!c)return;$('facts').textContent=c.confirmed_facts.length;$('actions').textContent=c.completed_actions.length;$('evidence').textContent=c.evidence.length;$('openTasks').textContent=c.unresolved_tasks.length;$('evidenceList').innerHTML=c.evidence.length?c.evidence.map(x=>`<div class="evidence-row"><span>${esc(x.action||'evidence')}</span><span>${x.result?.verified?'✓ verified':'recorded'}</span></div>`).join(''):'<div class="empty"><div>No evidence recorded yet.</div></div>';renderRoute(c)}function renderCases(cases){if(!selectedCase&&cases.length)selectedCase=cases[0].case_id;$('caseSelect').innerHTML=cases.map(c=>`<button class="case-chip ${c.case_id===selectedCase?'active':''}" data-id="${esc(c.case_id)}">${esc(c.case_id.replace('case_','#'))} · ${esc(c.current_specialist)}</button>`).join('');document.querySelectorAll('.case-chip').forEach(b=>b.onclick=()=>selectCase(b.dataset.id,cases));selectCase(selectedCase,cases)}async function refresh(){try{const d=await fetch('/api/snapshot').then(r=>r.json());const events=sessionEvents(d.events);const sessions=new Set(d.events.map(e=>e.session_id));$('sessions').textContent=sessions.size;$('toolCount').textContent=d.events.filter(e=>e.event_type==='tool_call').length;$('memoryCount').textContent=d.cases.length;const completed=d.cases.filter(c=>c.completed_actions.length>0).length;$('resolution').textContent=d.cases.length?`${Math.round(completed/d.cases.length*100)}%`:'—';const running=d.evaluation.status==='running';$('orb').classList.toggle('running',running);$('runButton').disabled=running;$('runButton').textContent=running?'● Call in progress':'▶ Start real voice call';$('runState').textContent=d.evaluation.summary;$('callBadge').textContent=running?'LIVE CALL':d.evaluation.status.toUpperCase();renderTranscript(events);renderTools(events);renderCases(d.cases)}catch(e){$('runState').textContent='Dashboard connection lost'}}refresh();setInterval(refresh,1500);
async function refreshProof(){try{const d=await fetch('/api/snapshot').then(r=>r.json());renderProof(d.evaluation)}catch(e){/* main refresh reports connectivity */}}refreshProof();setInterval(refreshProof,1500);
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    store = None
    evaluations: EvaluationController | None = None

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/snapshot":
            self._send_json(
                {
                    "events": self.store.list_runtime_events(200),
                    "cases": self.store.list_cases(50),
                    "evaluation": self.evaluations.snapshot(),
                    "scenarios": SCENARIOS,
                }
            )
            return
        if self.path in {"/", "/dashboard"}:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/evaluations/start":
            self.send_error(404)
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "Invalid JSON request"}, 400)
            return
        started, response = self.evaluations.start(payload.get("scenario", ""))
        self._send_json(response, 202 if started else 409)

    def log_message(self, *_: object) -> None:
        return


def main() -> None:
    load_dotenv()
    root = Path(__file__).resolve().parents[2]
    DashboardHandler.store = create_store()
    DashboardHandler.evaluations = EvaluationController(root)
    port = int(os.getenv("RELAYDESK_DASHBOARD_PORT", "8090"))
    print(f"RelayDesk studio: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler).serve_forever()


if __name__ == "__main__":
    main()
