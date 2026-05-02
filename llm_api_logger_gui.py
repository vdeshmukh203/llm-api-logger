"""
llm_api_logger_gui — browser-based dashboard for llm-api-logger.

Starts a lightweight HTTP server (stdlib only, no external dependencies)
and opens a self-contained HTML dashboard in the default browser.

Usage
-----
From the command line::

    llm-api-logger gui [log_file] [--host 127.0.0.1] [--port 7823]

Or programmatically::

    from llm_api_logger_gui import serve
    serve("llm_api.jsonl", port=7823)
"""

import argparse
import json
import pathlib
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# HTML dashboard (self-contained, no CDN)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM API Logger Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#f0f4f8;color:#1a202c;min-height:100vh}
  header{background:#2d3748;color:#fff;padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.25rem;font-weight:700;letter-spacing:-.5px}
  header .badge{background:#4299e1;border-radius:9999px;font-size:.7rem;padding:.2rem .6rem;font-weight:600}
  .toolbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:.75rem 2rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}
  .toolbar label{font-size:.8rem;font-weight:600;color:#4a5568}
  .toolbar select,.toolbar input{border:1px solid #cbd5e0;border-radius:.375rem;padding:.3rem .6rem;font-size:.85rem;background:#fff}
  .toolbar button{background:#4299e1;color:#fff;border:none;border-radius:.375rem;padding:.35rem .9rem;font-size:.85rem;cursor:pointer;font-weight:600}
  .toolbar button:hover{background:#3182ce}
  .toolbar button.danger{background:#e53e3e}
  .toolbar button.danger:hover{background:#c53030}
  .toolbar .spacer{flex:1}
  #status{font-size:.75rem;color:#718096}
  main{padding:1.5rem 2rem;display:grid;gap:1.25rem}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}
  .card{background:#fff;border-radius:.5rem;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .card .label{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#718096;margin-bottom:.3rem}
  .card .value{font-size:1.6rem;font-weight:700;color:#2d3748}
  .card .sub{font-size:.8rem;color:#a0aec0;margin-top:.15rem}
  .section{background:#fff;border-radius:.5rem;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}
  .section-header{padding:.75rem 1.25rem;font-size:.9rem;font-weight:700;background:#f7fafc;border-bottom:1px solid #e2e8f0;color:#2d3748}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{text-align:left;padding:.6rem 1rem;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#718096;background:#f7fafc;border-bottom:2px solid #e2e8f0}
  td{padding:.55rem 1rem;border-bottom:1px solid #edf2f7;color:#4a5568;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#ebf8ff}
  .tag{display:inline-block;border-radius:9999px;padding:.1rem .55rem;font-size:.7rem;font-weight:700;white-space:nowrap}
  .tag-openai{background:#d1fae5;color:#065f46}
  .tag-anthropic{background:#ede9fe;color:#5b21b6}
  .tag-google{background:#fef3c7;color:#92400e}
  .tag-mistral{background:#fee2e2;color:#991b1b}
  .tag-unknown{background:#e2e8f0;color:#4a5568}
  .tag-ok{background:#d1fae5;color:#065f46}
  .tag-err{background:#fee2e2;color:#991b1b}
  .pill-bar{display:flex;gap:.4rem;flex-wrap:wrap;padding:.75rem 1.25rem}
  .pill{display:inline-flex;align-items:center;gap:.35rem;border-radius:9999px;padding:.3rem .75rem;font-size:.8rem;font-weight:600;cursor:pointer;border:2px solid transparent;transition:all .15s}
  .pill.selected{border-color:#4299e1;background:#ebf8ff;color:#2b6cb0}
  .pill:not(.selected){background:#edf2f7;color:#4a5568}
  .pill:hover:not(.selected){background:#e2e8f0}
  .empty{padding:2.5rem;text-align:center;color:#a0aec0;font-size:.9rem}
  .hash{font-family:monospace;font-size:.7rem;color:#a0aec0;cursor:copy}
  #modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center}
  #modal-overlay.open{display:flex}
  #modal{background:#fff;border-radius:.5rem;max-width:700px;width:95%;max-height:85vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.3)}
  #modal-header{padding:1rem 1.25rem;font-weight:700;font-size:1rem;border-bottom:1px solid #e2e8f0;display:flex;justify-content:space-between;align-items:center}
  #modal-close{cursor:pointer;font-size:1.25rem;color:#718096;background:none;border:none;line-height:1}
  #modal-body{padding:1.25rem}
  #modal-body pre{background:#f7fafc;border-radius:.375rem;padding:.75rem;font-size:.78rem;overflow-x:auto;white-space:pre-wrap;word-break:break-all;border:1px solid #e2e8f0}
  #modal-body .row{display:grid;grid-template-columns:130px 1fr;gap:.4rem .75rem;font-size:.85rem;margin-bottom:.8rem}
  #modal-body .row .k{font-weight:600;color:#4a5568}
  @media(max-width:640px){
    .toolbar{padding:.75rem 1rem}
    main{padding:1rem}
    td,th{padding:.5rem .6rem}
  }
</style>
</head>
<body>
<header>
  <h1>LLM API Logger</h1>
  <span class="badge" id="log-badge">—</span>
  <span style="flex:1"></span>
  <span id="status" style="color:#a0aec0;font-size:.8rem"></span>
</header>

<div class="toolbar">
  <label>Provider</label>
  <select id="f-provider">
    <option value="">All</option>
    <option>openai</option>
    <option>anthropic</option>
    <option>google</option>
    <option>mistral</option>
    <option>together</option>
    <option>cohere</option>
    <option>huggingface</option>
    <option>unknown</option>
  </select>
  <label>Model</label>
  <select id="f-model"><option value="">All</option></select>
  <label>Status</label>
  <select id="f-status">
    <option value="">All</option>
    <option value="200">200 OK</option>
    <option value="429">429 Rate limited</option>
    <option value="500">500 Error</option>
  </select>
  <label>Since</label>
  <input type="date" id="f-since">
  <button onclick="applyFilters()">Filter</button>
  <button class="danger" onclick="resetFilters()">Reset</button>
  <span class="spacer"></span>
  <button onclick="refresh()">&#8635; Refresh</button>
</div>

<main>
  <!-- Stats cards -->
  <div class="cards">
    <div class="card"><div class="label">Total Calls</div><div class="value" id="c-calls">—</div><div class="sub">API requests logged</div></div>
    <div class="card"><div class="label">Total Cost</div><div class="value" id="c-cost">—</div><div class="sub">Estimated USD</div></div>
    <div class="card"><div class="label">Input Tokens</div><div class="value" id="c-tin">—</div><div class="sub">Prompt tokens</div></div>
    <div class="card"><div class="label">Output Tokens</div><div class="value" id="c-tout">—</div><div class="sub">Completion tokens</div></div>
    <div class="card"><div class="label">Avg Latency</div><div class="value" id="c-lat">—</div><div class="sub">ms per request</div></div>
  </div>

  <!-- Model breakdown -->
  <div class="section">
    <div class="section-header">By Model</div>
    <div class="pill-bar" id="model-pills"></div>
  </div>

  <!-- Entries table -->
  <div class="section">
    <div class="section-header">Log Entries <span id="entry-count" style="font-weight:400;color:#a0aec0"></span></div>
    <div id="table-wrap">
      <div class="empty">Loading…</div>
    </div>
  </div>
</main>

<!-- Detail modal -->
<div id="modal-overlay" onclick="closeModal(event)">
  <div id="modal">
    <div id="modal-header">
      Entry Detail
      <button id="modal-close" onclick="closeModal()">&#10005;</button>
    </div>
    <div id="modal-body"></div>
  </div>
</div>

<script>
let _allEntries = [];
let _filters = {};

function fmt(n){
  if(n===null||n===undefined) return '—';
  if(typeof n==='number'){
    if(n>=1e6) return (n/1e6).toFixed(2)+'M';
    if(n>=1e3) return (n/1e3).toFixed(1)+'k';
    return n.toLocaleString();
  }
  return n;
}
function fmtCost(v){return v===0?'$0.00':'$'+v.toFixed(4)}
function fmtLat(v){return v?v.toFixed(1)+'ms':'—'}
function fmtTs(v){if(!v)return '—';try{return new Date(v).toLocaleString();}catch{return v;}}

function providerTag(p){
  const cls='tag tag-'+(p||'unknown');
  return `<span class="${cls}">${p||'?'}</span>`;
}
function statusTag(code){
  const ok=code>=200&&code<300;
  return `<span class="tag ${ok?'tag-ok':'tag-err'}">${code}</span>`;
}

async function fetchSummary(){
  const r=await fetch('/api/summary');
  return r.json();
}
async function fetchEntries(params){
  const q=new URLSearchParams(params).toString();
  const r=await fetch('/api/entries'+(q?'?'+q:''));
  return r.json();
}

function renderCards(s){
  document.getElementById('c-calls').textContent=s.total_calls||0;
  document.getElementById('c-cost').textContent=fmtCost(s.total_cost_usd||0);
  document.getElementById('c-tin').textContent=fmt(s.total_tokens_in||0);
  document.getElementById('c-tout').textContent=fmt(s.total_tokens_out||0);
  document.getElementById('c-lat').textContent=fmtLat(s.avg_latency_ms||0);
}

function renderPills(s){
  const container=document.getElementById('model-pills');
  const cbm=s.calls_by_model||{};
  const cdm=s.cost_by_model||{};
  if(!Object.keys(cbm).length){container.innerHTML='<span style="color:#a0aec0;font-size:.85rem;padding:.5rem">No data</span>';return;}
  container.innerHTML=Object.entries(cbm).sort((a,b)=>b[1]-a[1]).map(([m,n])=>{
    const cost=cdm[m]||0;
    const active=_filters.model===m?'selected':'';
    return `<span class="pill ${active}" onclick="toggleModel('${m}')">${m} <b>${n}</b> <span style="color:#a0aec0">${fmtCost(cost)}</span></span>`;
  }).join('');
}

function toggleModel(m){
  if(_filters.model===m){delete _filters.model;}else{_filters.model=m;}
  applyFilters();
}

function renderTable(entries){
  const wrap=document.getElementById('table-wrap');
  document.getElementById('entry-count').textContent=entries.length?`(${entries.length})`:'';
  if(!entries.length){wrap.innerHTML='<div class="empty">No entries match the current filters.</div>';return;}
  const rows=entries.slice(0,200).map((e,i)=>`
    <tr onclick="showDetail(${i})" style="cursor:pointer">
      <td>${fmtTs(e.timestamp)}</td>
      <td>${providerTag(e.provider)}</td>
      <td title="${e.model}">${e.model||'—'}</td>
      <td>${statusTag(e.status_code)}</td>
      <td>${fmtLat(e.latency_ms)}</td>
      <td>${fmt(e.tokens_in)}</td>
      <td>${fmt(e.tokens_out)}</td>
      <td>${fmtCost(e.cost_usd)}</td>
      <td><span class="hash" title="${e.content_hash||''}">${(e.content_hash||'').slice(0,8)||'—'}</span></td>
    </tr>`).join('');
  wrap.innerHTML=`<table>
    <thead><tr>
      <th>Timestamp</th><th>Provider</th><th>Model</th><th>Status</th>
      <th>Latency</th><th>In Tok</th><th>Out Tok</th><th>Cost</th><th>Hash</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
  if(entries.length>200){
    wrap.innerHTML+=`<div style="padding:.75rem 1.25rem;font-size:.8rem;color:#a0aec0">Showing 200 of ${entries.length} entries. Export to CSV for the full dataset.</div>`;
  }
  _allEntries=entries;
}

function showDetail(i){
  const e=_allEntries[i];
  if(!e)return;
  let reqBody='', respBody='';
  try{reqBody=JSON.stringify(JSON.parse(e.request_body),null,2);}catch{reqBody=e.request_body||'—';}
  try{respBody=JSON.stringify(JSON.parse(e.response_body),null,2);}catch{respBody=e.response_body||'—';}
  document.getElementById('modal-body').innerHTML=`
    <div class="row">
      <span class="k">ID</span><span style="font-family:monospace;font-size:.8rem">${e.id}</span>
      <span class="k">Timestamp</span><span>${fmtTs(e.timestamp)}</span>
      <span class="k">URL</span><span style="word-break:break-all;font-size:.8rem">${e.url}</span>
      <span class="k">Provider</span><span>${providerTag(e.provider)}</span>
      <span class="k">Model</span><span><b>${e.model||'—'}</b></span>
      <span class="k">Status</span><span>${statusTag(e.status_code)}</span>
      <span class="k">Latency</span><span>${fmtLat(e.latency_ms)}</span>
      <span class="k">Tokens In</span><span>${fmt(e.tokens_in)}</span>
      <span class="k">Tokens Out</span><span>${fmt(e.tokens_out)}</span>
      <span class="k">Cost (USD)</span><span>${fmtCost(e.cost_usd)}</span>
      <span class="k">SHA-256</span><span style="font-family:monospace;font-size:.75rem;word-break:break-all">${e.content_hash||'—'}</span>
      ${e.error?`<span class="k">Error</span><span style="color:#e53e3e">${e.error}</span>`:''}
    </div>
    <p style="font-weight:700;margin-bottom:.4rem;font-size:.85rem">Request Body</p>
    <pre>${escHtml(reqBody)}</pre>
    <p style="font-weight:700;margin:.75rem 0 .4rem;font-size:.85rem">Response Body</p>
    <pre>${escHtml(respBody)}</pre>`;
  document.getElementById('modal-overlay').classList.add('open');
}

function escHtml(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function closeModal(ev){
  if(!ev||ev.target===document.getElementById('modal-overlay')||ev.target===document.getElementById('modal-close')){
    document.getElementById('modal-overlay').classList.remove('open');
  }
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});

function buildModelOptions(entries){
  const models=[...new Set(entries.map(e=>e.model).filter(Boolean))].sort();
  const sel=document.getElementById('f-model');
  const cur=sel.value;
  sel.innerHTML='<option value="">All</option>'+models.map(m=>`<option${m===cur?' selected':''}>${m}</option>`).join('');
}

function applyFilters(){
  const provider=document.getElementById('f-provider').value;
  const model=document.getElementById('f-model').value||_filters.model;
  const status=document.getElementById('f-status').value;
  const since=document.getElementById('f-since').value;
  const params={};
  if(provider)params.provider=provider;
  if(model)params.model=model;
  if(status)params.status_code=status;
  if(since)params.since=since+'T00:00:00';
  _filters=params;
  fetchEntries(params).then(data=>{
    renderTable(data.entries||[]);
    renderPills(data.summary||{});
  });
}

function resetFilters(){
  _filters={};
  document.getElementById('f-provider').value='';
  document.getElementById('f-model').value='';
  document.getElementById('f-status').value='';
  document.getElementById('f-since').value='';
  refresh();
}

async function refresh(){
  document.getElementById('status').textContent='Refreshing…';
  try{
    const [summary,data]=await Promise.all([fetchSummary(),fetchEntries(_filters)]);
    renderCards(summary);
    buildModelOptions(data.entries||[]);
    renderTable(data.entries||[]);
    renderPills(summary);
    document.getElementById('log-badge').textContent=summary.total_calls+' calls';
    document.getElementById('status').textContent='Updated '+new Date().toLocaleTimeString();
  }catch(err){
    document.getElementById('status').textContent='Error: '+err.message;
  }
}

refresh();
setInterval(refresh,15000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

def _make_handler(log_file: str):
    """Return a request handler class bound to *log_file*."""

    class _Handler(BaseHTTPRequestHandler):
        _log_file = log_file

        def log_message(self, fmt, *args):  # suppress default access log
            pass

        def _send_json(self, data: object, code: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _load(self):
            """Load the current log file into an LLMLogger."""
            # Import here to avoid circular import when gui is a standalone script
            import llm_api_logger as lal
            return lal._load_log(self._log_file)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._send_html(_DASHBOARD_HTML)
                return

            if path == "/api/summary":
                try:
                    log = self._load()
                    self._send_json(log.summary())
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)
                return

            if path == "/api/entries":
                qs = parse_qs(parsed.query)
                params = {k: v[0] for k, v in qs.items() if v}
                try:
                    log = self._load()
                    entries = log.query(
                        model=params.get("model"),
                        provider=params.get("provider"),
                        status_code=int(params["status_code"]) if "status_code" in params else None,
                        since=params.get("since"),
                        until=params.get("until"),
                    )
                    self._send_json({
                        "entries": [e.to_dict() for e in entries],
                        "summary": log.summary(),
                    })
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)
                return

            self.send_error(404, "Not found")

    return _Handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serve(
    log_file: str = "llm_api.jsonl",
    host: str = "127.0.0.1",
    port: int = 7823,
    open_browser: bool = True,
) -> None:
    """Start the dashboard HTTP server and optionally open a browser tab.

    Parameters
    ----------
    log_file:
        Path to the JSONL or SQLite log file to visualise.
    host:
        Bind address (default ``"127.0.0.1"``).
    port:
        TCP port (default ``7823``).
    open_browser:
        When ``True`` (default), open a browser tab after a short delay.
    """
    handler = _make_handler(log_file)
    server = HTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"LLM API Logger dashboard → {url}")
    print(f"Log file: {pathlib.Path(log_file).resolve()}")
    print("Press Ctrl-C to stop.")

    if open_browser:
        def _open():
            time.sleep(0.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Standalone entry point: ``llm-api-logger-gui``."""
    parser = argparse.ArgumentParser(
        prog="llm-api-logger-gui",
        description="Web dashboard for llm-api-logger log files.",
    )
    parser.add_argument(
        "log_file", nargs="?", default="llm_api.jsonl",
        help="JSONL or SQLite log file to visualise (default: llm_api.jsonl)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=7823, help="TCP port")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")
    args = parser.parse_args()
    serve(args.log_file, host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
