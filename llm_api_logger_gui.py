"""
Browser-based dashboard for LLM API Logger.

Uses only the Python standard library (``http.server``, ``json``,
``webbrowser``).  No external dependencies required.

Launch via::

    llm-api-logger gui [log_file]

or directly::

    python llm_api_logger_gui.py [log_file]

The dashboard opens automatically in your default web browser.
Press Ctrl-C (or close the terminal) to stop the server.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Ensure the root module is importable when running standalone
# ---------------------------------------------------------------------------
_root = str(pathlib.Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import llm_api_logger as _lal

# ---------------------------------------------------------------------------
# HTML template (self-contained single-page app)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM API Logger – Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5; color: #222; font-size: 14px; }
  header { background: #1a1a2e; color: #eee; padding: 12px 20px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; letter-spacing: .5px; }
  header span { font-size: 12px; opacity: .7; }
  .toolbar { background: #fff; border-bottom: 1px solid #ddd; padding: 8px 16px;
             display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .toolbar label { font-size: 12px; color: #555; }
  .toolbar select, .toolbar input { padding: 4px 8px; border: 1px solid #bbb;
                                    border-radius: 4px; font-size: 13px; }
  .toolbar button { padding: 5px 14px; border: none; border-radius: 4px; cursor: pointer;
                    font-size: 13px; background: #1a1a2e; color: #fff; }
  .toolbar button:hover { background: #333; }
  .toolbar button.secondary { background: #e9ecef; color: #333; border: 1px solid #ccc; }
  .toolbar button.secondary:hover { background: #dee2e6; }
  .summary { display: flex; gap: 12px; padding: 12px 16px; flex-wrap: wrap; }
  .card { background: #fff; border-radius: 8px; padding: 12px 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 130px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #888; }
  .card .value { font-size: 22px; font-weight: 700; color: #1a1a2e; margin-top: 2px; }
  .content { padding: 0 16px 16px; }
  table { width: 100%; border-collapse: collapse; background: #fff;
          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  thead th { background: #1a1a2e; color: #fff; text-align: left; padding: 9px 10px;
             font-size: 12px; letter-spacing: .3px; white-space: nowrap; cursor: pointer; }
  thead th:hover { background: #2d2d4e; }
  tbody tr { border-bottom: 1px solid #f0f0f0; cursor: pointer; }
  tbody tr:hover { background: #f8f9fa; }
  tbody tr.error-row { background: #fff5f5; }
  tbody tr.selected { background: #e8eaf6; }
  td { padding: 7px 10px; font-size: 13px; white-space: nowrap; max-width: 220px;
       overflow: hidden; text-overflow: ellipsis; }
  td.num { text-align: right; font-family: monospace; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 10px;
           font-size: 11px; font-weight: 600; }
  .badge-ok  { background: #d4edda; color: #155724; }
  .badge-err { background: #f8d7da; color: #721c24; }
  .badge-unk { background: #e9ecef; color: #495057; }
  .detail { margin-top: 12px; background: #fff; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); overflow: hidden; }
  .detail-tabs { display: flex; border-bottom: 2px solid #eee; }
  .tab-btn { padding: 9px 20px; cursor: pointer; font-size: 13px; border: none;
             background: none; color: #666; border-bottom: 2px solid transparent;
             margin-bottom: -2px; }
  .tab-btn.active { color: #1a1a2e; border-bottom-color: #1a1a2e; font-weight: 600; }
  .tab-content { display: none; padding: 12px; }
  .tab-content.active { display: block; }
  pre { font-size: 12px; white-space: pre-wrap; word-break: break-all;
        background: #f8f9fa; padding: 10px; border-radius: 4px; max-height: 280px;
        overflow-y: auto; }
  .meta-grid { display: grid; grid-template-columns: 140px 1fr; gap: 4px 12px; font-size: 13px; }
  .meta-grid .k { color: #666; }
  .meta-grid .v { font-family: monospace; }
  .status-bar { padding: 6px 16px; font-size: 12px; color: #666;
                border-top: 1px solid #ddd; background: #fff; }
  #no-entry { color: #aaa; padding: 20px; font-style: italic; }
  @media (max-width: 700px) { .summary { gap: 8px; } .card { min-width: 100px; } }
</style>
</head>
<body>
<header>
  <h1>LLM API Logger</h1>
  <span id="hdr-file">No file loaded</span>
</header>

<div class="toolbar">
  <label>Model
    <select id="f-model"><option value="">(all)</option></select>
  </label>
  <label>Provider
    <select id="f-provider"><option value="">(all)</option></select>
  </label>
  <label>Status
    <select id="f-status"><option value="">(all)</option></select>
  </label>
  <label>Search
    <input id="f-search" type="text" placeholder="URL / model / error…" style="width:180px">
  </label>
  <button onclick="applyFilters()">Apply</button>
  <button class="secondary" onclick="clearFilters()">Clear</button>
  <button class="secondary" onclick="refreshData()">&#8635; Refresh</button>
  <button class="secondary" onclick="exportCSV()">Export CSV</button>
</div>

<div class="summary">
  <div class="card"><div class="label">Total calls</div><div class="value" id="s-calls">—</div></div>
  <div class="card"><div class="label">Total cost</div><div class="value" id="s-cost">—</div></div>
  <div class="card"><div class="label">Input tokens</div><div class="value" id="s-tokin">—</div></div>
  <div class="card"><div class="label">Output tokens</div><div class="value" id="s-tokout">—</div></div>
  <div class="card"><div class="label">Avg latency</div><div class="value" id="s-lat">—</div></div>
</div>

<div class="content">
  <table id="main-table">
    <thead>
      <tr>
        <th onclick="sortBy('timestamp')">Timestamp &#8597;</th>
        <th onclick="sortBy('provider')">Provider &#8597;</th>
        <th onclick="sortBy('model')">Model &#8597;</th>
        <th onclick="sortBy('status_code')">Status &#8597;</th>
        <th onclick="sortBy('latency_ms')">Lat (ms) &#8597;</th>
        <th onclick="sortBy('tokens_in')">Tok In &#8597;</th>
        <th onclick="sortBy('tokens_out')">Tok Out &#8597;</th>
        <th onclick="sortBy('cost_usd')">Cost USD &#8597;</th>
        <th>Error</th>
      </tr>
    </thead>
    <tbody id="tbl-body"></tbody>
  </table>

  <div class="detail" id="detail-panel">
    <div id="no-entry">Click a row to inspect the request and response.</div>
    <div id="entry-detail" style="display:none">
      <div class="detail-tabs">
        <button class="tab-btn active" onclick="showTab('meta')">Metadata</button>
        <button class="tab-btn" onclick="showTab('req')">Request body</button>
        <button class="tab-btn" onclick="showTab('resp')">Response body</button>
      </div>
      <div id="tab-meta" class="tab-content active"><div class="meta-grid" id="meta-grid"></div></div>
      <div id="tab-req"  class="tab-content"><pre id="req-pre"></pre></div>
      <div id="tab-resp" class="tab-content"><pre id="resp-pre"></pre></div>
    </div>
  </div>
</div>
<div class="status-bar" id="status-bar">Loading…</div>

<script>
let allEntries = [];
let displayed  = [];
let sortCol    = 'timestamp';
let sortAsc    = false;
let selected   = null;

function fmt(n, decimals) {
  if (n === null || n === undefined) return '—';
  return Number(n).toFixed(decimals ?? 0);
}
function fmtK(n) { return n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n); }

async function fetchEntries() {
  const r = await fetch('/api/entries');
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function refreshData() {
  try {
    status('Loading…');
    allEntries = await fetchEntries();
    populateDropdowns();
    applyFilters();
    updateSummary(allEntries);
    document.getElementById('hdr-file').textContent =
      allEntries.length + ' entries';
    status('Loaded ' + allEntries.length + ' entries');
  } catch(e) { status('Error: ' + e.message); }
}

function populateDropdowns() {
  const models    = [...new Set(allEntries.map(e=>e.model))].sort();
  const providers = [...new Set(allEntries.map(e=>e.provider))].sort();
  const statuses  = [...new Set(allEntries.map(e=>String(e.status_code)))].sort();
  fillSelect('f-model',    models);
  fillSelect('f-provider', providers);
  fillSelect('f-status',   statuses);
}

function fillSelect(id, values) {
  const sel = document.getElementById(id);
  const prev = sel.value;
  sel.innerHTML = '<option value="">(all)</option>';
  values.forEach(v => {
    const o = document.createElement('option');
    o.value = o.textContent = v;
    if (v === prev) o.selected = true;
    sel.appendChild(o);
  });
}

function applyFilters() {
  const model    = document.getElementById('f-model').value;
  const provider = document.getElementById('f-provider').value;
  const status   = document.getElementById('f-status').value;
  const search   = document.getElementById('f-search').value.trim().toLowerCase();

  displayed = allEntries.filter(e => {
    if (model    && e.model !== model)            return false;
    if (provider && e.provider !== provider)       return false;
    if (status   && String(e.status_code)!==status) return false;
    if (search) {
      const haystack = [e.model,e.url,e.error||'',e.provider].join(' ').toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
  sortEntries();
  renderTable();
  status('Showing ' + displayed.length + ' of ' + allEntries.length + ' entries');
}

function clearFilters() {
  ['f-model','f-provider','f-status'].forEach(id => document.getElementById(id).value='');
  document.getElementById('f-search').value = '';
  applyFilters();
}

function sortBy(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = false; }
  sortEntries();
  renderTable();
}

function sortEntries() {
  displayed.sort((a,b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (va === null) va = '';
    if (vb === null) vb = '';
    if (typeof va === 'number') return sortAsc ? va-vb : vb-va;
    return sortAsc ? String(va).localeCompare(String(vb))
                   : String(vb).localeCompare(String(va));
  });
}

function badge(code) {
  const cls = code >= 200 && code < 300 ? 'badge-ok'
            : code >= 400              ? 'badge-err' : 'badge-unk';
  return `<span class="badge ${cls}">${code}</span>`;
}

function renderTable() {
  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = '';
  if (!displayed.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:20px;color:#aaa">No entries match the current filter</td></tr>';
    return;
  }
  displayed.forEach(e => {
    const tr = document.createElement('tr');
    if (e.error) tr.classList.add('error-row');
    if (e.id === selected) tr.classList.add('selected');
    tr.innerHTML = `
      <td>${(e.timestamp||'').substring(0,26)}</td>
      <td>${e.provider||''}</td>
      <td title="${e.model||''}">${(e.model||'').substring(0,28)}</td>
      <td>${badge(e.status_code)}</td>
      <td class="num">${fmt(e.latency_ms,1)}</td>
      <td class="num">${fmtK(e.tokens_in)}</td>
      <td class="num">${fmtK(e.tokens_out)}</td>
      <td class="num">$${fmt(e.cost_usd,6)}</td>
      <td style="color:#c00">${(e.error||'').substring(0,50)}</td>
    `;
    tr.addEventListener('click', () => selectEntry(e));
    tbody.appendChild(tr);
  });
}

function selectEntry(e) {
  selected = e.id;
  renderTable();
  showDetail(e);
}

function showDetail(e) {
  document.getElementById('no-entry').style.display = 'none';
  document.getElementById('entry-detail').style.display = 'block';

  const pretty = s => {
    if (!s) return '(empty)';
    try { return JSON.stringify(JSON.parse(s), null, 2); } catch(_) { return s; }
  };
  document.getElementById('req-pre').textContent  = pretty(e.request_body);
  document.getElementById('resp-pre').textContent = pretty(e.response_body);

  const fields = [
    ['ID', e.id], ['Timestamp', e.timestamp], ['URL', e.url],
    ['Method', e.method], ['Provider', e.provider], ['Model', e.model],
    ['Status', e.status_code], ['Latency ms', fmt(e.latency_ms,2)],
    ['Tokens in', e.tokens_in.toLocaleString()],
    ['Tokens out', e.tokens_out.toLocaleString()],
    ['Cost USD', '$'+fmt(e.cost_usd,6)],
  ];
  if (e.error) fields.push(['Error', e.error]);
  document.getElementById('meta-grid').innerHTML =
    fields.map(([k,v]) => `<span class="k">${k}</span><span class="v">${v??'—'}</span>`).join('');
}

function showTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b,i) => {
    const names = ['meta','req','resp'];
    b.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
}

function updateSummary(entries) {
  let cost=0, tin=0, tout=0, lat=0;
  entries.forEach(e => { cost+=e.cost_usd; tin+=e.tokens_in; tout+=e.tokens_out; lat+=e.latency_ms; });
  const n = entries.length;
  document.getElementById('s-calls').textContent  = n.toLocaleString();
  document.getElementById('s-cost').textContent   = '$'+cost.toFixed(4);
  document.getElementById('s-tokin').textContent  = fmtK(tin);
  document.getElementById('s-tokout').textContent = fmtK(tout);
  document.getElementById('s-lat').textContent    = (n ? (lat/n).toFixed(1) : 0)+' ms';
}

async function exportCSV() {
  const r = await fetch('/api/export_csv');
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'llm_api_log.csv';
  a.click();
}

function status(msg) { document.getElementById('status-bar').textContent = msg; }

document.getElementById('f-search').addEventListener('keydown', e => {
  if (e.key === 'Enter') applyFilters();
});

refreshData();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

def _make_handler(entries: List[_lal.LogEntry], logger: _lal.LLMLogger):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silence request logs in the terminal

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                self._respond(200, "text/html", _HTML.encode())
            elif path == "/api/entries":
                data = json.dumps([e.to_dict() for e in entries]).encode()
                self._respond(200, "application/json", data)
            elif path == "/api/export_csv":
                buf = io.StringIO()
                import csv as _csv
                fieldnames = [
                    "id", "url", "method", "provider", "model", "status_code",
                    "latency_ms", "tokens_in", "tokens_out", "cost_usd", "timestamp", "error",
                ]
                writer = _csv.DictWriter(buf, fieldnames=fieldnames)
                writer.writeheader()
                for e in entries:
                    writer.writerow({k: getattr(e, k) for k in fieldnames})
                self._respond(200, "text/csv", buf.getvalue().encode(),
                              headers={"Content-Disposition": 'attachment; filename="llm_api_log.csv"'})
            else:
                self._respond(404, "text/plain", b"Not found")

        def _respond(self, code, ctype, body, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if headers:
                for k, v in headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _load_entries(path: str) -> List[_lal.LogEntry]:
    if path.endswith(".jsonl"):
        logger = _lal.LLMLogger(backend="jsonl")
        p = pathlib.Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        logger.entries.append(_lal.LogEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        pass
        return logger.query()
    else:
        logger = _lal.LLMLogger(db_path=path, backend="sqlite")
        return logger.query()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch_gui(initial_file: Optional[str] = None, port: int = 0) -> None:
    """Start the web dashboard and open it in the default browser.

    Parameters
    ----------
    initial_file:
        JSONL or SQLite log file to pre-load.  If *None*, the dashboard starts
        empty (useful when used programmatically alongside an active
        :func:`llm_api_logger.session`).
    port:
        TCP port to bind.  ``0`` lets the OS choose a free port (default).
    """
    entries: List[_lal.LogEntry] = []
    logger = _lal.LLMLogger(backend="jsonl")

    if initial_file:
        try:
            entries = _load_entries(initial_file)
            print(f"Loaded {len(entries)} entries from {initial_file}")
        except Exception as exc:
            print(f"Warning: could not load {initial_file!r}: {exc}", file=sys.stderr)

    server = HTTPServer(("127.0.0.1", port), _make_handler(entries, logger))
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"

    print(f"LLM API Logger dashboard running at {url}")
    print("Press Ctrl-C to stop.")

    # Open browser after a short delay so the server is ready
    def _open():
        time.sleep(0.4)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    _file = sys.argv[1] if len(sys.argv) > 1 else None
    launch_gui(initial_file=_file)
