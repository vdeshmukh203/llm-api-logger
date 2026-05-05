"""
Web dashboard for LLM API Logger.

Launches a local Flask server that renders an interactive dashboard for
browsing, filtering, and exporting log entries produced by llm_api_logger.

Usage
-----
CLI entry point (installed with the ``gui`` extra)::

    llm-api-logger-gui [log_file] [--host HOST] [--port PORT] [--no-browser]

Standalone::

    python gui.py mylog.jsonl
    python gui.py mylog.db --port 8080

Programmatic::

    from gui import launch
    launch("mylog.jsonl", port=5000)

Requires Flask::

    pip install "llm-api-logger[gui]"
"""

import argparse
import csv
import io
import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional

try:
    from flask import Flask, Response, jsonify, render_template_string
    from flask import request as flask_request
except ImportError as _flask_err:
    sys.exit(
        f"Flask is required for the GUI: {_flask_err}\n"
        "Install with:  pip install 'llm-api-logger[gui]'"
    )

from llm_api_logger import LLMLogger, LogEntry

app = Flask(__name__)
_logger: Optional[LLMLogger] = None

# ---------------------------------------------------------------------------
# HTML template (single-file, no external template directory needed)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM API Logger – Dashboard</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
        crossorigin="anonymous">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
          crossorigin="anonymous"></script>
  <style>
    body   { background: #f0f2f5; }
    .kpi   { border-left: 4px solid; transition: box-shadow .15s; }
    .kpi:hover { box-shadow: 0 4px 12px rgba(0,0,0,.1); }
    .kpi-calls    { border-color: #0d6efd; }
    .kpi-cost     { border-color: #198754; }
    .kpi-latency  { border-color: #fd7e14; }
    .kpi-tokens   { border-color: #6f42c1; }
    .kpi-value    { font-size: 1.8rem; font-weight: 700; }
    .chart-wrap   { position: relative; height: 280px; }
    .tbl-scroll   { max-height: 480px; overflow-y: auto; }
    .badge-p      { font-size: .7em; }
    code          { font-size: .82em; }
    pre           { font-size: .78em; max-height: 280px; overflow: auto; }
    .err-row td   { color: #dc3545; }
  </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar navbar-dark bg-dark mb-4 px-3">
  <span class="navbar-brand fw-semibold">📊 LLM API Logger</span>
  <span class="navbar-text text-secondary small" id="lf-label"></span>
  <button class="btn btn-sm btn-outline-light ms-auto" onclick="reload()">⟳ Refresh</button>
</nav>

<div class="container-fluid px-4">

  <!-- KPI cards -->
  <div class="row g-3 mb-4">
    <div class="col-6 col-xl-3">
      <div class="card kpi kpi-calls h-100 p-3">
        <div class="text-muted small">Total API Calls</div>
        <div class="kpi-value text-primary" id="kpi-calls">—</div>
      </div>
    </div>
    <div class="col-6 col-xl-3">
      <div class="card kpi kpi-cost h-100 p-3">
        <div class="text-muted small">Total Cost (USD)</div>
        <div class="kpi-value text-success" id="kpi-cost">—</div>
      </div>
    </div>
    <div class="col-6 col-xl-3">
      <div class="card kpi kpi-latency h-100 p-3">
        <div class="text-muted small">Avg Latency (ms)</div>
        <div class="kpi-value text-warning" id="kpi-latency">—</div>
      </div>
    </div>
    <div class="col-6 col-xl-3">
      <div class="card kpi kpi-tokens h-100 p-3">
        <div class="text-muted small">Total Tokens</div>
        <div class="kpi-value text-purple" id="kpi-tokens" style="color:#6f42c1">—</div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div class="row g-3 mb-4">
    <div class="col-md-6">
      <div class="card h-100">
        <div class="card-header fw-semibold small">Calls by Model</div>
        <div class="card-body"><div class="chart-wrap"><canvas id="chart-calls"></canvas></div></div>
      </div>
    </div>
    <div class="col-md-6">
      <div class="card h-100">
        <div class="card-header fw-semibold small">Cost by Model (USD)</div>
        <div class="card-body"><div class="chart-wrap"><canvas id="chart-cost"></canvas></div></div>
      </div>
    </div>
  </div>

  <!-- Log table -->
  <div class="card mb-5">
    <div class="card-header d-flex align-items-center gap-2 flex-wrap">
      <span class="fw-semibold small me-auto">Log Entries</span>
      <!-- Filters -->
      <select class="form-select form-select-sm w-auto" id="f-model"   onchange="resetPage()"><option value="">All Models</option></select>
      <select class="form-select form-select-sm w-auto" id="f-provider" onchange="resetPage()"><option value="">All Providers</option></select>
      <select class="form-select form-select-sm w-auto" id="f-status"  onchange="resetPage()">
        <option value="">All Status</option>
        <option>200</option><option>400</option><option>401</option>
        <option>429</option><option>500</option>
      </select>
      <!-- Export -->
      <a href="/export?format=csv"  class="btn btn-sm btn-outline-secondary">⬇ CSV</a>
      <a href="/export?format=jsonl" class="btn btn-sm btn-outline-secondary">⬇ JSONL</a>
    </div>
    <div class="card-body p-0">
      <div class="tbl-scroll">
        <table class="table table-sm table-hover align-middle mb-0">
          <thead class="table-dark sticky-top">
            <tr>
              <th class="ps-3">Timestamp</th>
              <th>Provider</th>
              <th>Model</th>
              <th class="text-center">Status</th>
              <th class="text-end">Tok In</th>
              <th class="text-end">Tok Out</th>
              <th class="text-end">Cost (USD)</th>
              <th class="text-end pe-2">Latency</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="tbl-body">
            <tr><td colspan="9" class="text-center text-muted py-5">Loading…</td></tr>
          </tbody>
        </table>
      </div>
      <!-- Pagination -->
      <div class="d-flex justify-content-between align-items-center px-3 py-2 border-top">
        <span class="text-muted small" id="pg-info"></span>
        <div class="btn-group btn-group-sm">
          <button class="btn btn-outline-secondary" id="btn-prev" onclick="prevPage()" disabled>‹</button>
          <button class="btn btn-outline-secondary" id="btn-next" onclick="nextPage()" disabled>›</button>
        </div>
      </div>
    </div>
  </div>

</div><!-- /container -->

<!-- Detail modal -->
<div class="modal fade" id="modal-detail" tabindex="-1">
  <div class="modal-dialog modal-xl modal-dialog-scrollable">
    <div class="modal-content">
      <div class="modal-header">
        <h6 class="modal-title">Log Entry Detail</h6>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body" id="modal-body"></div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const PALETTE = [
  '#0d6efd','#198754','#fd7e14','#6f42c1',
  '#dc3545','#20c997','#0dcaf0','#ffc107',
  '#6c757d','#d63384',
];
let page = 1;
const PAGE = 25;
let total = 0;
let chartCalls = null;
let chartCost  = null;

async function api(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadSummary() {
  let s;
  try { s = await api('/api/summary'); } catch(e) { console.error(e); return; }

  document.getElementById('kpi-calls').textContent   = s.total_calls.toLocaleString();
  document.getElementById('kpi-cost').textContent    = '$' + s.total_cost_usd.toFixed(4);
  document.getElementById('kpi-latency').textContent = s.avg_latency_ms.toFixed(1);
  document.getElementById('kpi-tokens').textContent  =
    (s.total_tokens_in + s.total_tokens_out).toLocaleString();
  document.getElementById('lf-label').textContent = s.log_file ? '📄 ' + s.log_file : '';

  // Populate model filter
  const models = Object.keys(s.calls_by_model).sort();
  const mSel = document.getElementById('f-model');
  mSel.querySelectorAll('option:not(:first-child)').forEach(o => o.remove());
  models.forEach(m => { const o = new Option(m, m); mSel.add(o); });

  // Populate provider filter
  const providers = Object.keys(s.calls_by_provider || {}).sort();
  const pSel = document.getElementById('f-provider');
  pSel.querySelectorAll('option:not(:first-child)').forEach(o => o.remove());
  providers.forEach(p => { const o = new Option(p, p); pSel.add(o); });

  // Calls chart
  const callsCtx = document.getElementById('chart-calls').getContext('2d');
  if (chartCalls) chartCalls.destroy();
  chartCalls = new Chart(callsCtx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{ label: 'Calls', data: models.map(m => s.calls_by_model[m] || 0),
                   backgroundColor: PALETTE.slice(0, models.length) }]
    },
    options: { responsive: true, maintainAspectRatio: false,
               plugins: { legend: { display: false } },
               scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } }
  });

  // Cost chart
  const costCtx = document.getElementById('chart-cost').getContext('2d');
  if (chartCost) chartCost.destroy();
  chartCost = new Chart(costCtx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{ label: 'Cost (USD)', data: models.map(m => (s.cost_by_model[m] || 0).toFixed(6)),
                   backgroundColor: PALETTE.slice(0, models.length) }]
    },
    options: { responsive: true, maintainAspectRatio: false,
               plugins: { legend: { display: false } },
               scales: { y: { beginAtZero: true } } }
  });
}

async function loadEntries() {
  const params = new URLSearchParams({
    page, page_size: PAGE,
    model:       document.getElementById('f-model').value,
    provider:    document.getElementById('f-provider').value,
    status_code: document.getElementById('f-status').value,
  });
  // Remove empty params
  [...params.keys()].forEach(k => { if (!params.get(k)) params.delete(k); });

  let data;
  try { data = await api('/api/entries?' + params); } catch(e) { console.error(e); return; }

  total = data.total;
  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = '';

  if (!data.entries.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted py-5">No entries found</td></tr>';
  } else {
    data.entries.forEach(e => {
      const isErr = e.status_code >= 400;
      const sc = isErr
        ? `<span class="badge bg-danger">${e.status_code}</span>`
        : `<span class="badge bg-success">${e.status_code}</span>`;
      const ts = (e.timestamp || '').replace('T', ' ').slice(0, 19);
      const tr = document.createElement('tr');
      if (isErr) tr.className = 'err-row';
      tr.innerHTML = `
        <td class="ps-3"><small class="font-monospace">${ts}</small></td>
        <td><span class="badge bg-secondary badge-p">${e.provider}</span></td>
        <td><small>${e.model}</small></td>
        <td class="text-center">${sc}</td>
        <td class="text-end"><small>${(e.tokens_in||0).toLocaleString()}</small></td>
        <td class="text-end"><small>${(e.tokens_out||0).toLocaleString()}</small></td>
        <td class="text-end"><small>$${(e.cost_usd||0).toFixed(6)}</small></td>
        <td class="text-end pe-2"><small>${(e.latency_ms||0).toFixed(1)} ms</small></td>
        <td><button class="btn btn-outline-info btn-sm py-0 px-1"
                    onclick="showDetail('${e.id}')">…</button></td>
      `;
      tbody.appendChild(tr);
    });
  }

  const start = (page - 1) * PAGE + 1;
  const end   = Math.min(page * PAGE, total);
  document.getElementById('pg-info').textContent =
    total > 0 ? `Showing ${start}–${end} of ${total.toLocaleString()} entries` : '0 entries';
  document.getElementById('btn-prev').disabled = page <= 1;
  document.getElementById('btn-next').disabled = end >= total;
}

function resetPage() { page = 1; loadEntries(); }
function prevPage()  { if (page > 1) { page--; loadEntries(); } }
function nextPage()  { if (page * PAGE < total) { page++; loadEntries(); } }
function reload()    { loadSummary(); loadEntries(); }

async function showDetail(id) {
  let e;
  try { e = await api('/api/entry/' + id); } catch(err) { alert('Entry not found'); return; }

  const fmt = s => {
    if (!s) return '<em class="text-muted">—</em>';
    try {
      return '<pre class="bg-light p-2 rounded border">'
           + JSON.stringify(JSON.parse(s), null, 2) + '</pre>';
    } catch { return '<pre class="bg-light p-2 rounded border">' + s + '</pre>'; }
  };

  document.getElementById('modal-body').innerHTML = `
    <dl class="row small mb-3">
      <dt class="col-sm-3">ID</dt><dd class="col-sm-9"><code>${e.id}</code></dd>
      <dt class="col-sm-3">Timestamp</dt><dd class="col-sm-9">${e.timestamp}</dd>
      <dt class="col-sm-3">URL</dt><dd class="col-sm-9"><code class="text-break">${e.url}</code></dd>
      <dt class="col-sm-3">Provider / Model</dt>
      <dd class="col-sm-9">
        <span class="badge bg-secondary me-1">${e.provider}</span>${e.model}
      </dd>
      <dt class="col-sm-3">Status</dt>
      <dd class="col-sm-9">
        <span class="badge ${e.status_code >= 400 ? 'bg-danger' : 'bg-success'}">${e.status_code}</span>
        ${e.error ? ' <span class="text-danger">'+e.error+'</span>' : ''}
      </dd>
      <dt class="col-sm-3">Tokens In / Out</dt>
      <dd class="col-sm-9">${(e.tokens_in||0).toLocaleString()} / ${(e.tokens_out||0).toLocaleString()}</dd>
      <dt class="col-sm-3">Cost (USD)</dt><dd class="col-sm-9">$${(e.cost_usd||0).toFixed(6)}</dd>
      <dt class="col-sm-3">Latency</dt><dd class="col-sm-9">${(e.latency_ms||0).toFixed(2)} ms</dd>
    </dl>
    <h6 class="fw-semibold">Request Body</h6>${fmt(e.request_body)}
    <h6 class="fw-semibold mt-3">Response Body</h6>${fmt(e.response_body)}
  `;
  new bootstrap.Modal(document.getElementById('modal-detail')).show();
}

// Initial load
loadSummary();
loadEntries();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template_string(_DASHBOARD_HTML)


@app.route("/api/summary")
def api_summary():
    s = _logger.summary()
    s["log_file"] = app.config.get("LOG_FILE", "")
    return jsonify(s)


@app.route("/api/entries")
def api_entries():
    model       = flask_request.args.get("model") or None
    provider    = flask_request.args.get("provider") or None
    status_code = flask_request.args.get("status_code", type=int)
    page        = flask_request.args.get("page", 1, type=int)
    page_size   = flask_request.args.get("page_size", 25, type=int)

    all_entries = _logger.query(model=model, provider=provider, status_code=status_code)
    start = (page - 1) * page_size
    page_entries = all_entries[start : start + page_size]

    return jsonify({
        "total": len(all_entries),
        "page": page,
        "page_size": page_size,
        "entries": [e.to_dict() for e in page_entries],
    })


@app.route("/api/entry/<entry_id>")
def api_entry(entry_id: str):
    for e in _logger.query():
        if e.id == entry_id:
            return jsonify(e.to_dict())
    return jsonify({"error": "Entry not found"}), 404


@app.route("/export")
def export_logs():
    fmt = flask_request.args.get("format", "csv")
    entries = _logger.query()

    if fmt == "jsonl":
        body = "\n".join(json.dumps(e.to_dict()) for e in entries)
        return Response(
            body,
            mimetype="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=llm_logs.jsonl"},
        )

    si = io.StringIO()
    fieldnames = [
        "id", "url", "method", "provider", "model", "status_code",
        "latency_ms", "tokens_in", "tokens_out", "cost_usd", "timestamp", "error",
    ]
    writer = csv.DictWriter(si, fieldnames=fieldnames)
    writer.writeheader()
    for e in entries:
        writer.writerow({k: getattr(e, k) for k in fieldnames})
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=llm_logs.csv"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_logger(log_file: str) -> LLMLogger:
    """Load a log file into an LLMLogger instance."""
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    if backend == "sqlite":
        return LLMLogger(db_path=log_file, backend="sqlite")

    logger = LLMLogger(db_path=":memory:", backend="jsonl")
    path = Path(log_file)
    if path.exists():
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    logger.entries.append(LogEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
    return logger


def launch(
    log_file: str = "llm_api.jsonl",
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
) -> None:
    """Start the dashboard web server.

    Parameters
    ----------
    log_file:
        Path to the JSONL or SQLite log file to visualise.
    host:
        Network interface to bind (default ``127.0.0.1``).
    port:
        TCP port to listen on (default ``5000``).
    debug:
        Enable Flask debug mode (auto-reload, verbose errors).
    open_browser:
        Open a browser tab automatically when the server starts.
    """
    global _logger
    _logger = _load_logger(log_file)
    app.config["LOG_FILE"] = log_file

    url = f"http://{host}:{port}"
    print(f"LLM API Logger dashboard → {url}")
    print(f"Log file : {log_file}  ({_logger.count()} entries)")
    print("Press Ctrl-C to stop.\n")

    if open_browser and not debug:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=debug)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM API Logger – Web Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  llm-api-logger-gui mylog.jsonl --port 8080",
    )
    parser.add_argument(
        "log_file", nargs="?", default="llm_api.jsonl",
        help="Path to JSONL or SQLite log file (default: llm_api.jsonl)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()
    launch(args.log_file, host=args.host, port=args.port,
           debug=args.debug, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
