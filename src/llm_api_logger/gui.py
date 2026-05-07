"""
Tkinter-based dashboard for browsing and analysing LLM API logs.

Launch via the CLI::

    llm-api-logger gui [path/to/log.jsonl]

or directly::

    python -m llm_api_logger.gui [path/to/log.jsonl]

The dashboard loads a JSONL or SQLite log file and presents a sortable,
filterable table of log entries.  Double-clicking a row opens a detail view
that renders the request and response bodies as formatted JSON.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

# ---------------------------------------------------------------------------
# Import the core logger (works whether the package is installed or not)
# ---------------------------------------------------------------------------
try:
    from .logger import LLMLogger, LogEntry
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from llm_api_logger import LLMLogger, LogEntry  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------

class _Col:
    def __init__(self, key: str, heading: str, width: int, anchor: str = tk.W):
        self.key = key
        self.heading = heading
        self.width = width
        self.anchor = anchor


_COLUMNS: List[_Col] = [
    _Col("timestamp",  "Timestamp",    160),
    _Col("provider",   "Provider",      90),
    _Col("model",      "Model",        150),
    _Col("status_code","Status",        60, tk.CENTER),
    _Col("latency_ms", "Latency (ms)", 100, tk.E),
    _Col("tokens_in",  "Tokens In",     90, tk.E),
    _Col("tokens_out", "Tokens Out",    90, tk.E),
    _Col("cost_usd",   "Cost (USD)",    95, tk.E),
    _Col("error",      "Error",        220),
]


# ---------------------------------------------------------------------------
# Detail window
# ---------------------------------------------------------------------------

class _DetailWindow(tk.Toplevel):
    """Popup showing full metadata, request body, and response body."""

    def __init__(self, parent: tk.Tk, entry: LogEntry) -> None:
        super().__init__(parent)
        self.title(f"Entry — {entry.id[:8]}…")
        self.geometry("860x620")
        self.resizable(True, True)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Overview tab
        ov = ttk.Frame(nb, padding=8)
        nb.add(ov, text="Overview")
        fields = [
            ("ID",            entry.id),
            ("Timestamp",     entry.timestamp),
            ("URL",           entry.url),
            ("Method",        entry.method),
            ("Provider",      entry.provider),
            ("Model",         entry.model),
            ("Status Code",   str(entry.status_code)),
            ("Latency (ms)",  f"{entry.latency_ms:.2f}"),
            ("Tokens In",     f"{entry.tokens_in:,}"),
            ("Tokens Out",    f"{entry.tokens_out:,}"),
            ("Cost (USD)",    f"${entry.cost_usd:.8f}"),
            ("Error",         entry.error or "—"),
        ]
        for i, (label, value) in enumerate(fields):
            ttk.Label(ov, text=f"{label}:", font=("TkDefaultFont", 10, "bold")).grid(
                row=i, column=0, sticky=tk.W, padx=(0, 12), pady=2)
            ttk.Label(ov, text=value, wraplength=600, justify=tk.LEFT).grid(
                row=i, column=1, sticky=tk.W, pady=2)

        # Request body tab
        req_frame = ttk.Frame(nb)
        nb.add(req_frame, text="Request Body")
        self._text_tab(req_frame, entry.request_body)

        # Response body tab
        resp_frame = ttk.Frame(nb)
        nb.add(resp_frame, text="Response Body")
        self._text_tab(resp_frame, entry.response_body)

    def _text_tab(self, parent: ttk.Frame, content: Optional[str]) -> None:
        text = tk.Text(parent, wrap=tk.WORD, font=("Courier", 10))
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=text.yview)
        hsb = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        display = content or "(empty)"
        try:
            display = json.dumps(json.loads(display), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        text.insert("1.0", display)
        text.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

class Dashboard(tk.Tk):
    """Application window for the LLM API Logger dashboard."""

    def __init__(self) -> None:
        super().__init__()
        self.title("LLM API Logger — Dashboard")
        self.geometry("1260x720")
        self.minsize(900, 520)

        self._logger: Optional[LLMLogger] = None
        self._all_entries: List[LogEntry] = []
        self._log_file: Optional[str] = None
        self._sort_col: Optional[str] = None
        self._sort_rev: bool = False

        self._build_menu()
        self._build_filter_bar()
        self._build_table()
        self._build_status_bar()

        self._set_status("No log file loaded.  Use File → Open to begin.")

    # ------------------------------------------------------------------ #
    # Menu
    # ------------------------------------------------------------------ #

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Log File…", command=self._open_file,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export as CSV…",  command=self._export_csv)
        file_menu.add_command(label="Export as JSONL…", command=self._export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Refresh",       command=self._refresh, accelerator="F5")
        view_menu.add_command(label="Clear Filters", command=self._clear_filters)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self.bind("<Control-o>", lambda _: self._open_file())
        self.bind("<F5>",        lambda _: self._refresh())

    # ------------------------------------------------------------------ #
    # Filter bar
    # ------------------------------------------------------------------ #

    def _build_filter_bar(self) -> None:
        bar = ttk.LabelFrame(self, text="Filters", padding=6)
        bar.pack(fill=tk.X, padx=8, pady=(8, 0))

        ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar()
        self._model_cb = ttk.Combobox(bar, textvariable=self._model_var, width=24, state="readonly")
        self._model_cb.pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(bar, text="Provider:").pack(side=tk.LEFT)
        self._prov_var = tk.StringVar()
        self._prov_cb = ttk.Combobox(bar, textvariable=self._prov_var, width=14, state="readonly")
        self._prov_cb.pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(bar, text="Status Code:").pack(side=tk.LEFT)
        self._status_var = tk.StringVar()
        self._status_cb = ttk.Combobox(
            bar, textvariable=self._status_var, width=8,
            values=["", "200", "400", "401", "403", "429", "500", "502"],
        )
        self._status_cb.pack(side=tk.LEFT, padx=(2, 14))

        ttk.Button(bar, text="Apply",  command=self._apply_filter).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Clear",  command=self._clear_filters).pack(side=tk.LEFT)

    # ------------------------------------------------------------------ #
    # Main table
    # ------------------------------------------------------------------ #

    def _build_table(self) -> None:
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        col_ids = [c.key for c in _COLUMNS]
        self._tree = ttk.Treeview(frame, columns=col_ids, show="headings", selectmode="browse")

        for col in _COLUMNS:
            self._tree.heading(
                col.key, text=col.heading,
                command=lambda k=col.key: self._sort_by(k),
            )
            self._tree.column(
                col.key, width=col.width, anchor=col.anchor,
                stretch=(col.key == "error"),
            )

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL,   command=self._tree.yview)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self._tree.bind("<Double-1>", self._on_double_click)
        self._tree.tag_configure("error", background="#ffe0e0")
        self._tree.tag_configure("odd",   background="#f7f7f7")

    # ------------------------------------------------------------------ #
    # Status bar
    # ------------------------------------------------------------------ #

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self, relief=tk.SUNKEN, padding=(0, 2))
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_lbl = ttk.Label(bar, padding=(8, 0))
        self._status_lbl.pack(side=tk.LEFT)
        self._summary_lbl = ttk.Label(bar, padding=(8, 0))
        self._summary_lbl.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ #
    # File I/O
    # ------------------------------------------------------------------ #

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open log file",
            filetypes=[
                ("JSONL files",     "*.jsonl"),
                ("SQLite databases", "*.db *.sqlite"),
                ("All files",       "*.*"),
            ],
        )
        if path:
            self._log_file = path
            self._load(path)

    def _load(self, path: str) -> None:
        backend = "jsonl" if path.endswith(".jsonl") else "sqlite"
        try:
            log_instance = LLMLogger(db_path=path, backend=backend)
            if backend == "jsonl" and Path(path).exists():
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            log_instance.entries.append(
                                LogEntry.from_dict(json.loads(line))
                            )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
            self._logger = log_instance
            self._all_entries = self._logger.query()
            self._refresh_filter_options()
            self._populate(self._all_entries)
            self._set_status(f"Loaded: {path}  ({len(self._all_entries)} entries)")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    # ------------------------------------------------------------------ #
    # Table population
    # ------------------------------------------------------------------ #

    def _populate(self, entries: List[LogEntry]) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, e in enumerate(entries):
            tag = "error" if e.error else ("odd" if i % 2 else "")
            ts = e.timestamp[:19].replace("T", " ") if e.timestamp else ""
            self._tree.insert(
                "", tk.END, iid=e.id,
                values=(
                    ts,
                    e.provider,
                    e.model,
                    e.status_code,
                    f"{e.latency_ms:.1f}",
                    f"{e.tokens_in:,}",
                    f"{e.tokens_out:,}",
                    f"${e.cost_usd:.6f}",
                    e.error or "",
                ),
                tags=(tag,),
            )
        self._update_summary(entries)

    def _update_summary(self, entries: List[LogEntry]) -> None:
        if not entries:
            self._summary_lbl.config(text="No entries")
            return
        total_cost  = sum(e.cost_usd   for e in entries)
        total_tok   = sum(e.tokens_in + e.tokens_out for e in entries)
        avg_lat     = sum(e.latency_ms for e in entries) / len(entries)
        self._summary_lbl.config(
            text=(
                f"  {len(entries)} entries | "
                f"${total_cost:.4f} total | "
                f"{total_tok:,} tokens | "
                f"{avg_lat:.1f} ms avg  "
            )
        )

    def _set_status(self, msg: str) -> None:
        self._status_lbl.config(text=msg)

    # ------------------------------------------------------------------ #
    # Filters
    # ------------------------------------------------------------------ #

    def _refresh_filter_options(self) -> None:
        models = sorted({e.model for e in self._all_entries if e.model})
        provs  = sorted({e.provider for e in self._all_entries if e.provider})
        self._model_cb["values"] = [""] + models
        self._prov_cb["values"]  = [""] + provs

    def _apply_filter(self) -> None:
        if not self._logger:
            return
        model    = self._model_var.get() or None
        provider = self._prov_var.get()  or None
        status_s = self._status_var.get()
        status   = int(status_s) if status_s.isdigit() else None
        entries  = self._logger.query(model=model, provider=provider, status_code=status)
        self._populate(entries)
        self._set_status(
            f"Showing {len(entries)} of {len(self._all_entries)} entries"
        )

    def _clear_filters(self) -> None:
        self._model_var.set("")
        self._prov_var.set("")
        self._status_var.set("")
        if self._all_entries:
            self._populate(self._all_entries)
            self._set_status(f"Filters cleared.  {len(self._all_entries)} entries.")

    def _refresh(self) -> None:
        if self._log_file:
            self._load(self._log_file)

    # ------------------------------------------------------------------ #
    # Sorting
    # ------------------------------------------------------------------ #

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        entries = sorted(
            self._all_entries,
            key=lambda e: (getattr(e, col) or ""),
            reverse=self._sort_rev,
        )
        self._populate(entries)

    # ------------------------------------------------------------------ #
    # Row detail
    # ------------------------------------------------------------------ #

    def _on_double_click(self, _: tk.Event) -> None:
        item = self._tree.focus()
        if not item:
            return
        entry = next((e for e in self._all_entries if e.id == item), None)
        if entry:
            _DetailWindow(self, entry)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _export_csv(self) -> None:
        if not self._logger:
            messagebox.showwarning("No data", "Load a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._logger.export_csv(path)
            self._set_status(f"Exported {self._logger.count()} entries → {path}")

    def _export_jsonl(self) -> None:
        if not self._logger:
            messagebox.showwarning("No data", "Load a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._logger.export_jsonl(path)
            self._set_status(f"Exported {self._logger.count()} entries → {path}")

    # ------------------------------------------------------------------ #
    # About dialog
    # ------------------------------------------------------------------ #

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About LLM API Logger",
            "LLM API Logger Dashboard\n\n"
            "Transparent logging and analysis of LLM API calls.\n\n"
            "https://github.com/vdeshmukh203/llm-api-logger",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(log_file: Optional[str] = None) -> None:
    """Launch the dashboard, optionally pre-loading *log_file*."""
    app = Dashboard()
    path = log_file or (sys.argv[1] if len(sys.argv) > 1 else None)
    if path:
        app._load(path)
    app.mainloop()


if __name__ == "__main__":
    main()
