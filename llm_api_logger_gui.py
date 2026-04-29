#!/usr/bin/env python3
"""
LLM API Logger GUI - Interactive dashboard for exploring LLM API logs.

Usage:
    python llm_api_logger_gui.py [log_file]
"""

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
import llm_api_logger as lal

_COLUMNS = (
    ("timestamp",   "Timestamp",      165, tk.W),
    ("provider",    "Provider",        90, tk.W),
    ("model",       "Model",          155, tk.W),
    ("status_code", "Status",          58, tk.CENTER),
    ("tokens_in",   "In Tokens",       80, tk.E),
    ("tokens_out",  "Out Tokens",      85, tk.E),
    ("cost_usd",    "Cost (USD)",      95, tk.E),
    ("latency_ms",  "Latency (ms)",    95, tk.E),
)


class _SortState:
    def __init__(self):
        self.col: Optional[str] = None
        self.reverse = False


class LLMLoggerGUI:
    """Main GUI application for browsing LLM API logs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM API Logger Dashboard")
        self.root.geometry("1200x720")
        self.root.minsize(900, 550)

        self._all_entries: List[lal.LogEntry] = []
        self._sort = _SortState()
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()
        self._build_summary()
        self._build_paned()
        self._build_statusbar()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label="Open log file…", accelerator="Ctrl+O", command=self._open_file)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=fm)
        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda _: self._open_file())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(fill=tk.X)

        ttk.Button(bar, text="Open…", command=self._open_file).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, pady=2, padx=4)

        ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value="All")
        self._model_cb = ttk.Combobox(bar, textvariable=self._model_var, width=22, state="readonly")
        self._model_cb.pack(side=tk.LEFT, padx=(2, 8))
        self._model_cb.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        ttk.Label(bar, text="Provider:").pack(side=tk.LEFT)
        self._prov_var = tk.StringVar(value="All")
        self._prov_cb = ttk.Combobox(bar, textvariable=self._prov_var, width=14, state="readonly")
        self._prov_cb.pack(side=tk.LEFT, padx=(2, 8))
        self._prov_cb.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        ttk.Button(bar, text="Reset filters", command=self._reset_filters).pack(side=tk.LEFT)

    def _build_summary(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Summary", padding=(10, 6))
        frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        items = [
            ("calls",    "Total Calls"),
            ("cost",     "Total Cost (USD)"),
            ("tok_in",   "Input Tokens"),
            ("tok_out",  "Output Tokens"),
            ("latency",  "Avg Latency (ms)"),
            ("errors",   "Errors"),
        ]
        self._summary_vals: dict = {}
        for col, (key, label) in enumerate(items):
            cell = ttk.Frame(frame)
            cell.grid(row=0, column=col, padx=18)
            ttk.Label(cell, text=label, font=("TkDefaultFont", 9)).pack()
            lbl = ttk.Label(cell, text="—", font=("TkDefaultFont", 14, "bold"))
            lbl.pack()
            self._summary_vals[key] = lbl

    def _build_paned(self) -> None:
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        # --- table ---
        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)

        cols = tuple(c[0] for c in _COLUMNS)
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        for key, heading, width, anchor in _COLUMNS:
            self._tree.heading(
                key, text=heading,
                command=lambda k=key: self._sort_by(k),
            )
            self._tree.column(key, width=width, anchor=anchor, stretch=(key == "model"))

        self._tree.tag_configure("error", background="#ffe4e4")
        self._tree.tag_configure("odd",   background="#f7f7f7")

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL,   command=self._tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # --- detail panel ---
        detail_frame = ttk.LabelFrame(paned, text="Entry details", padding=5)
        paned.add(detail_frame, weight=1)

        self._detail = tk.Text(
            detail_frame, height=9, wrap=tk.WORD,
            font=("Courier", 10), state=tk.DISABLED,
        )
        dscroll = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self._detail.yview)
        self._detail.configure(yscrollcommand=dscroll.set)
        self._detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dscroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="No file loaded. Use File > Open log file…")
        ttk.Label(
            self.root, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(4, 2),
        ).pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open LLM API log file",
            filetypes=[
                ("Log files", "*.jsonl *.db *.sqlite"),
                ("JSONL files", "*.jsonl"),
                ("SQLite databases", "*.db *.sqlite"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        try:
            if path.endswith(".jsonl"):
                logger = lal.LLMLogger(db_path=":memory:", backend="jsonl")
                with open(path) as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                logger.entries.append(lal.LogEntry.from_dict(json.loads(line)))
                            except Exception:
                                pass
            else:
                logger = lal.LLMLogger(db_path=path, backend="sqlite")

            self._all_entries = logger.query()
            self._refresh_filters()
            self._populate(self._all_entries)
            self._update_summary(self._all_entries)
            self._status(f"Loaded {len(self._all_entries)} entries from {path}")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _refresh_filters(self) -> None:
        models    = sorted({e.model    for e in self._all_entries})
        providers = sorted({e.provider for e in self._all_entries})
        self._model_cb["values"] = ["All"] + models
        self._prov_cb["values"]  = ["All"] + providers
        self._model_var.set("All")
        self._prov_var.set("All")

    def _apply_filters(self) -> None:
        model = self._model_var.get()
        prov  = self._prov_var.get()
        entries = [
            e for e in self._all_entries
            if (model == "All" or e.model    == model)
            and (prov  == "All" or e.provider == prov)
        ]
        self._populate(entries)
        self._update_summary(entries)
        self._status(f"Showing {len(entries)} of {len(self._all_entries)} entries")

    def _reset_filters(self) -> None:
        self._model_var.set("All")
        self._prov_var.set("All")
        self._populate(self._all_entries)
        self._update_summary(self._all_entries)
        self._status(f"Showing all {len(self._all_entries)} entries")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate(self, entries: List[lal.LogEntry]) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, e in enumerate(entries):
            ts = (e.timestamp or "")[:19]
            values = (
                ts,
                e.provider,
                e.model,
                e.status_code,
                f"{e.tokens_in:,}",
                f"{e.tokens_out:,}",
                f"${e.cost_usd:.6f}",
                f"{e.latency_ms:.1f}",
            )
            is_error = bool(e.error) or (e.status_code is not None and e.status_code >= 400)
            tag = "error" if is_error else ("odd" if i % 2 else "")
            self._tree.insert("", tk.END, iid=e.id, values=values, tags=(tag,))

    def _sort_by(self, col: str) -> None:
        if self._sort.col == col:
            self._sort.reverse = not self._sort.reverse
        else:
            self._sort.col = col
            self._sort.reverse = False

        numeric_cols = {"status_code", "tokens_in", "tokens_out", "cost_usd", "latency_ms"}

        def key(item):
            raw = self._tree.set(item, col).replace("$", "").replace(",", "")
            if col in numeric_cols:
                try:
                    return float(raw)
                except ValueError:
                    return 0.0
            return raw

        items = list(self._tree.get_children(""))
        items.sort(key=key, reverse=self._sort.reverse)
        for idx, iid in enumerate(items):
            self._tree.move(iid, "", idx)

        arrow = " ▲" if not self._sort.reverse else " ▼"
        for key2, heading, *_ in _COLUMNS:
            self._tree.heading(key2, text=heading + (arrow if key2 == col else ""))

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------

    def _on_select(self, _event) -> None:
        iid = self._tree.focus()
        if not iid:
            return
        entry = next((e for e in self._all_entries if e.id == iid), None)
        if entry is None:
            return

        lines = [
            f"ID:          {entry.id}",
            f"Timestamp:   {entry.timestamp}",
            f"URL:         {entry.url}",
            f"Provider:    {entry.provider}",
            f"Model:       {entry.model}",
            f"Status:      {entry.status_code}",
            f"Latency:     {entry.latency_ms:.2f} ms",
            f"Tokens in:   {entry.tokens_in:,}",
            f"Tokens out:  {entry.tokens_out:,}",
            f"Cost:        ${entry.cost_usd:.6f}",
        ]
        if entry.error:
            lines.append(f"Error:       {entry.error}")
        if entry.request_body:
            lines.append("\n── Request body ──")
            try:
                pretty = json.dumps(json.loads(entry.request_body), indent=2)
                lines.append(pretty[:1200] + ("…" if len(pretty) > 1200 else ""))
            except Exception:
                lines.append(entry.request_body[:1200])

        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)
        self._detail.insert(tk.END, "\n".join(lines))
        self._detail.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Summary bar
    # ------------------------------------------------------------------

    def _update_summary(self, entries: List[lal.LogEntry]) -> None:
        if not entries:
            for lbl in self._summary_vals.values():
                lbl.config(text="—")
            return
        n = len(entries)
        cost    = sum(e.cost_usd   for e in entries)
        tok_in  = sum(e.tokens_in  for e in entries)
        tok_out = sum(e.tokens_out for e in entries)
        lat     = sum(e.latency_ms for e in entries) / n
        errors  = sum(1 for e in entries if e.error or (e.status_code and e.status_code >= 400))

        self._summary_vals["calls"].config(text=str(n))
        self._summary_vals["cost"].config(text=f"${cost:.4f}")
        self._summary_vals["tok_in"].config(text=f"{tok_in:,}")
        self._summary_vals["tok_out"].config(text=f"{tok_out:,}")
        self._summary_vals["latency"].config(text=f"{lat:.1f}")
        self._summary_vals["errors"].config(
            text=str(errors),
            foreground="red" if errors else "black",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)


def main() -> None:
    root = tk.Tk()
    app = LLMLoggerGUI(root)
    if len(sys.argv) > 1:
        app._load(sys.argv[1])
    root.mainloop()


if __name__ == "__main__":
    main()
