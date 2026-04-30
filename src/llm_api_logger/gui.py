"""Tkinter dashboard for browsing and analysing LLM API log records.

Launch via::

    python -m llm_api_logger.gui [log_file.jsonl]

or via the CLI::

    llm-api-logger gui [log_file.jsonl]
"""

import csv
import json
import pathlib
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from .store import LogRecord, LogStore

# ---------------------------------------------------------------------------
# Palette / layout constants
# ---------------------------------------------------------------------------
_BG = "#1e1e2e"
_FG = "#cdd6f4"
_ACCENT = "#89b4fa"
_ROW_ALT = "#2a2a3d"
_ROW_SEL = "#313244"
_ERROR_FG = "#f38ba8"
_MONO = ("Courier", 10)
_SANS = ("TkDefaultFont", 10)


class _DetailPane(tk.Frame):
    """Scrollable text pane showing full detail of a selected record."""

    def __init__(self, parent):
        super().__init__(parent, bg=_BG)
        self._text = tk.Text(
            self, bg=_BG, fg=_FG, font=_MONO, wrap=tk.WORD, state=tk.DISABLED
        )
        sb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._text.yview)
        self._text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def show(self, record: Optional[LogRecord]) -> None:
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        if record is None:
            self._text.insert(tk.END, "Select a row to inspect.")
        else:
            lines = [
                f"ID          : {record.id}",
                f"Timestamp   : {record.timestamp}",
                f"Provider    : {record.provider}",
                f"Model       : {record.model}",
                f"URL         : {record.url}",
                f"Method      : {record.method}",
                f"Status      : {record.status_code}",
                f"Latency(ms) : {record.latency_ms:.1f}",
                f"Tokens in   : {record.tokens_in}",
                f"Tokens out  : {record.tokens_out}",
                f"Cost (USD)  : ${record.cost_usd:.6f}",
                f"SHA-256     : {record.sha256}",
                "",
            ]
            if record.error:
                lines += [f"ERROR: {record.error}", ""]
            lines.append("── Request body ──────────────────────────────────────")
            try:
                req = json.dumps(json.loads(record.request_body or ""), indent=2)
            except Exception:
                req = record.request_body or "(empty)"
            lines.append(req)
            lines.append("")
            lines.append("── Response body ─────────────────────────────────────")
            try:
                resp = json.dumps(json.loads(record.response_body or ""), indent=2)
            except Exception:
                resp = record.response_body or "(empty)"
            lines.append(resp)
            self._text.insert(tk.END, "\n".join(lines))
        self._text.configure(state=tk.DISABLED)


class _SummaryBar(tk.Frame):
    """Top bar showing aggregate statistics."""

    def __init__(self, parent):
        super().__init__(parent, bg=_BG, pady=4)
        self._labels: dict = {}
        fields = [
            ("calls", "Calls"),
            ("cost", "Cost (USD)"),
            ("tok_in", "Tokens in"),
            ("tok_out", "Tokens out"),
            ("latency", "Avg latency"),
        ]
        for key, title in fields:
            col = tk.Frame(self, bg=_BG, padx=12)
            col.pack(side=tk.LEFT)
            tk.Label(col, text=title, fg=_ACCENT, bg=_BG, font=("TkDefaultFont", 8)).pack()
            lbl = tk.Label(col, text="—", fg=_FG, bg=_BG, font=("TkDefaultFont", 12, "bold"))
            lbl.pack()
            self._labels[key] = lbl

    def update_summary(self, stats: dict) -> None:
        self._labels["calls"].config(text=str(stats.get("total_calls", 0)))
        self._labels["cost"].config(text=f"${stats.get('total_cost_usd', 0):.4f}")
        self._labels["tok_in"].config(text=f"{stats.get('total_tokens_in', 0):,}")
        self._labels["tok_out"].config(text=f"{stats.get('total_tokens_out', 0):,}")
        latency = stats.get("avg_latency_ms", 0)
        self._labels["latency"].config(text=f"{latency:.1f} ms")


class _FilterBar(tk.Frame):
    """Row of filter widgets (provider, model, status, free-text search)."""

    def __init__(self, parent, on_change):
        super().__init__(parent, bg=_BG, pady=4)
        self._on_change = on_change

        def _lbl(text):
            tk.Label(self, text=text, fg=_ACCENT, bg=_BG, padx=6).pack(side=tk.LEFT)

        _lbl("Provider")
        self.provider_var = tk.StringVar()
        self.provider_combo = ttk.Combobox(
            self, textvariable=self.provider_var, width=12, state="readonly"
        )
        self.provider_combo.pack(side=tk.LEFT)
        self.provider_combo.bind("<<ComboboxSelected>>", lambda _: on_change())

        _lbl("Model")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self, textvariable=self.model_var, width=18, state="readonly"
        )
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _: on_change())

        _lbl("Status")
        self.status_var = tk.StringVar()
        self.status_combo = ttk.Combobox(
            self, textvariable=self.status_var, width=7, state="readonly"
        )
        self.status_combo.pack(side=tk.LEFT)
        self.status_combo.bind("<<ComboboxSelected>>", lambda _: on_change())

        _lbl("Search")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: on_change())
        tk.Entry(self, textvariable=self.search_var, bg=_ROW_ALT, fg=_FG,
                 insertbackground=_FG, width=20).pack(side=tk.LEFT)

        tk.Button(self, text="Clear", command=self._clear, bg=_ROW_ALT,
                  fg=_FG, bd=0, padx=8).pack(side=tk.LEFT, padx=6)

    def _clear(self) -> None:
        self.provider_var.set("")
        self.model_var.set("")
        self.status_var.set("")
        self.search_var.set("")
        self._on_change()

    def populate_options(self, records: List[LogRecord]) -> None:
        providers = sorted({r.provider for r in records if r.provider})
        models = sorted({r.model for r in records if r.model})
        statuses = sorted({str(r.status_code) for r in records})
        self.provider_combo["values"] = [""] + providers
        self.model_combo["values"] = [""] + models
        self.status_combo["values"] = [""] + statuses

    @property
    def filters(self) -> dict:
        return {
            "provider": self.provider_var.get() or None,
            "model": self.model_var.get() or None,
            "status": self.status_var.get() or None,
            "search": self.search_var.get().strip().lower() or None,
        }


_COLUMNS = [
    ("timestamp", "Timestamp", 160),
    ("provider", "Provider", 80),
    ("model", "Model", 140),
    ("status_code", "Status", 60),
    ("latency_ms", "Latency(ms)", 90),
    ("tokens_in", "Tok↑", 60),
    ("tokens_out", "Tok↓", 60),
    ("cost_usd", "Cost($)", 80),
]


class LogDashboard(tk.Tk):
    """Main dashboard window."""

    def __init__(self, store: Optional[LogStore] = None, auto_refresh_ms: int = 3000):
        super().__init__()
        self.title("LLM API Logger — Dashboard")
        self.geometry("1200x700")
        self.configure(bg=_BG)
        self._store = store or LogStore(":memory:")
        self._auto_refresh_ms = auto_refresh_ms
        self._records: List[LogRecord] = []
        self._selected: Optional[LogRecord] = None

        self._build_ui()
        self._apply_theme()
        self.refresh()
        self._schedule_refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Menu
        menu = tk.Menu(self, bg=_BG, fg=_FG, activebackground=_ACCENT,
                       activeforeground=_BG)
        self.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=False, bg=_BG, fg=_FG,
                            activebackground=_ACCENT, activeforeground=_BG)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open log file…", command=self._open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_command(label="Export JSONL…", command=self._export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)

        # Summary bar
        self._summary = _SummaryBar(self)
        self._summary.pack(fill=tk.X)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Filter bar
        self._filter = _FilterBar(self, on_change=self._apply_filters)
        self._filter.pack(fill=tk.X)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Main pane — table left, detail right
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=_BG, sashwidth=4)
        pane.pack(fill=tk.BOTH, expand=True)

        # Table
        table_frame = tk.Frame(pane, bg=_BG)
        pane.add(table_frame, minsize=500)

        cols = [c[0] for c in _COLUMNS]
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                  selectmode="browse")
        for cid, label, width in _COLUMNS:
            self._tree.heading(cid, text=label,
                               command=lambda c=cid: self._sort_by(c))
            self._tree.column(cid, width=width, minwidth=40, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL,
                             command=self._tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL,
                             command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Detail pane
        self._detail = _DetailPane(pane)
        pane.add(self._detail, minsize=300)

        # Status bar
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self._status_var, bg=_BG, fg=_ACCENT,
                 anchor=tk.W, padx=6).pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=_BG, foreground=_FG,
                        fieldbackground=_BG, rowheight=22)
        style.configure("Treeview.Heading", background=_ROW_ALT, foreground=_ACCENT,
                        relief="flat")
        style.map("Treeview", background=[("selected", _ROW_SEL)],
                  foreground=[("selected", _FG)])
        style.configure("TSeparator", background=_ROW_ALT)
        style.configure("TScrollbar", background=_ROW_ALT, troughcolor=_BG)

    # ------------------------------------------------------------------
    def _populate_tree(self, records: List[LogRecord]) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, r in enumerate(records):
            tag = "alt" if i % 2 else "normal"
            err_tag = "error" if r.error else tag
            self._tree.insert(
                "",
                tk.END,
                iid=r.id,
                values=(
                    r.timestamp[:19],
                    r.provider,
                    r.model,
                    r.status_code,
                    f"{r.latency_ms:.1f}",
                    r.tokens_in,
                    r.tokens_out,
                    f"${r.cost_usd:.4f}",
                ),
                tags=(err_tag,),
            )
        self._tree.tag_configure("alt", background=_ROW_ALT)
        self._tree.tag_configure("normal", background=_BG)
        self._tree.tag_configure("error", foreground=_ERROR_FG)
        self._status_var.set(f"{len(records)} records displayed")

    def _apply_filters(self) -> None:
        f = self._filter.filters
        records = self._records
        if f["provider"]:
            records = [r for r in records if r.provider == f["provider"]]
        if f["model"]:
            records = [r for r in records if r.model == f["model"]]
        if f["status"]:
            records = [r for r in records if str(r.status_code) == f["status"]]
        if f["search"]:
            term = f["search"]
            records = [
                r for r in records
                if term in (r.url or "").lower()
                or term in (r.model or "").lower()
                or term in (r.provider or "").lower()
                or term in (r.request_body or "").lower()
                or term in (r.response_body or "").lower()
            ]
        self._populate_tree(records)

    def _on_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            self._detail.show(None)
            return
        rid = sel[0]
        record = next((r for r in self._records if r.id == rid), None)
        self._selected = record
        self._detail.show(record)

    def _sort_by(self, col: str) -> None:
        reverse = getattr(self, f"_sort_{col}_rev", False)
        try:
            self._records.sort(
                key=lambda r: (getattr(r, col) or 0), reverse=reverse
            )
        except TypeError:
            self._records.sort(
                key=lambda r: str(getattr(r, col) or ""), reverse=reverse
            )
        setattr(self, f"_sort_{col}_rev", not reverse)
        self._apply_filters()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._records = self._store.all()
        self._filter.populate_options(self._records)
        self._summary.update_summary(self._store.summary())
        self._apply_filters()

    def _schedule_refresh(self) -> None:
        self.after(self._auto_refresh_ms, self._auto_refresh)

    def _auto_refresh(self) -> None:
        self.refresh()
        self._schedule_refresh()

    # ------------------------------------------------------------------
    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open log file",
            filetypes=[("JSONL logs", "*.jsonl"), ("All files", "*")],
        )
        if not path:
            return
        try:
            self._store = LogStore(path)
            self.title(f"LLM API Logger — {pathlib.Path(path).name}")
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load file:\n{exc}")

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        records = self._records
        fields = [c[0] for c in _COLUMNS]
        try:
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for r in records:
                    writer.writerow({f: getattr(r, f) for f in fields})
            messagebox.showinfo("Exported", f"Wrote {len(records)} rows to {path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _export_jsonl(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export JSONL",
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl")],
        )
        if not path:
            return
        records = self._records
        try:
            with open(path, "w") as fh:
                for r in records:
                    fh.write(json.dumps(r.to_dict()) + "\n")
            messagebox.showinfo("Exported", f"Wrote {len(records)} records to {path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


# ---------------------------------------------------------------------------
def launch(log_path: Optional[str] = None) -> None:
    """Open the dashboard, optionally loading *log_path* on start."""
    store = None
    if log_path:
        store = LogStore(log_path)
    app = LogDashboard(store=store)
    app.mainloop()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    launch(path)


if __name__ == "__main__":
    main()
