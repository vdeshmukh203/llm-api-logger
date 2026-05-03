"""
llm-api-logger GUI — desktop viewer for LLM API call logs.

Usage:
    python llm_api_logger_gui.py [log_file]
    llm-api-logger-gui [log_file]
"""

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import llm_api_logger as lal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_entries(path: str) -> List[lal.LogEntry]:
    """Load a JSONL or SQLite log file and return a list of LogEntry objects."""
    p = Path(path)
    if not p.exists():
        return []
    if path.endswith(".jsonl"):
        entries: List[lal.LogEntry] = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(lal.LogEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        return entries
    else:
        logger = lal.LLMLogger(db_path=path, backend="sqlite")
        return logger.query()


def _fmt_cost(cost: float) -> str:
    return f"${cost:.6f}"


def _fmt_latency(ms: float) -> str:
    return f"{ms:.0f} ms"


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """Main application window for the LLM API Logger GUI."""

    _COLS = ("timestamp", "provider", "model", "status", "tokens_in", "tokens_out", "cost_usd", "latency_ms")
    _COL_HEADERS = ("Timestamp", "Provider", "Model", "Status", "Tok In", "Tok Out", "Cost (USD)", "Latency")
    _COL_WIDTHS = (165, 80, 160, 55, 70, 70, 90, 80)

    def __init__(self, initial_file: Optional[str] = None) -> None:
        super().__init__()
        self.title("LLM API Logger")
        self.geometry("1100x700")
        self.minsize(800, 500)
        self._entries: List[lal.LogEntry] = []
        self._filtered: List[lal.LogEntry] = []
        self._current_file: Optional[str] = None
        self._build_ui()
        if initial_file:
            self._open_file(initial_file)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()
        self._build_summary_bar()
        self._build_filter_bar()
        self._build_table()
        self._build_detail_pane()
        self._build_statusbar()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O", command=self._cmd_open)
        file_menu.add_command(label="Reload", accelerator="F5", command=self._cmd_reload)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._cmd_export_csv)
        file_menu.add_command(label="Export JSONL…", command=self._cmd_export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q", command=self.destroy)

        help_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._cmd_about)

        self.bind("<Control-o>", lambda _e: self._cmd_open())
        self.bind("<Control-q>", lambda _e: self.destroy())
        self.bind("<F5>", lambda _e: self._cmd_reload())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, relief="flat")
        bar.pack(fill="x", padx=4, pady=(4, 0))

        ttk.Label(bar, text="Log file:").pack(side="left")
        self._file_var = tk.StringVar(value="(no file open)")
        ttk.Label(bar, textvariable=self._file_var, foreground="#555", width=55,
                  anchor="w").pack(side="left", padx=(4, 8))
        ttk.Button(bar, text="Browse…", command=self._cmd_open).pack(side="left")
        ttk.Button(bar, text="⟳ Reload", command=self._cmd_reload).pack(side="left", padx=4)

    def _build_summary_bar(self) -> None:
        frame = ttk.LabelFrame(self, text="Summary", padding=4)
        frame.pack(fill="x", padx=4, pady=4)

        self._sum_calls = tk.StringVar(value="0 calls")
        self._sum_cost = tk.StringVar(value="$0.000000")
        self._sum_tok_in = tk.StringVar(value="0 tokens in")
        self._sum_tok_out = tk.StringVar(value="0 tokens out")
        self._sum_latency = tk.StringVar(value="0 ms avg")

        for var in (self._sum_calls, self._sum_cost, self._sum_tok_in,
                    self._sum_tok_out, self._sum_latency):
            ttk.Label(frame, textvariable=var, font=("TkDefaultFont", 10, "bold"),
                      width=18, anchor="center", relief="groove", padding=4).pack(
                side="left", expand=True, fill="x", padx=4)

    def _build_filter_bar(self) -> None:
        frame = ttk.Frame(self, padding=(4, 0, 4, 0))
        frame.pack(fill="x")

        ttk.Label(frame, text="Filter — Model:").pack(side="left")
        self._filter_model = tk.StringVar()
        ttk.Entry(frame, textvariable=self._filter_model, width=18).pack(side="left", padx=(2, 8))

        ttk.Label(frame, text="Provider:").pack(side="left")
        self._filter_provider = tk.StringVar()
        ttk.Entry(frame, textvariable=self._filter_provider, width=12).pack(side="left", padx=(2, 8))

        ttk.Label(frame, text="Status:").pack(side="left")
        self._filter_status = tk.StringVar()
        ttk.Entry(frame, textvariable=self._filter_status, width=6).pack(side="left", padx=(2, 8))

        ttk.Button(frame, text="Apply", command=self._apply_filters).pack(side="left", padx=4)
        ttk.Button(frame, text="Clear", command=self._clear_filters).pack(side="left")

        # bind Enter key on filter fields
        for var in (self._filter_model, self._filter_provider, self._filter_status):
            # We can't bind to StringVar, so we find the entry and bind on apply
            pass
        frame.bind_all("<Return>", lambda _e: self._apply_filters())

    def _build_table(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=4, pady=4)

        self._tree = ttk.Treeview(
            container,
            columns=self._COLS,
            show="headings",
            selectmode="browse",
        )

        for col, header, width in zip(self._COLS, self._COL_HEADERS, self._COL_WIDTHS):
            self._tree.heading(col, text=header, command=lambda c=col: self._sort_by(c))
            anchor = "e" if col in ("tokens_in", "tokens_out", "cost_usd", "latency_ms", "status") else "w"
            self._tree.column(col, width=width, anchor=anchor, stretch=(col == "model"))

        # Scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Alternating row colours
        self._tree.tag_configure("odd", background="#f5f5f5")
        self._tree.tag_configure("even", background="#ffffff")
        self._tree.tag_configure("error", foreground="#c0392b")

        self._sort_col: Optional[str] = None
        self._sort_asc = True

    def _build_detail_pane(self) -> None:
        pane = ttk.LabelFrame(self, text="Entry Details", padding=4)
        pane.pack(fill="x", padx=4, pady=(0, 4))

        nb = ttk.Notebook(pane)
        nb.pack(fill="both", expand=True)

        req_frame = ttk.Frame(nb)
        nb.add(req_frame, text="Request Body")
        self._req_text = self._scrolled_text(req_frame, height=6)

        resp_frame = ttk.Frame(nb)
        nb.add(resp_frame, text="Response Body")
        self._resp_text = self._scrolled_text(resp_frame, height=6)

        meta_frame = ttk.Frame(nb)
        nb.add(meta_frame, text="Metadata")
        self._meta_text = self._scrolled_text(meta_frame, height=6)

    @staticmethod
    def _scrolled_text(parent, height: int = 8) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="none", height=height, font=("Courier", 9))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        text.pack(fill="both", expand=True)
        return text

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self._status_var, relief="sunken",
                  anchor="w", padding=(4, 2)).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Data loading & display
    # ------------------------------------------------------------------

    def _open_file(self, path: str) -> None:
        try:
            self._entries = _load_entries(path)
            self._current_file = path
            self._file_var.set(path)
            self._apply_filters()
            self._set_status(f"Loaded {len(self._entries)} entries from {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))

    def _populate_table(self, entries: List[lal.LogEntry]) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, e in enumerate(entries):
            tag = "odd" if i % 2 else "even"
            if e.error:
                tag = "error"
            cost_str = _fmt_cost(e.cost_usd)
            lat_str = _fmt_latency(e.latency_ms)
            self._tree.insert(
                "", "end", iid=str(i),
                values=(
                    e.timestamp, e.provider, e.model, str(e.status_code),
                    str(e.tokens_in), str(e.tokens_out), cost_str, lat_str,
                ),
                tags=(tag,),
            )

    def _update_summary(self, entries: List[lal.LogEntry]) -> None:
        n = len(entries)
        total_cost = sum(e.cost_usd for e in entries)
        total_in = sum(e.tokens_in for e in entries)
        total_out = sum(e.tokens_out for e in entries)
        avg_lat = (sum(e.latency_ms for e in entries) / n) if n else 0.0
        self._sum_calls.set(f"{n:,} calls")
        self._sum_cost.set(f"${total_cost:.4f}")
        self._sum_tok_in.set(f"{total_in:,} tok in")
        self._sum_tok_out.set(f"{total_out:,} tok out")
        self._sum_latency.set(f"{avg_lat:.0f} ms avg")

    # ------------------------------------------------------------------
    # Filtering & sorting
    # ------------------------------------------------------------------

    def _apply_filters(self) -> None:
        model_q = self._filter_model.get().strip().lower()
        prov_q = self._filter_provider.get().strip().lower()
        status_q = self._filter_status.get().strip()

        filtered = self._entries
        if model_q:
            filtered = [e for e in filtered if model_q in e.model.lower()]
        if prov_q:
            filtered = [e for e in filtered if prov_q in e.provider.lower()]
        if status_q:
            try:
                sc = int(status_q)
                filtered = [e for e in filtered if e.status_code == sc]
            except ValueError:
                pass

        self._filtered = filtered
        self._populate_table(filtered)
        self._update_summary(filtered)
        self._set_status(f"Showing {len(filtered)} of {len(self._entries)} entries")

    def _clear_filters(self) -> None:
        self._filter_model.set("")
        self._filter_provider.set("")
        self._filter_status.set("")
        self._apply_filters()

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        numeric = col in ("tokens_in", "tokens_out", "cost_usd", "latency_ms", "status_code")

        def key(e: lal.LogEntry):
            val = getattr(e, col)
            return (val or 0) if numeric else (val or "").lower()

        self._filtered.sort(key=key, reverse=not self._sort_asc)
        self._populate_table(self._filtered)

    # ------------------------------------------------------------------
    # Selection detail view
    # ------------------------------------------------------------------

    def _on_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self._filtered):
            return
        entry = self._filtered[idx]
        self._show_detail(entry)

    def _show_detail(self, entry: lal.LogEntry) -> None:
        self._set_text(self._req_text, self._pretty_json(entry.request_body))
        self._set_text(self._resp_text, self._pretty_json(entry.response_body))

        meta = {
            "id": entry.id,
            "timestamp": entry.timestamp,
            "url": entry.url,
            "method": entry.method,
            "provider": entry.provider,
            "model": entry.model,
            "status_code": entry.status_code,
            "tokens_in": entry.tokens_in,
            "tokens_out": entry.tokens_out,
            "cost_usd": entry.cost_usd,
            "latency_ms": entry.latency_ms,
            "error": entry.error,
        }
        self._set_text(self._meta_text, json.dumps(meta, indent=2))

    @staticmethod
    def _pretty_json(text: Optional[str]) -> str:
        if not text:
            return "(empty)"
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            return text

    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content)
        widget.config(state="disabled")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Open log file",
            filetypes=[("JSONL files", "*.jsonl"), ("SQLite databases", "*.db *.sqlite"), ("All files", "*")],
        )
        if path:
            self._open_file(path)

    def _cmd_reload(self) -> None:
        if self._current_file:
            self._open_file(self._current_file)
        else:
            self._set_status("No file open — use File > Open first.")

    def _cmd_export_csv(self) -> None:
        if not self._filtered:
            messagebox.showinfo("Export", "No entries to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        tmp_logger = lal.LLMLogger(backend="jsonl")
        for e in self._filtered:
            tmp_logger.entries.append(e)
        tmp_logger.export_csv(path)
        self._set_status(f"Exported {len(self._filtered)} entries to {Path(path).name}")

    def _cmd_export_jsonl(self) -> None:
        if not self._filtered:
            messagebox.showinfo("Export", "No entries to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export JSONL",
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl")],
        )
        if not path:
            return
        tmp_logger = lal.LLMLogger(backend="jsonl")
        for e in self._filtered:
            tmp_logger.entries.append(e)
        tmp_logger.export_jsonl(path)
        self._set_status(f"Exported {len(self._filtered)} entries to {Path(path).name}")

    def _cmd_about(self) -> None:
        messagebox.showinfo(
            "About LLM API Logger GUI",
            f"LLM API Logger v{lal.__version__}\n\n"
            "Desktop viewer for LLM API call logs.\n"
            "Supports JSONL and SQLite backends.\n\n"
            "https://github.com/vdeshmukh203/llm-api-logger",
        )

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the GUI, optionally opening *sys.argv[1]* on startup."""
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = App(initial_file=initial)
    app.mainloop()


if __name__ == "__main__":
    main()
