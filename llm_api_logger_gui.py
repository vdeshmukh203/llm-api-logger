"""
LLM API Logger — Graphical User Interface

Provides an interactive Tkinter dashboard for loading, browsing, filtering,
and exporting LLM API call logs (JSONL or SQLite format).

Launch via::

    llm-api-logger-gui [LOG_FILE]

or programmatically::

    from llm_api_logger_gui import main_gui
    main_gui()
"""

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure the repo root is importable when running the script directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from llm_api_logger import LLMLogger, LogEntry, __version__


# ---------------------------------------------------------------------------
# Palette / style constants
# ---------------------------------------------------------------------------
_ROW_EVEN = "#f5f5f5"
_ROW_ODD  = "#ffffff"
_SEL_BG   = "#2c82c9"
_HEADING_FONT = ("Helvetica", 9, "bold")
_MONO_FONT    = ("Courier", 9)
_STAT_FONT    = ("Helvetica", 10, "bold")


class _SortState:
    """Tracks per-column sort direction for the Treeview."""

    def __init__(self):
        self._col: Optional[str] = None
        self._asc = True

    def toggle(self, col: str) -> bool:
        if self._col == col:
            self._asc = not self._asc
        else:
            self._col = col
            self._asc = True
        return self._asc


class LLMLoggerGUI:
    """Main application window."""

    # Column definitions: (id, heading, width, stretch)
    _COLUMNS = [
        ("timestamp",  "Timestamp",     165, False),
        ("provider",   "Provider",       78, False),
        ("model",      "Model",         180, True),
        ("status_code","Status",         52, False),
        ("tokens_in",  "Tokens In",      80, False),
        ("tokens_out", "Tokens Out",     85, False),
        ("cost_usd",   "Cost (USD)",     90, False),
        ("latency_ms", "Latency (ms)",   90, False),
        ("error",      "Error",         130, True),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"LLM API Logger  v{__version__}")
        self.root.geometry("1260x820")
        self.root.minsize(900, 600)

        self._logger: Optional[LLMLogger] = None
        self._all_entries: List[LogEntry] = []
        self._shown_entries: List[LogEntry] = []
        self._sort = _SortState()

        self._build_menu()
        self._build_toolbar()
        self._build_main()

        self._apply_style()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Treeview",
                    font=_MONO_FONT,
                    rowheight=23,
                    fieldbackground=_ROW_ODD)
        s.configure("Treeview.Heading",
                    font=_HEADING_FONT,
                    relief="groove")
        s.map("Treeview",
              background=[("selected", _SEL_BG)],
              foreground=[("selected", "#ffffff")])
        s.configure("Stat.TLabel", font=_STAT_FONT, foreground="#2c3e50")
        s.configure("StatVal.TLabel", font=_MONO_FONT, foreground="#16a085")
        s.configure("Section.TLabelframe.Label", font=_HEADING_FONT)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        bar = tk.Menu(self.root)

        file_m = tk.Menu(bar, tearoff=0)
        file_m.add_command(label="Open log file…",
                           command=self._open_file,
                           accelerator="Ctrl+O")
        file_m.add_separator()
        file_m.add_command(label="Export to CSV…",   command=self._export_csv)
        file_m.add_command(label="Export to JSONL…", command=self._export_jsonl)
        file_m.add_separator()
        file_m.add_command(label="Quit",
                           command=self.root.quit,
                           accelerator="Ctrl+Q")
        bar.add_cascade(label="File", menu=file_m)

        view_m = tk.Menu(bar, tearoff=0)
        view_m.add_command(label="Refresh",
                           command=self._refresh,
                           accelerator="F5")
        view_m.add_command(label="Clear filters",
                           command=self._clear_filters,
                           accelerator="Ctrl+R")
        bar.add_cascade(label="View", menu=view_m)

        help_m = tk.Menu(bar, tearoff=0)
        help_m.add_command(label="About", command=self._show_about)
        bar.add_cascade(label="Help", menu=help_m)

        self.root.config(menu=bar)
        self.root.bind("<Control-o>", lambda _e: self._open_file())
        self.root.bind("<Control-q>", lambda _e: self.root.quit())
        self.root.bind("<F5>",        lambda _e: self._refresh())
        self.root.bind("<Control-r>", lambda _e: self._clear_filters())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(fill="x", side="top")

        ttk.Label(bar, text="Log file:").pack(side="left")
        self._file_var = tk.StringVar(value="(none)")
        ttk.Entry(bar, textvariable=self._file_var, width=56,
                  state="readonly").pack(side="left", padx=4)
        ttk.Button(bar, text="Browse…",  command=self._open_file).pack(side="left", padx=2)
        ttk.Button(bar, text="Refresh",  command=self._refresh).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="Export CSV…",   command=self._export_csv).pack(side="left", padx=2)
        ttk.Button(bar, text="Export JSONL…", command=self._export_jsonl).pack(side="left", padx=2)

    def _build_main(self) -> None:
        paned = ttk.PanedWindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        top_pane = ttk.Frame(paned)
        paned.add(top_pane, weight=5)

        self._build_stats(top_pane)
        self._build_filters(top_pane)
        self._build_table(top_pane)

        bot_pane = ttk.Frame(paned)
        paned.add(bot_pane, weight=2)
        self._build_detail(bot_pane)

    def _build_stats(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Summary", padding=6,
                               style="Section.TLabelframe")
        frame.pack(fill="x", pady=(0, 4))

        stat_defs = [
            ("Calls",         "calls"),
            ("Cost (USD)",    "cost"),
            ("Tokens In",     "tokens_in"),
            ("Tokens Out",    "tokens_out"),
            ("Avg Latency",   "latency"),
        ]
        self._stat_vars = {k: tk.StringVar(value="—") for _, k in stat_defs}

        for col, (label, key) in enumerate(stat_defs):
            ttk.Label(frame, text=label + ":", style="Stat.TLabel"
                      ).grid(row=0, column=col * 2, sticky="e", padx=(16, 4))
            ttk.Label(frame, textvariable=self._stat_vars[key], style="StatVal.TLabel"
                      ).grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 12))

    def _build_filters(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Filters", padding=6,
                               style="Section.TLabelframe")
        frame.pack(fill="x", pady=(0, 4))

        ttk.Label(frame, text="Model:").grid(row=0, column=0, padx=(0, 2))
        self._flt_model = ttk.Combobox(frame, width=22, state="readonly")
        self._flt_model.grid(row=0, column=1, padx=4)
        self._flt_model.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(frame, text="Provider:").grid(row=0, column=2, padx=(8, 2))
        self._flt_provider = ttk.Combobox(frame, width=14, state="readonly")
        self._flt_provider.grid(row=0, column=3, padx=4)
        self._flt_provider.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(frame, text="Status:").grid(row=0, column=4, padx=(8, 2))
        self._flt_status = ttk.Combobox(frame, width=7, state="readonly")
        self._flt_status.grid(row=0, column=5, padx=4)
        self._flt_status.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        ttk.Label(frame, text="Search:").grid(row=0, column=6, padx=(8, 2))
        self._flt_search = tk.StringVar()
        ttk.Entry(frame, textvariable=self._flt_search, width=22
                  ).grid(row=0, column=7, padx=4)
        self._flt_search.trace_add("write", lambda *_a: self._apply_filters())

        ttk.Button(frame, text="Clear", command=self._clear_filters
                   ).grid(row=0, column=8, padx=12)

    def _build_table(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        cols = [c[0] for c in self._COLUMNS]
        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                  selectmode="browse")

        for col_id, heading, width, stretch in self._COLUMNS:
            self._tree.heading(
                col_id, text=heading,
                command=lambda c=col_id: self._sort_by(c),
            )
            self._tree.column(col_id, width=width, minwidth=40,
                              stretch=stretch)

        vsb = ttk.Scrollbar(frame, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

        self._count_var = tk.StringVar(value="No log loaded")
        ttk.Label(parent, textvariable=self._count_var, anchor="w",
                  padding=(2, 1)).pack(fill="x")

    def _build_detail(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=2, pady=2)

        def _tab(label: str) -> tk.Text:
            f = ttk.Frame(nb)
            nb.add(f, text=label)
            t = tk.Text(f, wrap="word", font=_MONO_FONT, state="disabled",
                        relief="flat", borderwidth=0)
            sb = ttk.Scrollbar(f, command=t.yview)
            t.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            t.pack(fill="both", expand=True)
            return t

        self._req_text  = _tab("Request Body")
        self._resp_text = _tab("Response Body")
        self._meta_text = _tab("Metadata")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open LLM API log file",
            filetypes=[
                ("Log files", "*.jsonl *.db *.sqlite *.db3"),
                ("JSONL",     "*.jsonl"),
                ("SQLite",    "*.db *.sqlite *.db3"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            if path.lower().endswith(".jsonl"):
                log = LLMLogger(db_path=":memory:", backend="sqlite")
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            log.record(LogEntry.from_dict(json.loads(line)))
                        except Exception:
                            pass
            else:
                log = LLMLogger(db_path=path, backend="sqlite")

            if self._logger is not None:
                self._logger.close()
            self._logger = log
            self._file_var.set(path)
            self._refresh()

        except Exception as exc:
            messagebox.showerror("Open error", f"Could not load file:\n{exc}")

    def _refresh(self) -> None:
        if self._logger is None:
            return
        self._all_entries = self._logger.query()
        self._update_filter_options()
        self._apply_filters()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _update_filter_options(self) -> None:
        models    = sorted({e.model    for e in self._all_entries if e.model})
        providers = sorted({e.provider for e in self._all_entries if e.provider})
        statuses  = sorted({str(e.status_code) for e in self._all_entries})

        def _set_combo(combo: ttk.Combobox, values: list) -> None:
            current = combo.get()
            combo["values"] = ["(all)"] + values
            if current not in combo["values"]:
                combo.set("(all)")

        _set_combo(self._flt_model,    models)
        _set_combo(self._flt_provider, providers)
        _set_combo(self._flt_status,   statuses)

        for combo in (self._flt_model, self._flt_provider, self._flt_status):
            if not combo.get():
                combo.set("(all)")

    def _apply_filters(self) -> None:
        entries = self._all_entries[:]

        m = self._flt_model.get()
        if m and m != "(all)":
            entries = [e for e in entries if e.model == m]

        p = self._flt_provider.get()
        if p and p != "(all)":
            entries = [e for e in entries if e.provider == p]

        st = self._flt_status.get()
        if st and st != "(all)":
            try:
                code = int(st)
                entries = [e for e in entries if e.status_code == code]
            except ValueError:
                pass

        q = self._flt_search.get().strip().lower()
        if q:
            entries = [
                e for e in entries
                if q in (e.model    or "").lower()
                or q in (e.provider or "").lower()
                or q in (e.url      or "").lower()
                or q in (e.error    or "").lower()
            ]

        self._shown_entries = entries
        self._populate_table(entries)
        self._update_stats(entries)

    def _clear_filters(self) -> None:
        self._flt_model.set("(all)")
        self._flt_provider.set("(all)")
        self._flt_status.set("(all)")
        self._flt_search.set("")
        self._apply_filters()

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self, entries: List[LogEntry]) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, e in enumerate(entries):
            tag = "even" if i % 2 == 0 else "odd"
            vals = (
                (e.timestamp or "")[:19],
                e.provider or "",
                e.model or "",
                str(e.status_code),
                f"{e.tokens_in:,}"    if e.tokens_in  else "0",
                f"{e.tokens_out:,}"   if e.tokens_out else "0",
                f"${e.cost_usd:.6f}"  if e.cost_usd   else "$0.000000",
                f"{e.latency_ms:.1f}" if e.latency_ms else "0.0",
                e.error or "",
            )
            self._tree.insert("", "end", iid=str(i), values=vals, tags=(tag,))

        self._tree.tag_configure("even", background=_ROW_EVEN)
        self._tree.tag_configure("odd",  background=_ROW_ODD)
        self._count_var.set(
            f"Showing {len(entries):,} of {len(self._all_entries):,} entries"
        )

    def _sort_by(self, col: str) -> None:
        asc = self._sort.toggle(col)
        items = [(self._tree.set(k, col), k) for k in self._tree.get_children("")]

        # Try numeric sort first, fall back to string
        try:
            items.sort(
                key=lambda t: float(t[0].replace("$", "").replace(",", "")),
                reverse=not asc,
            )
        except ValueError:
            items.sort(key=lambda t: t[0].lower(), reverse=not asc)

        for idx, (_, k) in enumerate(items):
            self._tree.move(k, "", idx)
            tag = "even" if idx % 2 == 0 else "odd"
            self._tree.item(k, tags=(tag,))

        # Show sort direction in heading
        for col_id, heading, *_ in self._COLUMNS:
            label = heading + (" ▲" if col_id == col and asc else
                               " ▼" if col_id == col else "")
            self._tree.heading(col_id, text=label)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def _update_stats(self, entries: List[LogEntry]) -> None:
        n = len(entries)
        cost   = sum(e.cost_usd    for e in entries)
        tin    = sum(e.tokens_in   for e in entries)
        tout   = sum(e.tokens_out  for e in entries)
        lat    = sum(e.latency_ms  for e in entries) / n if n else 0.0

        self._stat_vars["calls"].set(f"{n:,}")
        self._stat_vars["cost"].set(f"${cost:.4f}")
        self._stat_vars["tokens_in"].set(f"{tin:,}")
        self._stat_vars["tokens_out"].set(f"{tout:,}")
        self._stat_vars["latency"].set(f"{lat:.1f} ms")

    # ------------------------------------------------------------------
    # Detail pane
    # ------------------------------------------------------------------

    def _on_row_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < len(self._shown_entries):
            self._show_entry(self._shown_entries[idx])

    def _show_entry(self, entry: LogEntry) -> None:
        def _write(widget: tk.Text, content: Optional[str]) -> None:
            widget.config(state="normal")
            widget.delete("1.0", "end")
            if content:
                try:
                    widget.insert("end", json.dumps(json.loads(content), indent=2))
                except Exception:
                    widget.insert("end", content)
            else:
                widget.insert("end", "(empty)")
            widget.config(state="disabled")

        _write(self._req_text,  entry.request_body)
        _write(self._resp_text, entry.response_body)

        meta_lines = [
            f"ID          : {entry.id}",
            f"Timestamp   : {entry.timestamp}",
            f"URL         : {entry.url}",
            f"Method      : {entry.method}",
            f"Provider    : {entry.provider}",
            f"Model       : {entry.model}",
            f"Status Code : {entry.status_code}",
            f"Latency     : {entry.latency_ms:.2f} ms",
            f"Tokens In   : {entry.tokens_in:,}",
            f"Tokens Out  : {entry.tokens_out:,}",
            f"Cost (USD)  : ${entry.cost_usd:.8f}",
            f"Error       : {entry.error or '(none)'}",
        ]
        _write(self._meta_text, "\n".join(meta_lines))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        self._export("csv")

    def _export_jsonl(self) -> None:
        self._export("jsonl")

    def _export(self, fmt: str) -> None:
        if self._logger is None:
            messagebox.showwarning("No data", "Open a log file first.")
            return
        ext  = ".csv" if fmt == "csv" else ".jsonl"
        ftypes = [("CSV files", "*.csv")] if fmt == "csv" else [("JSONL files", "*.jsonl")]
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=ftypes + [("All files", "*.*")],
        )
        if not path:
            return
        try:
            if fmt == "csv":
                self._logger.export_csv(path)
            else:
                self._logger.export_jsonl(path)
            n = self._logger.count()
            messagebox.showinfo("Export complete",
                                f"Exported {n:,} entries to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export error", f"Export failed:\n{exc}")

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About LLM API Logger",
            f"LLM API Logger  v{__version__}\n\n"
            "Middleware for logging and analysing LLM API calls.\n\n"
            "Supports OpenAI, Anthropic, Google, Mistral, Cohere,\n"
            "Together AI, HuggingFace, and any OpenAI-compatible API.\n\n"
            "https://github.com/vdeshmukh203/llm-api-logger",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_gui(argv: Optional[list] = None) -> None:
    """Launch the LLM API Logger GUI.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv[1:]``).
        Pass a log-file path to open it on startup.
    """
    parser = argparse.ArgumentParser(
        prog="llm-api-logger-gui",
        description="Graphical viewer for LLM API Logger log files",
    )
    parser.add_argument("log_file", nargs="?", default=None,
                        help="JSONL or SQLite log file to open on startup")
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = LLMLoggerGUI(root)

    if args.log_file:
        root.after(100, lambda: app._load_file(args.log_file))

    root.mainloop()


if __name__ == "__main__":
    main_gui()
