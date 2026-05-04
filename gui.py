"""
LLM API Logger — Graphical Dashboard

A Tkinter desktop GUI for loading, browsing, filtering, and exporting
LLM API call logs produced by the llm_api_logger library.

Usage::

    python gui.py [log_file]

    # or after installing the package:
    python -m gui
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import List, Optional

from llm_api_logger import LLMLogger, LogEntry

# ---------------------------------------------------------------------------
# Colour palette (works with both light and dark system themes)
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#f5f5f5",
    "card": "#ffffff",
    "header": "#1a237e",
    "accent": "#3949ab",
    "ok": "#2e7d32",
    "warn": "#f57f17",
    "err": "#b71c1c",
    "text": "#212121",
    "muted": "#757575",
    "border": "#bdbdbd",
}

STAT_LABELS = [
    ("Total Calls", "total_calls", "{}", PALETTE["accent"]),
    ("Total Cost", "total_cost_usd", "${:.4f}", PALETTE["ok"]),
    ("Input Tokens", "total_tokens_in", "{:,}", PALETTE["accent"]),
    ("Output Tokens", "total_tokens_out", "{:,}", PALETTE["accent"]),
    ("Avg Latency", "avg_latency_ms", "{:.1f} ms", PALETTE["warn"]),
]

LOG_COLUMNS = (
    ("timestamp", "Timestamp", 155),
    ("provider", "Provider", 90),
    ("model", "Model", 190),
    ("status_code", "Status", 55),
    ("latency_ms", "Latency ms", 80),
    ("tokens_in", "Tok In", 65),
    ("tokens_out", "Tok Out", 65),
    ("cost_usd", "Cost USD", 85),
    ("error", "Error", 120),
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_log(path: str) -> LLMLogger:
    """Load a JSONL or SQLite log file into an in-memory LLMLogger."""
    if path.endswith(".jsonl"):
        log = LLMLogger(backend="jsonl")
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            log.entries.append(LogEntry.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, TypeError, KeyError):
                            pass
    else:
        log = LLMLogger(db_path=path, backend="sqlite")
    return log


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class App:
    """Root application controller."""

    def __init__(self, root: tk.Tk, initial_file: Optional[str] = None) -> None:
        self.root = root
        self.root.title("LLM API Logger — Dashboard")
        self.root.geometry("1100x700")
        self.root.minsize(800, 500)
        self.root.configure(bg=PALETTE["bg"])

        self._log: Optional[LLMLogger] = None
        self._entries: List[LogEntry] = []
        self._log_path = tk.StringVar()

        self._build_menu()
        self._build_toolbar()
        self._build_stats_bar()
        self._build_main_area()
        self._build_status_bar()

        if initial_file:
            self._log_path.set(initial_file)
            self._load()

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open log…", accelerator="Ctrl+O", command=self._browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export to CSV…", command=self._export_csv)
        file_menu.add_command(label="Export to JSONL…", command=self._export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Refresh", accelerator="F5", command=self._load)
        view_menu.add_command(label="Clear filters", command=self._clear_filters)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.configure(menu=menubar)
        self.root.bind("<Control-o>", lambda _: self._browse_file())
        self.root.bind("<Control-q>", lambda _: self.root.quit())
        self.root.bind("<F5>", lambda _: self._load())

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self.root, bg=PALETTE["header"], padx=8, pady=6)
        bar.pack(fill="x")

        tk.Label(bar, text="Log file:", bg=PALETTE["header"], fg="white",
                 font=("Helvetica", 10)).pack(side="left")
        tk.Entry(bar, textvariable=self._log_path, width=50,
                 font=("Courier", 9)).pack(side="left", padx=(4, 2))
        tk.Button(bar, text="Browse…", command=self._browse_file,
                  relief="flat", bg="#5c6bc0", fg="white",
                  activebackground="#7986cb",
                  padx=8).pack(side="left", padx=2)
        tk.Button(bar, text="Load / Refresh", command=self._load,
                  relief="flat", bg="#43a047", fg="white",
                  activebackground="#66bb6a",
                  padx=8).pack(side="left", padx=6)

    def _build_stats_bar(self) -> None:
        outer = tk.Frame(self.root, bg=PALETTE["bg"], pady=8)
        outer.pack(fill="x", padx=12)
        self._stat_vars: dict[str, tk.StringVar] = {}
        for label, key, fmt, colour in STAT_LABELS:
            card = tk.Frame(outer, bg=PALETTE["card"], relief="raised", bd=1,
                            padx=14, pady=8)
            card.pack(side="left", fill="y", padx=6)
            tk.Label(card, text=label, bg=PALETTE["card"], fg=PALETTE["muted"],
                     font=("Helvetica", 8)).pack(anchor="w")
            var = tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(card, textvariable=var, bg=PALETTE["card"], fg=colour,
                     font=("Helvetica", 14, "bold")).pack(anchor="w")

    def _build_main_area(self) -> None:
        paned = tk.PanedWindow(self.root, orient="vertical",
                               bg=PALETTE["bg"], sashwidth=5)
        paned.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        top = tk.Frame(paned, bg=PALETTE["bg"])
        paned.add(top, stretch="always")
        self._build_filter_bar(top)
        self._build_log_table(top)

        bottom = tk.Frame(paned, bg=PALETTE["bg"])
        paned.add(bottom, stretch="never", minsize=130)
        self._build_detail_pane(bottom)

    def _build_filter_bar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=PALETTE["bg"], pady=4)
        bar.pack(fill="x")

        def lbl(text: str) -> None:
            tk.Label(bar, text=text, bg=PALETTE["bg"], fg=PALETTE["text"],
                     font=("Helvetica", 9)).pack(side="left")

        lbl("  Filter — Model:")
        self._filter_model = tk.StringVar()
        tk.Entry(bar, textvariable=self._filter_model, width=20,
                 font=("Courier", 9)).pack(side="left", padx=(2, 8))

        lbl("Provider:")
        self._filter_provider = tk.StringVar()
        self._provider_combo = ttk.Combobox(bar, textvariable=self._filter_provider,
                                            width=14, state="readonly",
                                            values=["", "openai", "anthropic", "google",
                                                    "mistral", "cohere", "together",
                                                    "huggingface", "unknown"])
        self._provider_combo.pack(side="left", padx=(2, 8))

        lbl("Status:")
        self._filter_status = tk.StringVar()
        tk.Entry(bar, textvariable=self._filter_status, width=6,
                 font=("Courier", 9)).pack(side="left", padx=(2, 8))

        tk.Button(bar, text="Apply", command=self._apply_filters,
                  relief="flat", bg=PALETTE["accent"], fg="white",
                  activebackground="#5c6bc0", padx=8).pack(side="left")
        tk.Button(bar, text="Clear", command=self._clear_filters,
                  relief="flat", bg=PALETTE["border"], fg=PALETTE["text"],
                  padx=8).pack(side="left", padx=4)

    def _build_log_table(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=PALETTE["bg"])
        frame.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("Treeview", rowheight=22, font=("Courier", 9))
        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))

        cols = [c[0] for c in LOG_COLUMNS]
        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                  selectmode="browse")
        for col, heading, width in LOG_COLUMNS:
            self._tree.heading(col, text=heading,
                               command=lambda c=col: self._sort_by(c))
            self._tree.column(col, width=width, anchor="w", stretch=False)

        self._tree.tag_configure("error", background="#ffebee")
        self._tree.tag_configure("ok", background="#ffffff")
        self._tree.tag_configure("alt", background="#f3f3f3")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self._sort_col: Optional[str] = None
        self._sort_rev = False

    def _build_detail_pane(self, parent: tk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        def _tab(label: str) -> tk.Text:
            frame = tk.Frame(nb, bg=PALETTE["card"])
            nb.add(frame, text=label)
            sb = ttk.Scrollbar(frame)
            sb.pack(side="right", fill="y")
            txt = tk.Text(frame, height=6, wrap="none",
                          font=("Courier", 9), bg=PALETTE["card"],
                          fg=PALETTE["text"], relief="flat",
                          yscrollcommand=sb.set, state="disabled")
            txt.pack(fill="both", expand=True)
            sb.config(command=txt.yview)
            return txt

        self._detail_url = _tab("URL / Meta")
        self._detail_req = _tab("Request body")
        self._detail_res = _tab("Response body")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="No log loaded.")
        bar = tk.Label(self.root, textvariable=self._status_var, anchor="w",
                       bg=PALETTE["border"], fg=PALETTE["text"],
                       font=("Helvetica", 8), padx=8, pady=2)
        bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open LLM API log",
            filetypes=[("JSONL log", "*.jsonl"), ("SQLite database", "*.db *.sqlite"),
                       ("All files", "*")],
        )
        if path:
            self._log_path.set(path)
            self._load()

    def _load(self) -> None:
        path = self._log_path.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please specify a log file path.")
            return
        try:
            self._log = _load_log(path)
            self._entries = self._log.query()
            self._populate_table(self._entries)
            self._update_stats(self._log.summary())
            self._status(f"Loaded {len(self._entries)} entries from '{path}'")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _apply_filters(self) -> None:
        if self._log is None:
            return
        model = self._filter_model.get().strip() or None
        provider = self._filter_provider.get().strip() or None
        status_raw = self._filter_status.get().strip()
        status: Optional[int] = int(status_raw) if status_raw.isdigit() else None
        results = self._log.query(model=model, provider=provider, status_code=status)
        self._entries = results
        self._populate_table(results)
        self._status(f"Filter matched {len(results)} entries")

    def _clear_filters(self) -> None:
        self._filter_model.set("")
        self._filter_provider.set("")
        self._filter_status.set("")
        if self._log:
            self._entries = self._log.query()
            self._populate_table(self._entries)
            self._status(f"Showing all {len(self._entries)} entries")

    def _export_csv(self) -> None:
        if not self._log:
            messagebox.showwarning("No data", "Load a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if path:
            self._log.export_csv(path)
            self._status(f"Exported {self._log.count()} entries to '{path}'")

    def _export_jsonl(self) -> None:
        if not self._log:
            messagebox.showwarning("No data", "Load a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*")],
        )
        if path:
            self._log.export_jsonl(path)
            self._status(f"Exported {self._log.count()} entries to '{path}'")

    def _on_select(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        idx = int(self._tree.item(iid, "values")[0] if False else iid.lstrip("I"))
        # Use the tag we stored as the item id to look up the entry
        entry = self._get_entry_by_iid(iid)
        if entry is None:
            return
        self._show_detail(entry)

    def _get_entry_by_iid(self, iid: str) -> Optional[LogEntry]:
        """Retrieve the LogEntry associated with a Treeview item."""
        try:
            idx = int(iid) - 1
            return self._entries[idx]
        except (ValueError, IndexError):
            return None

    def _show_detail(self, entry: LogEntry) -> None:
        meta = (
            f"ID        : {entry.id}\n"
            f"Timestamp : {entry.timestamp}\n"
            f"URL       : {entry.url}\n"
            f"Method    : {entry.method}\n"
            f"Provider  : {entry.provider}\n"
            f"Model     : {entry.model}\n"
            f"Status    : {entry.status_code}\n"
            f"Latency   : {entry.latency_ms:.2f} ms\n"
            f"Tokens in : {entry.tokens_in}\n"
            f"Tokens out: {entry.tokens_out}\n"
            f"Cost USD  : ${entry.cost_usd:.6f}\n"
        )
        if entry.error:
            meta += f"Error     : {entry.error}\n"
        self._set_text(self._detail_url, meta)
        self._set_text(self._detail_req, self._pretty_json(entry.request_body))
        self._set_text(self._detail_res, self._pretty_json(entry.response_body))

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _populate_table(self, entries: List[LogEntry]) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, e in enumerate(entries, start=1):
            cost = f"${e.cost_usd:.6f}"
            row = (
                e.timestamp, e.provider, e.model, str(e.status_code),
                f"{e.latency_ms:.1f}", str(e.tokens_in), str(e.tokens_out),
                cost, e.error or "",
            )
            tag = "error" if e.error else ("alt" if i % 2 == 0 else "ok")
            # Use 1-based integer as iid for easy reverse-lookup
            self._tree.insert("", "end", iid=str(i), values=row, tags=(tag,))

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        numeric_cols = {"status_code", "latency_ms", "tokens_in", "tokens_out", "cost_usd"}
        rev = self._sort_rev
        if col in numeric_cols:
            self._entries.sort(key=lambda e: getattr(e, col) or 0, reverse=rev)
        else:
            self._entries.sort(key=lambda e: str(getattr(e, col) or ""), reverse=rev)
        self._populate_table(self._entries)

    # ------------------------------------------------------------------
    # Stat cards
    # ------------------------------------------------------------------

    def _update_stats(self, s: dict) -> None:
        for label, key, fmt, _ in STAT_LABELS:
            val = s.get(key, 0)
            self._stat_vars[key].set(fmt.format(val))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text or "")
        widget.configure(state="disabled")

    @staticmethod
    def _pretty_json(raw: Optional[str]) -> str:
        if not raw:
            return "(empty)"
        try:
            return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return raw

    def _about(self) -> None:
        messagebox.showinfo(
            "About LLM API Logger",
            "LLM API Logger v1.0.0\n\n"
            "A tool for logging, analysing, and visualising LLM API calls.\n\n"
            "Supports: OpenAI, Anthropic, Google, Mistral, Cohere, and more.\n\n"
            "https://github.com/vdeshmukh203/llm-api-logger",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the GUI dashboard."""
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    App(root, initial_file=initial)
    root.mainloop()


if __name__ == "__main__":
    main()
