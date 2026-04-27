"""
LLM API Logger GUI - tkinter log file viewer.

Launch via the CLI:
    llm-api-logger gui [log_file]

Or directly:
    python llm_api_logger_gui.py [log_file]
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Optional

from llm_api_logger import LLMLogger, LogEntry, _load_jsonl_into


class LLMLoggerGUI:
    """Main window for browsing and inspecting LLM API log files."""

    _COLUMNS = (
        # (column_id,    heading,        width, anchor)
        ("timestamp",   "Timestamp",     175,  tk.W),
        ("provider",    "Provider",       88,  tk.CENTER),
        ("model",       "Model",         160,  tk.W),
        ("status_code", "Status",         55,  tk.CENTER),
        ("latency_ms",  "Latency (ms)",   95,  tk.E),
        ("tokens_in",   "Tokens In",      85,  tk.E),
        ("tokens_out",  "Tokens Out",     85,  tk.E),
        ("cost_usd",    "Cost (USD)",    105,  tk.E),
    )

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM API Logger")
        self.root.geometry("1200x750")
        self.root.minsize(900, 550)

        self._logger: Optional[LLMLogger] = None
        self._entries_by_id: Dict[str, LogEntry] = {}
        self._sort_col: str = "timestamp"
        self._sort_rev: bool = True
        self._auto_refresh_id: Optional[str] = None
        self._current_file: Optional[str] = None

        self._build_menu()
        self._build_toolbar()
        self._build_panes()
        self._build_statusbar()

    # ------------------------------------------------------------------ build

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…",         accelerator="Ctrl+O", command=self.open_file)
        file_menu.add_command(label="Refresh",       accelerator="F5",     command=self.refresh)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…",   command=self.export_csv)
        file_menu.add_command(label="Export JSONL…", command=self.export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Quit",          accelerator="Ctrl+Q", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda _: self.open_file())
        self.root.bind("<F5>",        lambda _: self.refresh())
        self.root.bind("<Control-q>", lambda _: self.root.destroy())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Open File", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Refresh",   command=self.refresh).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(bar, text="Model:").pack(side=tk.LEFT, padx=(0, 2))
        self._model_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self._model_var, width=18).pack(side=tk.LEFT, padx=2)

        ttk.Label(bar, text="Provider:").pack(side=tk.LEFT, padx=(6, 2))
        self._provider_var = tk.StringVar()
        ttk.Combobox(
            bar, textvariable=self._provider_var, width=13,
            values=["", "openai", "anthropic", "google", "mistral",
                    "cohere", "together", "huggingface", "groq", "fireworks", "perplexity"],
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(bar, text="Apply",  command=self.apply_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Clear",  command=self.clear_filter).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="Auto-refresh (10 s)",
            variable=self._auto_var, command=self._toggle_auto_refresh,
        ).pack(side=tk.LEFT, padx=2)

    def _build_panes(self) -> None:
        pw = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── top: entry table ────────────────────────────────────────────────
        top = ttk.LabelFrame(pw, text="Log Entries")
        pw.add(top, weight=3)

        cols = [c[0] for c in self._COLUMNS]
        self._tree = ttk.Treeview(top, columns=cols, show="headings", selectmode="browse")
        for col_id, heading, width, anchor in self._COLUMNS:
            self._tree.heading(col_id, text=heading, command=lambda c=col_id: self._sort_by(c))
            self._tree.column(col_id, width=width, anchor=anchor, minwidth=40)

        vsb = ttk.Scrollbar(top, orient=tk.VERTICAL,   command=self._tree.yview)
        hsb = ttk.Scrollbar(top, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        self._tree.tag_configure("error", background="#ffd0d0")
        self._tree.tag_configure("alt",   background="#f5f5f5")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── bottom: summary + detail ────────────────────────────────────────
        bot = ttk.Frame(pw)
        pw.add(bot, weight=1)

        sum_frame = ttk.LabelFrame(bot, text="Summary", padding=4)
        sum_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self._summary_text = tk.Text(
            sum_frame, height=9, wrap=tk.WORD, state=tk.DISABLED,
            font=("Courier", 10), relief=tk.FLAT, bg="#fafafa",
        )
        self._summary_text.pack(fill=tk.BOTH, expand=True)

        det_frame = ttk.LabelFrame(bot, text="Entry Detail", padding=4)
        det_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._detail_text = tk.Text(
            det_frame, height=9, wrap=tk.WORD, state=tk.DISABLED,
            font=("Courier", 10), relief=tk.FLAT, bg="#fafafa",
        )
        self._detail_text.pack(fill=tk.BOTH, expand=True)

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="Ready – open a log file to begin.")
        ttk.Label(
            self.root, textvariable=self._status_var,
            anchor=tk.W, relief=tk.SUNKEN, padding=(4, 2),
        ).pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------------- file I/O

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Log File",
            filetypes=[
                ("Log files",  "*.jsonl *.db *.sqlite"),
                ("JSONL",      "*.jsonl"),
                ("SQLite",     "*.db *.sqlite"),
                ("All files",  "*"),
            ],
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        try:
            if path.endswith(".jsonl"):
                logger: LLMLogger = LLMLogger(db_path=":memory:", backend="sqlite")
                _load_jsonl_into(logger, path)
            else:
                logger = LLMLogger(db_path=path, backend="sqlite")
            self._logger = logger
            self._current_file = path
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))

    def refresh(self) -> None:
        if self._logger is None:
            return
        # For JSONL, reload from disk to pick up new entries
        if self._current_file and self._current_file.endswith(".jsonl"):
            try:
                fresh = LLMLogger(db_path=":memory:", backend="sqlite")
                _load_jsonl_into(fresh, self._current_file)
                self._logger = fresh
            except Exception:
                pass
        self._populate_tree(self._logger.query())
        self._update_summary()
        self._status(f"Loaded {self._logger.count()} entries  ·  {self._current_file or 'in-memory'}")

    # ---------------------------------------------------------------- filters

    def apply_filter(self) -> None:
        if self._logger is None:
            return
        model    = self._model_var.get().strip()    or None
        provider = self._provider_var.get().strip() or None
        filtered = self._logger.query(model=model, provider=provider)
        self._populate_tree(filtered)
        self._status(f"Filter: {len(filtered)} of {self._logger.count()} entries shown")

    def clear_filter(self) -> None:
        self._model_var.set("")
        self._provider_var.set("")
        if self._logger:
            self._populate_tree(self._logger.query())
            self._status(f"Filter cleared  ·  {self._logger.count()} entries")

    # -------------------------------------------------------------------- tree

    def _populate_tree(self, entries) -> None:
        self._tree.delete(*self._tree.get_children())
        self._entries_by_id = {}
        for idx, entry in enumerate(entries):
            self._entries_by_id[entry.id] = entry
            tag = "error" if entry.error else ("alt" if idx % 2 else "")
            self._tree.insert("", tk.END, iid=entry.id, tags=(tag,), values=(
                entry.timestamp[:19].replace("T", " "),
                entry.provider,
                entry.model,
                entry.status_code,
                f"{entry.latency_ms:.1f}",
                f"{entry.tokens_in:,}",
                f"{entry.tokens_out:,}",
                f"${entry.cost_usd:.6f}",
            ))

    def _sort_by(self, col: str) -> None:
        items = [(self._tree.set(iid, col), iid) for iid in self._tree.get_children()]
        rev = (col == self._sort_col) and not self._sort_rev
        try:
            items.sort(key=lambda x: float(x[0].lstrip("$").replace(",", "")), reverse=rev)
        except ValueError:
            items.sort(reverse=rev)
        for i, (_, iid) in enumerate(items):
            self._tree.move(iid, "", i)
        self._sort_col, self._sort_rev = col, rev

    def _on_select(self, _event) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        entry = self._entries_by_id.get(sel[0])
        if entry is None:
            return

        lines = [
            f"ID:         {entry.id}",
            f"URL:        {entry.url}",
            f"Provider:   {entry.provider}",
            f"Model:      {entry.model}",
            f"Status:     {entry.status_code}",
            f"Timestamp:  {entry.timestamp}",
            f"Latency:    {entry.latency_ms:.2f} ms",
            f"Tokens In:  {entry.tokens_in:,}",
            f"Tokens Out: {entry.tokens_out:,}",
            f"Cost:       ${entry.cost_usd:.8f}",
        ]
        if entry.request_hash:
            lines.append(f"Req SHA256: {entry.request_hash[:24]}…")
        if entry.response_hash:
            lines.append(f"Res SHA256: {entry.response_hash[:24]}…")
        if entry.error:
            lines.append(f"\nError:\n{entry.error}")
        if entry.request_body:
            try:
                snippet = json.dumps(json.loads(entry.request_body), indent=2)[:700]
            except (json.JSONDecodeError, TypeError):
                snippet = entry.request_body[:500]
            lines.append(f"\nRequest:\n{snippet}")

        self._set_text(self._detail_text, "\n".join(lines))

    # ---------------------------------------------------------------- summary

    def _update_summary(self) -> None:
        if self._logger is None:
            return
        s = self._logger.summary()
        lines = [
            f"Total calls:   {s['total_calls']}",
            f"Total cost:    ${s['total_cost_usd']:.4f}",
            f"Tokens in:     {s['total_tokens_in']:,}",
            f"Tokens out:    {s['total_tokens_out']:,}",
            f"Avg latency:   {s['avg_latency_ms']:.1f} ms",
        ]
        if s["calls_by_model"]:
            lines.append("\nBy model:")
            for model, count in sorted(s["calls_by_model"].items()):
                cost = s["cost_by_model"].get(model, 0.0)
                lines.append(f"  {model:<22} {count:>4} calls  ${cost:.4f}")
        self._set_text(self._summary_text, "\n".join(lines))

    # ----------------------------------------------------------------- export

    def export_csv(self) -> None:
        self._export("csv")

    def export_jsonl(self) -> None:
        self._export("jsonl")

    def _export(self, fmt: str) -> None:
        if self._logger is None:
            messagebox.showwarning("No data", "Open a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}"), ("All files", "*")],
        )
        if not path:
            return
        try:
            if fmt == "csv":
                self._logger.export_csv(path)
            else:
                self._logger.export_jsonl(path)
            self._status(f"Exported {self._logger.count()} entries → {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    # ---------------------------------------------------------- auto-refresh

    def _toggle_auto_refresh(self) -> None:
        if self._auto_var.get():
            self._schedule_refresh()
        elif self._auto_refresh_id:
            self.root.after_cancel(self._auto_refresh_id)
            self._auto_refresh_id = None

    def _schedule_refresh(self) -> None:
        self.refresh()
        if self._auto_var.get():
            self._auto_refresh_id = self.root.after(10_000, self._schedule_refresh)

    # --------------------------------------------------------------- helpers

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)

    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)


def launch_gui(log_file: Optional[str] = None) -> None:
    """Start the GUI, optionally pre-loading *log_file*."""
    root = tk.Tk()
    app = LLMLoggerGUI(root)
    if log_file:
        app.load_file(log_file)
    root.mainloop()


if __name__ == "__main__":
    import sys
    launch_gui(sys.argv[1] if len(sys.argv) > 1 else None)
