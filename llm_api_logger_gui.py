"""
llm_api_logger_gui – Tkinter GUI for exploring LLM API log files.

Launch with:
    llm-api-logger-gui [log_file]
"""

import json
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from llm_api_logger import LLMLogger, LogEntry


# ---------------------------------------------------------------------------
# Palette (Tableau-inspired)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


class LLMLoggerGUI:
    """Main application window."""

    def __init__(self, root: tk.Tk, log_file: str = None) -> None:
        self.root = root
        self.logger: LLMLogger = None
        self.entries: list = []
        self._sort_reverse: dict = {}

        self._build_ui()

        if log_file:
            self._load_file(log_file)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("LLM API Logger")
        self.root.geometry("1200x720")
        self.root.minsize(900, 550)

        self._build_menu()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self._build_summary_tab()
        self._build_entries_tab()
        self._build_charts_tab()

        self.status_var = tk.StringVar(value="No file loaded — use File → Open")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O",
                               command=self._open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_command(label="Export JSONL…", command=self._export_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda _e: self._open_file())

    def _build_summary_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Summary  ")

        self.summary_text = tk.Text(
            frame, font=("Courier", 11), state=tk.DISABLED,
            bg="#fafafa", relief=tk.FLAT, padx=12, pady=8,
        )
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                             command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

    def _build_entries_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Log Entries  ")

        # --- filter bar ---
        fbar = ttk.LabelFrame(frame, text="Filter")
        fbar.pack(fill=tk.X, padx=6, pady=(6, 2))

        ttk.Label(fbar, text="Model:").grid(row=0, column=0, padx=(8, 2), pady=4)
        self.model_filter = ttk.Combobox(fbar, width=24, state="readonly")
        self.model_filter.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(fbar, text="Provider:").grid(row=0, column=2, padx=(12, 2))
        self.provider_filter = ttk.Combobox(fbar, width=14, state="readonly")
        self.provider_filter.grid(row=0, column=3, padx=4)

        ttk.Button(fbar, text="Apply",
                   command=self._apply_filter).grid(row=0, column=4, padx=8)
        ttk.Button(fbar, text="Clear",
                   command=self._clear_filter).grid(row=0, column=5, padx=4)

        # --- treeview ---
        cols = ("timestamp", "provider", "model", "status",
                "latency_ms", "tokens_in", "tokens_out", "cost_usd")
        col_cfg = {
            "timestamp":  ("Timestamp",     160, tk.W),
            "provider":   ("Provider",        90, tk.W),
            "model":      ("Model",          170, tk.W),
            "status":     ("Status",          60, tk.CENTER),
            "latency_ms": ("Latency (ms)",   100, tk.E),
            "tokens_in":  ("Tokens In",       90, tk.E),
            "tokens_out": ("Tokens Out",      90, tk.E),
            "cost_usd":   ("Cost (USD)",      90, tk.E),
        }

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                  selectmode="browse")
        for col in cols:
            heading, width, anchor = col_cfg[col]
            self.tree.heading(col, text=heading,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=width, anchor=anchor, stretch=False)

        vsb2 = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                              command=self.tree.yview)
        hsb2 = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL,
                              command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)
        vsb2.pack(side=tk.RIGHT, fill=tk.Y)
        hsb2.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_entry_select)

        # --- detail pane ---
        detail_frame = ttk.LabelFrame(frame, text="Entry Detail")
        detail_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.detail_text = tk.Text(
            detail_frame, height=7, font=("Courier", 9),
            state=tk.DISABLED, bg="#f5f5f5", wrap=tk.WORD,
        )
        dsb = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL,
                             command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=dsb.set)
        dsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.detail_text.pack(fill=tk.BOTH)

    def _build_charts_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Charts  ")

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        self.chart_type = tk.StringVar(value="cost")
        for label, value in [("Cost by Model (USD)", "cost"),
                              ("Calls by Model", "calls"),
                              ("Avg Latency by Model (ms)", "latency")]:
            ttk.Radiobutton(ctrl, text=label, variable=self.chart_type,
                            value=value,
                            command=self._draw_chart).pack(side=tk.LEFT, padx=10)

        self.chart_canvas = tk.Canvas(frame, bg="white")
        self.chart_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.chart_canvas.bind("<Configure>", lambda _e: self._draw_chart())

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Log File",
            filetypes=[
                ("JSONL files", "*.jsonl"),
                ("SQLite databases", "*.db *.sqlite"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            if path.endswith(".jsonl"):
                self.logger = LLMLogger(backend="jsonl")
                p = Path(path)
                if p.exists():
                    with p.open() as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    self.logger.entries.append(
                                        LogEntry.from_dict(json.loads(line))
                                    )
                                except Exception:
                                    pass
            else:
                self.logger = LLMLogger(db_path=path, backend="sqlite")

            self.entries = self.logger.query()
            self._refresh_all()
            n = len(self.entries)
            self.status_var.set(
                f"Loaded {n} {'entry' if n == 1 else 'entries'} from {path}"
            )
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _export_csv(self) -> None:
        if not self.logger:
            messagebox.showwarning("No data", "Open a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            self.logger.export_csv(path)
            messagebox.showinfo("Exported", f"Saved {self.logger.count()} entries to {path}")

    def _export_jsonl(self) -> None:
        if not self.logger:
            messagebox.showwarning("No data", "Open a log file first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl")],
        )
        if path:
            self.logger.export_jsonl(path)
            messagebox.showinfo("Exported", f"Saved {self.logger.count()} entries to {path}")

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        self._refresh_summary()
        self._populate_tree(self.entries)
        self._refresh_filter_options()
        self._draw_chart()

    def _refresh_summary(self) -> None:
        if not self.logger:
            return
        s = self.logger.summary()
        sep = "=" * 62
        lines = [
            sep,
            "  LLM API CALL SUMMARY",
            sep,
            f"  Total API Calls      : {s['total_calls']}",
            f"  Total Cost (USD)     : ${s['total_cost_usd']:.4f}",
            f"  Total Input Tokens   : {s['total_tokens_in']:,}",
            f"  Total Output Tokens  : {s['total_tokens_out']:,}",
            f"  Average Latency (ms) : {s['avg_latency_ms']:.2f}",
            "",
            f"  {'Model':<32} {'Calls':>6}  {'Cost (USD)':>12}",
            "  " + "-" * 54,
        ]
        for model, count in sorted(s["calls_by_model"].items()):
            cost = s["cost_by_model"].get(model, 0.0)
            lines.append(f"  {model:<32} {count:>6}  ${cost:>11.4f}")
        lines.append(sep)
        text = "\n".join(lines)

        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state=tk.DISABLED)

    def _populate_tree(self, entries: list) -> None:
        self.tree.delete(*self.tree.get_children())
        for e in entries:
            self.tree.insert("", tk.END, iid=e.id, values=(
                e.timestamp[:19],
                e.provider,
                e.model,
                e.status_code,
                f"{e.latency_ms:.1f}",
                e.tokens_in,
                e.tokens_out,
                f"${e.cost_usd:.6f}",
            ))

    def _refresh_filter_options(self) -> None:
        models = sorted({e.model for e in self.entries})
        providers = sorted({e.provider for e in self.entries})
        self.model_filter["values"] = ["All"] + models
        self.model_filter.set("All")
        self.provider_filter["values"] = ["All"] + providers
        self.provider_filter.set("All")

    # ------------------------------------------------------------------
    # Filter actions
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        if not self.logger:
            return
        model = self.model_filter.get()
        provider = self.provider_filter.get()
        filtered = self.logger.query(
            model=None if model == "All" else model,
            provider=None if provider == "All" else provider,
        )
        self._populate_tree(filtered)
        self.status_var.set(
            f"Showing {len(filtered)} of {len(self.entries)} entries"
        )

    def _clear_filter(self) -> None:
        self.model_filter.set("All")
        self.provider_filter.set("All")
        self._populate_tree(self.entries)
        self.status_var.set(f"Showing all {len(self.entries)} entries")

    # ------------------------------------------------------------------
    # Treeview interaction
    # ------------------------------------------------------------------

    def _on_entry_select(self, _event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        entry_id = selected[0]
        matches = [e for e in self.entries if e.id == entry_id]
        if not matches:
            return
        e = matches[0]

        parts = [
            f"ID        : {e.id}",
            f"Timestamp : {e.timestamp}",
            f"URL       : {e.url}",
            f"Provider  : {e.provider}   Model: {e.model}",
            f"Status    : {e.status_code}   Latency: {e.latency_ms:.1f} ms",
            f"Tokens    : in={e.tokens_in}  out={e.tokens_out}  cost=${e.cost_usd:.6f}",
        ]
        if e.error:
            parts.append(f"Error     : {e.error}")
        if e.request_body:
            try:
                body = json.dumps(json.loads(e.request_body), indent=2)
            except Exception:
                body = e.request_body
            parts.append(f"\nRequest Body:\n{body[:600]}")

        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "\n".join(parts))
        self.detail_text.config(state=tk.DISABLED)

    def _sort_by(self, col: str) -> None:
        items = [(self.tree.set(child, col), child)
                 for child in self.tree.get_children("")]
        reverse = self._sort_reverse.get(col, False)
        try:
            items.sort(key=lambda x: float(x[0].lstrip("$")), reverse=reverse)
        except ValueError:
            items.sort(key=lambda x: x[0], reverse=reverse)
        for idx, (_, child) in enumerate(items):
            self.tree.move(child, "", idx)
        self._sort_reverse[col] = not reverse

    # ------------------------------------------------------------------
    # Charts (pure Canvas — no matplotlib dependency)
    # ------------------------------------------------------------------

    def _draw_chart(self) -> None:
        self.chart_canvas.delete("all")
        if not self.entries:
            self.chart_canvas.create_text(
                (self.chart_canvas.winfo_width() or 400) // 2,
                (self.chart_canvas.winfo_height() or 300) // 2,
                text="No data loaded", font=("Arial", 14), fill="#999999",
            )
            return

        chart_type = self.chart_type.get()
        data: dict = {}
        for e in self.entries:
            m = e.model or "unknown"
            if m not in data:
                data[m] = {"cost": 0.0, "calls": 0, "lat_total": 0.0}
            data[m]["cost"] += e.cost_usd
            data[m]["calls"] += 1
            data[m]["lat_total"] += e.latency_ms

        if chart_type == "cost":
            values = {m: v["cost"] for m, v in data.items()}
            title = "Cost by Model (USD)"
            fmt = "${:.4f}"
        elif chart_type == "calls":
            values = {m: float(v["calls"]) for m, v in data.items()}
            title = "API Calls by Model"
            fmt = "{:.0f}"
        else:
            values = {m: v["lat_total"] / v["calls"]
                      for m, v in data.items() if v["calls"] > 0}
            title = "Average Latency by Model (ms)"
            fmt = "{:.1f} ms"

        if not values:
            return

        w = self.chart_canvas.winfo_width() or 800
        h = self.chart_canvas.winfo_height() or 500
        margin_l, margin_r, margin_t, margin_b = 210, 160, 50, 20
        bar_h, spacing = 28, 8
        chart_w = max(w - margin_l - margin_r, 100)
        max_val = max(values.values()) or 1.0

        # Title
        self.chart_canvas.create_text(
            w // 2, margin_t // 2, text=title,
            font=("Arial", 13, "bold"), fill="#333333",
        )

        for idx, (model, val) in enumerate(
            sorted(values.items(), key=lambda x: x[1], reverse=True)
        ):
            y = margin_t + idx * (bar_h + spacing)
            if y + bar_h > h - margin_b:
                break

            bar_w = int(chart_w * val / max_val)
            color = _PALETTE[idx % len(_PALETTE)]

            # Model label (right-aligned)
            self.chart_canvas.create_text(
                margin_l - 8, y + bar_h // 2,
                text=model[:28], anchor=tk.E,
                font=("Arial", 10), fill="#222222",
            )
            # Bar
            if bar_w > 0:
                self.chart_canvas.create_rectangle(
                    margin_l, y,
                    margin_l + bar_w, y + bar_h,
                    fill=color, outline="",
                )
            # Value label
            self.chart_canvas.create_text(
                margin_l + bar_w + 6, y + bar_h // 2,
                text=fmt.format(val), anchor=tk.W,
                font=("Arial", 9), fill="#444444",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def gui_main() -> None:
    """Launch the LLM API Logger GUI."""
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    try:
        # Use a built-in theme if available
        style = ttk.Style()
        available = style.theme_names()
        for preferred in ("clam", "alt", "default"):
            if preferred in available:
                style.theme_use(preferred)
                break
    except Exception:
        pass
    LLMLoggerGUI(root, log_file=log_file)
    root.mainloop()


if __name__ == "__main__":
    gui_main()
