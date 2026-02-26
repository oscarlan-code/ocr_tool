import os
import threading
import queue
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

try:
    from engine import preprocess_pdf
except Exception as exc:
    preprocess_pdf = None
    _ENGINE_ERROR = exc
else:
    _ENGINE_ERROR = None

try:
    from chunk_tool import build_chunks
except Exception as exc:
    build_chunks = None
    _CHUNK_ERROR = exc
else:
    _CHUNK_ERROR = None

try:
    from preview_tool import PreviewApp
except Exception:
    PreviewApp = None


class FullApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OCR Engine + Semantic Chunking")
        self.geometry("1100x760")

        self._queue = queue.Queue()
        self._running = False

        self.pdf_path_var = tk.StringVar(value="")
        self.out_dir_var = tk.StringVar(value="")
        self.ocr_mode_var = tk.StringVar(value="auto")
        self.lang_var = tk.StringVar(value="kor+eng")
        self.dpi_var = tk.StringVar(value="300")
        self.llm_clean_var = tk.BooleanVar(value=False)
        self.prefer_text_layer_var = tk.BooleanVar(value=True)

        self.manifest_var = tk.StringVar(value="")
        self.chunks_var = tk.StringVar(value="")
        self.chunk_size_var = tk.StringVar(value="1000")
        self.chunk_overlap_var = tk.StringVar(value="100")
        self.semantic_threshold_var = tk.StringVar(value="95")
        self.embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        self.text_pref_var = tk.StringVar(value="clean")
        self.llm_media_summary_var = tk.BooleanVar(value=False)
        self.llm_chunk_refine_var = tk.BooleanVar(value=False)
        self.llm_drop_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        preprocess = ttk.LabelFrame(root, text="1) Preprocess Engine")
        preprocess.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preprocess, text="PDF:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(preprocess, textvariable=self.pdf_path_var, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(preprocess, text="Browse", command=self._browse_pdf).grid(row=0, column=2, padx=6)

        ttk.Label(preprocess, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(preprocess, textvariable=self.out_dir_var, width=80).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(preprocess, text="Browse", command=self._browse_out).grid(row=1, column=2, padx=6)

        ttk.Label(preprocess, text="OCR mode:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(preprocess, textvariable=self.ocr_mode_var, values=["auto", "ocr", "text"], width=10, state="readonly").grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(preprocess, text="Lang:").grid(row=2, column=1, sticky="w", padx=(140, 6))
        ttk.Entry(preprocess, textvariable=self.lang_var, width=15).grid(row=2, column=1, sticky="w", padx=(190, 6))
        ttk.Label(preprocess, text="DPI:").grid(row=2, column=1, sticky="w", padx=(320, 6))
        ttk.Entry(preprocess, textvariable=self.dpi_var, width=6).grid(row=2, column=1, sticky="w", padx=(360, 6))
        ttk.Checkbutton(preprocess, text="LLM clean", variable=self.llm_clean_var).grid(row=2, column=2, sticky="w", padx=6)
        ttk.Checkbutton(preprocess, text="Prefer text layer", variable=self.prefer_text_layer_var).grid(row=2, column=3, sticky="w", padx=6)

        ttk.Button(preprocess, text="Run preprocess", command=self._run_preprocess).grid(row=3, column=1, sticky="w", padx=6, pady=6)
        preprocess.columnconfigure(1, weight=1)

        chunk = ttk.LabelFrame(root, text="2) Semantic Chunking")
        chunk.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(chunk, text="Manifest:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(chunk, textvariable=self.manifest_var, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(chunk, text="Browse", command=self._browse_manifest).grid(row=0, column=2, padx=6)

        ttk.Label(chunk, text="Chunk size:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(chunk, textvariable=self.chunk_size_var, width=8).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(chunk, text="Overlap:").grid(row=1, column=1, sticky="w", padx=(120, 6))
        ttk.Entry(chunk, textvariable=self.chunk_overlap_var, width=6).grid(row=1, column=1, sticky="w", padx=(180, 6))
        ttk.Label(chunk, text="Semantic %:").grid(row=1, column=1, sticky="w", padx=(250, 6))
        ttk.Entry(chunk, textvariable=self.semantic_threshold_var, width=6).grid(row=1, column=1, sticky="w", padx=(330, 6))

        ttk.Label(chunk, text="Embedding model:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            chunk,
            textvariable=self.embedding_model_var,
            values=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],
            width=50,
            state="readonly",
        ).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(chunk, text="Text source:").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(chunk, textvariable=self.text_pref_var, values=["raw", "clean", "llm_clean"], width=12, state="readonly").grid(row=3, column=1, sticky="w", padx=6)
        ttk.Checkbutton(chunk, text="LLM media summary", variable=self.llm_media_summary_var).grid(row=3, column=2, sticky="w", padx=6)
        ttk.Checkbutton(chunk, text="LLM refine chunks", variable=self.llm_chunk_refine_var).grid(row=3, column=3, sticky="w", padx=6)
        ttk.Checkbutton(chunk, text="Drop low-quality", variable=self.llm_drop_var).grid(row=3, column=4, sticky="w", padx=6)
        ttk.Button(chunk, text="Run chunk", command=self._run_chunk).grid(row=3, column=5, padx=6)
        chunk.columnconfigure(1, weight=1)

        preview = ttk.LabelFrame(root, text="3) Preview")
        preview.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(preview, text="Chunks file:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(preview, textvariable=self.chunks_var, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(preview, text="Browse", command=self._browse_chunks).grid(row=0, column=2, padx=6)
        ttk.Button(preview, text="Open preview", command=self._open_preview).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        preview.columnconfigure(1, weight=1)

        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, pady=(0, 6))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=6)
        self.progress = ttk.Progressbar(status_frame, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(log_frame, height=12, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True)

    def _browse_pdf(self):
        path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF", "*.pdf")])
        if path:
            self.pdf_path_var.set(path)
            if not self.out_dir_var.get().strip():
                self.out_dir_var.set(os.path.dirname(path))

    def _browse_out(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.out_dir_var.set(path)

    def _browse_manifest(self):
        path = filedialog.askopenfilename(title="Select manifest", filetypes=[("JSON", "*.json")])
        if path:
            self.manifest_var.set(path)

    def _browse_chunks(self):
        path = filedialog.askopenfilename(title="Select chunks", filetypes=[("JSON", "*.json")])
        if path:
            self.chunks_var.set(path)

    def _open_preview(self):
        if PreviewApp is None:
            messagebox.showerror("Preview error", "preview_tool is not available.")
            return
        manifest = self.manifest_var.get().strip()
        chunks = self.chunks_var.get().strip()
        if not manifest:
            messagebox.showwarning("Missing", "Select a manifest first.")
            return
        preview = PreviewApp(manifest, chunks, master=self)
        preview.focus()

    def _run_preprocess(self):
        if _ENGINE_ERROR is not None or preprocess_pdf is None:
            messagebox.showerror("Engine error", str(_ENGINE_ERROR))
            return
        if self._running:
            return
        pdf_path = self.pdf_path_var.get().strip()
        out_dir = self.out_dir_var.get().strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            messagebox.showwarning("Missing", "Select a PDF file.")
            return
        if not out_dir:
            out_dir = os.path.dirname(pdf_path)
            self.out_dir_var.set(out_dir)

        self._running = True
        self._queue.put(("status", "Preprocess starting..."))
        self._queue.put(("progress_mode", "determinate"))
        self._queue.put(("progress", 0, 1))
        self._queue.put(("log", "Starting preprocess..."))
        thread = threading.Thread(target=self._run_preprocess_thread, args=(pdf_path, out_dir), daemon=True)
        thread.start()

    def _run_preprocess_thread(self, pdf_path, out_dir):
        try:
            def on_status(msg):
                self._queue.put(("status", msg))

            def on_page(cur, total):
                self._queue.put(("progress", cur, total))

            manifest = preprocess_pdf(
                pdf_path=pdf_path,
                out_dir=out_dir,
                ocr_mode=self.ocr_mode_var.get(),
                prefer_text_layer=self.prefer_text_layer_var.get(),
                lang=self.lang_var.get().strip() or "kor+eng",
                dpi=int(self.dpi_var.get() or 300),
                llm_clean=self.llm_clean_var.get(),
                on_status=on_status,
                on_page=on_page,
            )
            self._queue.put(("log", f"Manifest created: {manifest}"))
            self._queue.put(("manifest", manifest))
            self._queue.put(("status", "Preprocess done."))
        except Exception as exc:
            self._queue.put(("log", f"Preprocess failed: {exc}"))
            self._queue.put(("status", "Preprocess failed."))
        finally:
            self._queue.put(("done",))

    def _run_chunk(self):
        if _CHUNK_ERROR is not None or build_chunks is None:
            messagebox.showerror("Chunk error", str(_CHUNK_ERROR))
            return
        if self._running:
            return
        manifest = self.manifest_var.get().strip()
        if not manifest or not os.path.isfile(manifest):
            messagebox.showwarning("Missing", "Select a manifest.")
            return
        self._running = True
        self._queue.put(("status", "Chunking..."))
        self._queue.put(("progress_mode", "indeterminate"))
        self._queue.put(("progress_start",))
        self._queue.put(("log", "Starting chunk..."))
        thread = threading.Thread(target=self._run_chunk_thread, args=(manifest,), daemon=True)
        thread.start()

    def _run_chunk_thread(self, manifest):
        try:
            chunk_size = int(self.chunk_size_var.get() or 1000)
            overlap = int(self.chunk_overlap_var.get() or 100)
            semantic_threshold = int(self.semantic_threshold_var.get() or 95)
            model_name = self.embedding_model_var.get().strip() or "sentence-transformers/all-MiniLM-L6-v2"
            text_pref = self.text_pref_var.get().strip() or "clean"
            out_path = build_chunks(
                manifest,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                semantic_threshold=semantic_threshold,
                embedding_model=model_name,
                text_preference=text_pref,
                llm_media_summary=self.llm_media_summary_var.get(),
                llm_chunk_refine=self.llm_chunk_refine_var.get(),
                llm_drop=self.llm_drop_var.get(),
            )
            self._queue.put(("log", f"Chunks saved: {out_path}"))
            self._queue.put(("chunks", out_path))
            self._queue.put(("status", "Chunking done."))
        except Exception as exc:
            self._queue.put(("log", f"Chunk failed: {exc}"))
            self._queue.put(("status", "Chunk failed."))
        finally:
            self._queue.put(("done",))

    def _poll_queue(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "log":
                self.log.insert(tk.END, item[1] + "\n")
                self.log.see(tk.END)
            elif kind == "status":
                self.status_var.set(item[1])
            elif kind == "progress":
                current, total = item[1], item[2]
                self.progress["maximum"] = max(total, 1)
                self.progress["value"] = min(current, total)
            elif kind == "progress_mode":
                self.progress.stop()
                self.progress["mode"] = item[1]
            elif kind == "progress_start":
                self.progress.start(10)
            elif kind == "progress_stop":
                self.progress.stop()
            elif kind == "manifest":
                self.manifest_var.set(item[1])
            elif kind == "chunks":
                self.chunks_var.set(item[1])
            elif kind == "done":
                self._running = False
                self._queue.put(("progress_mode", "determinate"))
                self._queue.put(("progress_stop",))
        self.after(100, self._poll_queue)


def main():
    app = FullApp()
    app.mainloop()


if __name__ == "__main__":
    main()
