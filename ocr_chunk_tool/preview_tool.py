import argparse
import json
import os
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import simpledialog

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


class PreviewApp(tk.Toplevel):
    def __init__(self, manifest_path="", chunks_path="", master=None):
        self._owns_root = False
        if master is None:
            master = tk.Tk()
            master.withdraw()
            self._owns_root = True
        self._master_root = master
        super().__init__(master)
        self.title("Chunk Preview")
        self.geometry("1000x720")

        self.manifest = None
        self.chunks = []
        self.media = {}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="Open Manifest", command=self.open_manifest).pack(side=tk.LEFT)
        ttk.Button(top, text="Open Chunks", command=self.open_chunks).pack(side=tk.LEFT, padx=8)

        self.info_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.info_var).pack(side=tk.LEFT, padx=8)

        tools = ttk.Frame(self)
        tools.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(tools, text="Chunk size:").pack(side=tk.LEFT)
        self.chunk_size_var = tk.StringVar(value="1000")
        ttk.Entry(tools, textvariable=self.chunk_size_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(tools, text="Overlap:").pack(side=tk.LEFT)
        self.chunk_overlap_var = tk.StringVar(value="100")
        ttk.Entry(tools, textvariable=self.chunk_overlap_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(tools, text="Semantic %:").pack(side=tk.LEFT, padx=(6, 0))
        self.semantic_threshold_var = tk.StringVar(value="95")
        ttk.Entry(tools, textvariable=self.semantic_threshold_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(tools, text="Model:").pack(side=tk.LEFT)
        self.embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        ttk.Combobox(
            tools,
            textvariable=self.embedding_model_var,
            values=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],
            width=38,
            state="readonly",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(tools, text="Text:").pack(side=tk.LEFT)
        self.text_pref_var = tk.StringVar(value="clean")
        ttk.Combobox(
            tools,
            textvariable=self.text_pref_var,
            values=["raw", "clean", "llm_clean"],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(tools, text="Rechunk", command=self.rechunk).pack(side=tk.LEFT, padx=8)
        ttk.Button(tools, text="Save preset", command=self.save_preset).pack(side=tk.LEFT, padx=6)
        ttk.Button(tools, text="Load preset", command=self.load_preset).pack(side=tk.LEFT, padx=6)

        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.chunk_list = tk.Listbox(main, width=40)
        self.chunk_list.pack(side=tk.LEFT, fill=tk.Y)
        self.chunk_list.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        self.chunk_text = tk.Text(right, height=16, wrap="word")
        self.chunk_text.pack(fill=tk.BOTH, expand=False)

        self.media_list = tk.Listbox(right, height=6)
        self.media_list.pack(fill=tk.X, pady=6)
        self.media_list.bind("<<ListboxSelect>>", self.on_media_select)

        self.image_label = ttk.Label(right)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        if manifest_path:
            self.load_manifest(manifest_path)
        if chunks_path:
            self.load_chunks(chunks_path)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self.destroy()
        finally:
            if self._owns_root and self._master_root is not None:
                try:
                    self._master_root.destroy()
                except Exception:
                    pass

    def open_manifest(self):
        path = filedialog.askopenfilename(title="Select manifest", filetypes=[("JSON", "*.json")])
        if path:
            self.load_manifest(path)

    def open_chunks(self):
        path = filedialog.askopenfilename(title="Select chunks.json", filetypes=[("JSON", "*.json")])
        if path:
            self.load_chunks(path)

    def load_manifest(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self._manifest_path = path
        self.info_var.set(f"Manifest: {os.path.basename(path)}")

    def load_chunks(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        self.chunk_list.delete(0, tk.END)
        for idx, ch in enumerate(self.chunks, start=1):
            text = ch.get("text", "")
            keep = ch.get("llm_keep", True)
            flag = "" if keep else " [DROP]"
            label = f"{idx:04d} | {len(text)} chars{flag}"
            self.chunk_list.insert(tk.END, label)

    def rechunk(self):
        if self.manifest is None:
            return
        try:
            from chunk_tool import build_chunks
        except Exception:
            self.info_var.set("Chunk tool not available")
            return
        try:
            chunk_size = int(self.chunk_size_var.get().strip())
        except Exception:
            chunk_size = 1000
        try:
            overlap = int(self.chunk_overlap_var.get().strip())
        except Exception:
            overlap = 100
        try:
            threshold = int(self.semantic_threshold_var.get().strip())
        except Exception:
            threshold = 95
        model_name = self.embedding_model_var.get().strip() or "sentence-transformers/all-MiniLM-L6-v2"
        text_pref = self.text_pref_var.get().strip() or "clean"
        manifest_path = getattr(self, "_manifest_path", "")
        if not manifest_path:
            self.info_var.set("Open a manifest first.")
            return
        out_path = build_chunks(
            manifest_path,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            text_preference=text_pref,
            semantic_threshold=threshold,
            embedding_model=model_name,
        )
        self.load_chunks(out_path)
        self.info_var.set(f"Rechunked: {os.path.basename(out_path)}")

    def _preset_path(self):
        if not getattr(self, "_manifest_path", ""):
            return ""
        return os.path.join(os.path.dirname(self._manifest_path), "chunk_presets.json")

    def save_preset(self):
        name = simpledialog.askstring("Preset name", "Name this preset:")
        if not name:
            return
        preset = {
            "chunk_size": self.chunk_size_var.get(),
            "chunk_overlap": self.chunk_overlap_var.get(),
            "semantic_threshold": self.semantic_threshold_var.get(),
            "embedding_model": self.embedding_model_var.get(),
            "text_preference": self.text_pref_var.get(),
        }
        path = self._preset_path()
        if not path:
            self.info_var.set("Open a manifest first.")
            return
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[name] = preset
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.info_var.set(f"Preset saved: {name}")

    def load_preset(self):
        path = self._preset_path()
        if not path or not os.path.isfile(path):
            self.info_var.set("No presets found.")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            self.info_var.set("No presets found.")
            return
        name = simpledialog.askstring("Load preset", "Preset name to load:")
        if not name or name not in data:
            return
        preset = data[name]
        self.chunk_size_var.set(str(preset.get("chunk_size", "1000")))
        self.chunk_overlap_var.set(str(preset.get("chunk_overlap", "100")))
        self.semantic_threshold_var.set(str(preset.get("semantic_threshold", "95")))
        self.embedding_model_var.set(preset.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.text_pref_var.set(preset.get("text_preference", "clean"))
        self.info_var.set(f"Preset loaded: {name}")

    def on_select(self, event):
        if not self.chunk_list.curselection():
            return
        idx = self.chunk_list.curselection()[0]
        chunk = self.chunks[idx]
        self.chunk_text.delete("1.0", tk.END)
        raw_text = chunk.get("text", "")
        llm_clean = chunk.get("llm_clean_text", "")
        if llm_clean:
            self.chunk_text.insert(tk.END, raw_text + "\n\nLLM Cleaned:\n" + llm_clean)
        else:
            self.chunk_text.insert(tk.END, raw_text)

        self.media_list.delete(0, tk.END)
        self._current_media = chunk.get("media", [])
        for i, item in enumerate(self._current_media, start=1):
            title = (item.get("title") or "").strip()
            label = f"{i:02d} {item.get('type','')}: {title}" if title else f"{i:02d} {item.get('type','')}"
            self.media_list.insert(tk.END, label)
        if self._current_media:
            self.media_list.selection_set(0)
            self.on_media_select(None)
        else:
            self.image_label.configure(image="", text="(No media)")

    def on_media_select(self, event):
        if not hasattr(self, "_current_media") or not self._current_media:
            return
        if not self.media_list.curselection():
            return
        idx = self.media_list.curselection()[0]
        item = self._current_media[idx]
        img_path = item.get("image_local_path") or item.get("image_path", "")
        if img_path and os.path.isfile(img_path):
            try:
                if Image is not None and ImageTk is not None:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((800, 360))
                    photo = ImageTk.PhotoImage(img, master=self)
                else:
                    photo = tk.PhotoImage(file=img_path, master=self)
                self._photo = photo
                self.image_label.configure(image=photo, text="")
            except Exception as exc:
                print(f"[preview] failed to load image: {img_path} -> {exc}")
                self.image_label.configure(image="", text=f"(Failed to load image: {exc})")
        else:
            if img_path:
                print(f"[preview] missing image path: {img_path}")
            self.image_label.configure(image="", text="(No image)")

        summary = (item.get("summary") or "").strip()
        summary_ctx = (item.get("summary_contextual") or "").strip()
        context = (item.get("context") or "").strip()
        ocr_text = (item.get("ocr_text") or "").strip()
        body = []
        if summary:
            body.append("Summary:\n" + summary)
        if summary_ctx:
            body.append("LLM Summary:\n" + summary_ctx)
        if context:
            body.append("Context:\n" + context)
        if ocr_text:
            body.append("OCR:\n" + ocr_text)
        if body:
            existing = self.chunk_text.get("1.0", tk.END).strip()
            self.chunk_text.delete("1.0", tk.END)
            self.chunk_text.insert(tk.END, existing + "\n\n" + "\n\n".join(body))


def main():
    parser = argparse.ArgumentParser(description="Chunk preview tool")
    parser.add_argument("--manifest", default="", help="Manifest JSON")
    parser.add_argument("--chunks", default="", help="Chunks JSON")
    args = parser.parse_args()
    app = PreviewApp(args.manifest, args.chunks)
    app.mainloop()


if __name__ == "__main__":
    main()
