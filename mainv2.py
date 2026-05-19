"""
YOLO Real-Time Inference GUI  —  Industrial Edition
====================================================
Requirements:
    pip install ultralytics opencv-python pillow requests

Poppins is downloaded once to ~/.yolo_gui_fonts/ and registered at startup.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import urllib.request
from pathlib import Path

import cv2
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG_DARK   = "#0d0f14"
BG_PANEL  = "#13161e"
BG_CARD   = "#1a1e2a"
BG_ENTRY  = "#0d0f14"
ACCENT    = "#00e5ff"
ACCENT2   = "#a78bfa"
SUCCESS   = "#22c55e"
DANGER    = "#ef4444"
WARNING   = "#f59e0b"
TEXT_PRI  = "#e2e8f0"
TEXT_SEC  = "#64748b"
BORDER    = "#1e2333"
RIVET     = "#2d3348"

# ══════════════════════════════════════════════════════════════════════════════
#  FONT — Poppins (downloaded once, cached locally)
# ══════════════════════════════════════════════════════════════════════════════
FONT_DIR = Path.home() / ".yolo_gui_fonts"
POPPINS_FILES = {
    "Regular":  ("Poppins-Regular.ttf",
                 "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf"),
    "SemiBold": ("Poppins-SemiBold.ttf",
                 "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf"),
    "Bold":     ("Poppins-Bold.ttf",
                 "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"),
}


def _ensure_poppins():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for _, (fname, url) in POPPINS_FILES.items():
        dest = FONT_DIR / fname
        if not dest.exists():
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception:
                ok = False
    return ok


def _register_poppins(root):
    """Try to make Poppins available to Tk; return family name or fallback."""
    # On Linux: symlink into ~/.fonts so fontconfig picks it up
    if os.name != "nt":
        fc_dir = Path.home() / ".fonts"
        fc_dir.mkdir(exist_ok=True)
        for _, (fname, _) in POPPINS_FILES.items():
            src = FONT_DIR / fname
            dst = fc_dir / fname
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except Exception:
                    pass

    # On Windows: use GDI AddFontResourceEx via ctypes
    if os.name == "nt":
        try:
            import ctypes
            for _, (fname, _) in POPPINS_FILES.items():
                p = str(FONT_DIR / fname)
                ctypes.windll.gdi32.AddFontResourceExW(p, 0x10, 0)
        except Exception:
            pass

    # Probe whether Tk can resolve it
    try:
        from tkinter import font as tkfont
        f = tkfont.Font(root=root, family="Poppins", size=10)
        actual = f.actual()["family"]
        if "Poppins" in actual or "poppins" in actual.lower():
            return "Poppins"
    except Exception:
        pass

    return "Segoe UI" if os.name == "nt" else "Helvetica Neue"


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class YOLOApp(tk.Tk):
    def __init__(self, font_family="Poppins"):
        super().__init__()
        self.FF = font_family

        self.title("YOLO // SISTEMA DE VISÃO COMPUTACIONAL")
        self.configure(bg=BG_DARK)
        self.minsize(1120, 700)
        self.resizable(True, True)

        # state
        self.model       = None
        self.model_path  = tk.StringVar(value="best.pt")
        self.cap         = None
        self.running     = False
        self.paused      = False
        self.fps_display = tk.StringVar(value="FPS: --")
        self.det_count   = tk.StringVar(value="DET: 0")
        self.status_var  = tk.StringVar(value="AGUARDANDO")
        self._photo      = None

        # settings
        self.conf_var    = tk.DoubleVar(value=0.25)
        self.iou_var     = tk.DoubleVar(value=0.45)
        self.imgsz_var   = tk.IntVar(value=640)
        self.source_var  = tk.StringVar(value="0")
        self.show_labels = tk.BooleanVar(value=True)
        self.show_conf   = tk.BooleanVar(value=True)
        self.show_boxes  = tk.BooleanVar(value=True)
        self.device_var  = tk.StringVar(value="cpu")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # font helpers
    def F(self, size, weight="normal"):
        return (self.FF, size, weight)
    def FB(self, size):
        return self.F(size, "bold")

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── LEFT ──────────────────────────────────────────────────────────────────

    def _build_left(self):
        left = tk.Frame(self, bg=BG_PANEL, width=300)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)

        # title bar
        tb = tk.Frame(left, bg=ACCENT, height=52)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        inner = tk.Frame(tb, bg=ACCENT)
        inner.pack(side="left", padx=14, pady=8)
        tk.Label(inner, text="UFSC", font=self.FB(12),
                 bg=ACCENT, fg=BG_DARK).pack(anchor="w")
        tk.Label(inner, text="SISTEMA DE INSPEÇÃO DE GARRAFAS",
                 font=self.F(6), bg=ACCENT, fg=BG_DARK).pack(anchor="w")

        # scrollable body
        cv = tk.Canvas(left, bg=BG_PANEL, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner2 = tk.Frame(cv, bg=BG_PANEL)
        wid = cv.create_window((0, 0), window=inner2, anchor="nw")
        inner2.bind("<Configure>",
                    lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))

        self._section_model(inner2)
        self._section_source(inner2)
        self._section_inference(inner2)
        self._section_display(inner2)
        self._section_controls(inner2)
        self._section_stats(inner2)

    def _card(self, parent, label):
        outer = tk.Frame(parent, bg=BG_PANEL)
        outer.pack(fill="x", padx=10, pady=(8, 0))
        hdr = tk.Frame(outer, bg=BG_PANEL)
        hdr.pack(fill="x", pady=(0, 2))
        tk.Frame(hdr, bg=ACCENT, width=3).pack(side="left", fill="y")
        tk.Label(hdr, text=label, font=self.FB(8),
                 bg=BG_PANEL, fg=ACCENT2, padx=8, pady=4).pack(side="left")
        body = tk.Frame(outer, bg=BG_CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        body.pack(fill="x")
        return body

    def _scale_row(self, parent, label, var, lo, hi, fmt="{:.2f}"):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(4, 0))
        top = tk.Frame(row, bg=BG_CARD)
        top.pack(fill="x")
        tk.Label(top, text=label, font=self.F(8),
                 bg=BG_CARD, fg=TEXT_SEC).pack(side="left")
        val = tk.Label(top, text=fmt.format(var.get()),
                       font=self.FB(8), bg=BG_CARD, fg=ACCENT)
        val.pack(side="right")
        def _upd(v, val=val, fmt=fmt):
            try: val.config(text=fmt.format(float(v)))
            except Exception: pass
        ttk.Scale(row, from_=lo, to=hi, variable=var,
                  orient="horizontal", command=_upd).pack(fill="x", pady=2)

    def _section_model(self, p):
        card = self._card(p, "PESOS DO MODELO")
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(8, 4))
        tk.Entry(row, textvariable=self.model_path, bg=BG_ENTRY, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat",
                 font=self.F(8), bd=4).pack(side="left", fill="x", expand=True)
        self._btn(row, "···", self._browse_model, bg=RIVET,
                  fg=TEXT_PRI, w=4).pack(side="right", padx=(4, 0))
        self._btn(card, "CARREGAR MODELO", self._load_model,
                  bg=ACCENT, fg=BG_DARK).pack(fill="x", padx=10, pady=(2, 6))
        dev = tk.Frame(card, bg=BG_CARD)
        dev.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(dev, text="DISPOSITIVO", font=self.F(7),
                 bg=BG_CARD, fg=TEXT_SEC).pack(side="left", padx=(0, 8))
        for d in ("cpu", "cuda", "mps"):
            tk.Radiobutton(dev, text=d.upper(), variable=self.device_var,
                           value=d, bg=BG_CARD, fg=TEXT_PRI,
                           selectcolor=ACCENT, activebackground=BG_CARD,
                           font=self.F(8)).pack(side="left", padx=4)

    def _section_source(self, p):
        card = self._card(p, "FONTE DE VÍDEO")
        tk.Label(card, text="0 = webcam   |   caminho   |   rtsp://",
                 font=self.F(7), bg=BG_CARD,
                 fg=TEXT_SEC).pack(anchor="w", padx=10, pady=(8, 2))
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(row, textvariable=self.source_var, bg=BG_ENTRY, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat",
                 font=self.F(8), bd=4).pack(side="left", fill="x", expand=True)
        self._btn(row, "ARQUIVO", self._browse_video, bg=RIVET,
                  fg=TEXT_PRI, w=7).pack(side="right", padx=(4, 0))

    def _section_inference(self, p):
        card = self._card(p, "CONFIGURAÇÕES DE INFERÊNCIA")
        self._scale_row(card, "Limiar de Confiança", self.conf_var, 0.01, 1.0)
        self._scale_row(card, "Limiar de IoU",       self.iou_var,  0.01, 1.0)
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(6, 10))
        tk.Label(row, text="TAMANHO DA IMAGEM", font=self.F(7),
                 bg=BG_CARD, fg=TEXT_SEC).pack(side="left", padx=(0, 6))
        for sz in (320, 416, 640, 1280):
            tk.Radiobutton(row, text=str(sz), variable=self.imgsz_var,
                           value=sz, bg=BG_CARD, fg=TEXT_PRI,
                           selectcolor=ACCENT, activebackground=BG_CARD,
                           font=self.F(8)).pack(side="left", padx=3)

    def _section_display(self, p):
        card = self._card(p, "OPÇÕES DE EXIBIÇÃO")
        for lbl, var in [("Caixas delimitadoras", self.show_boxes),
                          ("Rótulos de classe",    self.show_labels),
                          ("Confiança",            self.show_conf)]:
            tk.Checkbutton(card, text=lbl, variable=var,
                           bg=BG_CARD, fg=TEXT_PRI, selectcolor=ACCENT,
                           activebackground=BG_CARD, activeforeground=TEXT_PRI,
                           font=self.F(8)).pack(anchor="w", padx=12, pady=2)
        tk.Frame(card, bg=BG_CARD, height=6).pack()

    def _section_controls(self, p):
        card = self._card(p, "CONTROLES")
        tk.Frame(card, bg=BG_CARD, height=4).pack()
        self.btn_start = self._btn(card, "▶  INICIAR", self._start_inference,
                                   bg=SUCCESS, fg=BG_DARK)
        self.btn_start.pack(fill="x", padx=10, pady=3)
        self.btn_pause = self._btn(card, "⏸  PAUSAR", self._toggle_pause,
                                   bg=WARNING, fg=BG_DARK)
        self.btn_pause.pack(fill="x", padx=10, pady=3)
        self.btn_pause.config(state="disabled")
        self.btn_stop = self._btn(card, "■  PARAR", self._stop_inference,
                                  bg=DANGER, fg=TEXT_PRI)
        self.btn_stop.pack(fill="x", padx=10, pady=3)
        self.btn_stop.config(state="disabled")
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)
        self._btn(card, "📷  CAPTURAR TELA", self._save_snapshot,
                  bg=RIVET, fg=TEXT_PRI).pack(fill="x", padx=10, pady=(0, 10))

    def _section_stats(self, p):
        card = self._card(p, "TELEMETRIA")
        for key, var, color in [
            ("FPS",    self.fps_display, ACCENT),
            ("DET",    self.det_count,   WARNING),
            ("STATUS", self.status_var,  SUCCESS),
        ]:
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=5)
            tk.Label(row, text=key, font=self.FB(7), bg=BG_CARD,
                     fg=TEXT_SEC, width=7, anchor="w").pack(side="left")
            tk.Frame(row, bg=BORDER, width=1).pack(side="left", fill="y", padx=4)
            tk.Label(row, textvariable=var, font=self.FB(9),
                     bg=BG_CARD, fg=color, anchor="w").pack(
                     side="left", fill="x", expand=True)
        tk.Frame(card, bg=BG_CARD, height=8).pack()
        tk.Frame(p, bg=BG_PANEL, height=16).pack()

    # ── RIGHT ─────────────────────────────────────────────────────────────────

    def _build_right(self):
        right = tk.Frame(self, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        bar = tk.Frame(right, bg=BG_PANEL, height=44)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        tk.Label(bar, text="ZONA DE INFERÊNCIA", font=self.FB(10),
                 bg=BG_PANEL, fg=TEXT_PRI).pack(side="left", padx=16, pady=10)
        for var, bg, fg in [(self.status_var,  ACCENT,   BG_DARK),
                             (self.det_count,   BG_CARD,  WARNING),
                             (self.fps_display, BG_CARD,  ACCENT)]:
            tk.Label(bar, textvariable=var, font=self.FB(8),
                     bg=bg, fg=fg, padx=10, pady=4).pack(
                     side="right", padx=(0, 8), pady=8)

        self.feed_canvas = tk.Canvas(right, bg="#111111",
                                     highlightthickness=2,
                                     highlightbackground=BORDER)
        self.feed_canvas.grid(row=1, column=0, sticky="nsew",
                              padx=12, pady=12)
        self.feed_canvas.bind("<Configure>", lambda e: self._draw_placeholder())
        self._draw_placeholder()

    def _draw_placeholder(self):
        c = self.feed_canvas
        c.update_idletasks()
        w = c.winfo_width() or 720
        h = c.winfo_height() or 520
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill="#111111", outline="")
        for x in range(0, w, 48):
            c.create_line(x, 0, x, h, fill="#1c1c1c")
        for y in range(0, h, 48):
            c.create_line(0, y, w, y, fill="#1c1c1c")
        cx, cy = w // 2, h // 2
        bl = 30
        for ox, oy, sx, sy in [(cx-100, cy-60, 1, 1), (cx+100, cy-60, -1, 1),
                                (cx-100, cy+60, 1,-1), (cx+100, cy+60, -1,-1)]:
            c.create_line(ox, oy, ox+sx*bl, oy, fill=ACCENT, width=2)
            c.create_line(ox, oy, ox, oy+sy*bl, fill=ACCENT, width=2)
        c.create_line(cx-14, cy, cx+14, cy, fill=ACCENT2)
        c.create_line(cx, cy-14, cx, cy+14, fill=ACCENT2)
        c.create_oval(cx-5, cy-5, cx+5, cy+5, outline=ACCENT2)
        c.create_text(cx, cy+32, text="SEM SINAL",
                      font=self.FB(14), fill=TEXT_SEC)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, bg=BG_CARD, fg=TEXT_PRI, w=None):
        kw = dict(text=text, command=cmd, bg=bg, fg=fg, font=self.FB(8),
                  relief="flat", cursor="hand2", pady=7, bd=0,
                  activebackground=self._lighten(bg), activeforeground=fg)
        if w: kw["width"] = w
        b = tk.Button(parent, **kw)
        b.bind("<Enter>", lambda e, b=b, c=bg: b.config(bg=self._lighten(c)))
        b.bind("<Leave>", lambda e, b=b, c=bg: b.config(bg=c))
        return b

    @staticmethod
    def _lighten(hx, f=0.15):
        hx = hx.lstrip("#")
        r, g, b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        return "#{:02x}{:02x}{:02x}".format(
            min(255,int(r+(255-r)*f)),
            min(255,int(g+(255-g)*f)),
            min(255,int(b+(255-b)*f)))

    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="Select YOLO weights",
            filetypes=[("YOLO weights","*.pt *.onnx *.torchscript"),
                       ("All files","*.*")])
        if p: self.model_path.set(p)

    def _browse_video(self):
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files","*.*")])
        if p: self.source_var.set(p)

    def _load_model(self):
        if not YOLO_AVAILABLE:
            messagebox.showerror("Pacote ausente",
                                 "pip install ultralytics"); return
        path = self.model_path.get().strip()
        self.status_var.set("CARREGANDO…"); self.update_idletasks()
        try:
            self.model = YOLO(path)
            self.status_var.set("MODELO OK")
        except Exception as e:
            messagebox.showerror("Erro no modelo", str(e))
            self.status_var.set("ERRO")

    def _start_inference(self):
        if not YOLO_AVAILABLE:
            messagebox.showerror("Pacote ausente",
                                 "pip install ultralytics"); return
        if self.model is None:
            messagebox.showwarning("Sem modelo", "Carregue um modelo primeiro."); return
        if self.running: return
        src = self.source_var.get().strip()
        try: src = int(src)
        except ValueError: pass
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            messagebox.showerror("Erro na fonte", f"Não foi possível abrir: {src}"); return
        self.running = True; self.paused = False
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.status_var.set("EXECUTANDO")
        threading.Thread(target=self._loop, daemon=True).start()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶  RETOMAR"); self.status_var.set("PAUSADO")
        else:
            self.btn_pause.config(text="⏸  PAUSAR"); self.status_var.set("EXECUTANDO")

    def _stop_inference(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸  PAUSAR")
        self.btn_stop.config(state="disabled")
        self.fps_display.set("FPS: --"); self.det_count.set("DET: 0")
        self.status_var.set("AGUARDANDO")
        self.after(120, self._draw_placeholder)

    def _loop(self):
        while self.running:
            if self.paused: time.sleep(0.04); continue
            ret, frame = self.cap.read()
            if not ret: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            t0 = time.time()
            results = self.model.predict(
                source=frame, conf=self.conf_var.get(), iou=self.iou_var.get(),
                imgsz=self.imgsz_var.get(), device=self.device_var.get(),
                verbose=False)
            fps = 1.0 / max(time.time()-t0, 1e-6)
            ann = self._annotate(frame, results[0])
            self._update_feed(ann, fps, len(results[0].boxes))

    def _annotate(self, frame, result):
        if not self.show_boxes.get(): return frame.copy()
        out = frame.copy()
        for box in result.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]); cls = int(box.cls[0])
            color = self._cls_color(cls)
            cv2.rectangle(out,(x1,y1),(x2,y2),color,1)
            cl=10
            for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out,(px,py),(px+dx*cl,py),color,3)
                cv2.line(out,(px,py),(px,py+dy*cl),color,3)
            parts=[]
            if self.show_labels.get():
                parts.append((result.names or {}).get(cls,str(cls)).upper())
            if self.show_conf.get(): parts.append(f"{conf:.2f}")
            text="  ".join(parts)
            if text:
                (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,0.45,1)
                cv2.rectangle(out,(x1,y1-th-8),(x1+tw+6,y1),color,-1)
                cv2.putText(out,text,(x1+3,y1-4),
                            cv2.FONT_HERSHEY_DUPLEX,0.45,(20,20,20),1,cv2.LINE_AA)
        return out

    @staticmethod
    def _cls_color(cls_id):
        pal=[(232,119,34),(192,192,192),(240,192,64),(192,57,43),
             (141,182,0),(180,140,60),(100,160,200),(200,100,50),(160,160,80)]
        return pal[cls_id % len(pal)]

    def _update_feed(self, frame, fps, n_det):
        c = self.feed_canvas
        c.update_idletasks()
        cw,ch = c.winfo_width(), c.winfo_height()
        if cw<2 or ch<2: return
        fh,fw = frame.shape[:2]
        scale = min(cw/fw, ch/fh)
        nw,nh = int(fw*scale), int(fh*scale)
        rgb = cv2.cvtColor(cv2.resize(frame,(nw,nh)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._photo = photo
        x0,y0 = (cw-nw)//2, (ch-nh)//2
        c.delete("all")
        c.create_rectangle(0,0,cw,ch,fill="#111111",outline="")
        c.create_image(x0,y0,anchor="nw",image=photo)
        self.fps_display.set(f"FPS: {fps:.1f}")
        self.det_count.set(f"DET: {n_det}")

    def _save_snapshot(self):
        if not self._photo:
            messagebox.showinfo("Captura","Nenhum frame ativo."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path:
            try:
                self._photo._PhotoImage__photo.write(path,format="png")
                messagebox.showinfo("Salvo",path)
            except Exception as e:
                messagebox.showerror("Erro ao salvar",str(e))

    def _on_close(self):
        self._stop_inference(); self.destroy()


# ══════════════════════════════════════════════════════════════════════════════

def apply_style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TScale", background=BG_CARD, troughcolor=BG_ENTRY,
                slidercolor=ACCENT, sliderlength=16, sliderthickness=16,
                bordercolor=BG_CARD, darkcolor=BG_CARD, lightcolor=BG_CARD)
    s.map("TScale", slidercolor=[("active", WARNING)])
    s.configure("TScrollbar", background=BG_PANEL, troughcolor=BG_DARK,
                arrowcolor=TEXT_SEC, bordercolor=BG_PANEL,
                darkcolor=BG_PANEL, lightcolor=BG_PANEL)


if __name__ == "__main__":
    poppins_ok = _ensure_poppins()
    app = YOLOApp()
    ff = _register_poppins(app) if poppins_ok else (
        "Segoe UI" if os.name == "nt" else "Helvetica Neue")
    app.FF = ff
    apply_style()
    app.mainloop()