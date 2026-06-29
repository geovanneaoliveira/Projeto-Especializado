"""
YOLO Real-Time Inference GUI  —  Industrial Edition
====================================================
Requirements:
    pip install ultralytics opencv-python pillow requests pyserial

Poppins is downloaded once to ~/.yolo_gui_fonts/ and registered at startup.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import queue
import urllib.request
from pathlib import Path

import cv2
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


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
#  FONT — Poppins
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
    if os.name != "nt":
        fc_dir = Path.home() / ".fonts"
        fc_dir.mkdir(exist_ok=True)
        for _, (fname, _) in POPPINS_FILES.items():
            src = FONT_DIR / fname
            dst = fc_dir / fname
            if src.exists() and not dst.exists():
                try: dst.symlink_to(src)
                except Exception: pass

    if os.name == "nt":
        try:
            import ctypes
            for _, (fname, _) in POPPINS_FILES.items():
                p = str(FONT_DIR / fname)
                ctypes.windll.gdi32.AddFontResourceExW(p, 0x10, 0)
        except Exception: pass

    try:
        from tkinter import font as tkfont
        f = tkfont.Font(root=root, family="Poppins", size=10)
        actual = f.actual()["family"]
        if "Poppins" in actual or "poppins" in actual.lower():
            return "Poppins"
    except Exception: pass

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
        self.minsize(1120, 780)
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
        self._frame_queue = queue.Queue(maxsize=1)
        self.source_is_camera = False
        # Cache de dimensão do canvas (atualizado via bind <Configure>)
        self._canvas_w = 1
        self._canvas_h = 1
        # Cache de parâmetros estáticos de inferência
        self._cached_imgsz  = 640
        self._cached_device = "cpu"

        # Serial State
        self.serial_port = None
        self.serial_connected = False
        self.last_trigger_time = 0
        self.trigger_cooldown = 2.0

        # ── LÓGICA DE DISPARO ROBUSTA ─────────────────────────────────────────
        self._insp_state = "IDLE"
        self._defect_name = ""
        self._consecutive_empty = 0

        self.BLINK_TOLERANCE = 4
        self.CONFIRM_EXIT = 6
        self.ENTRY_CONFIRM = 6
        self._entry_frames = 0

        # ── CONFIDENCE-LOCK TRACKING ──────────────────────────────────────────
        self.LOCK_THRESHOLD     = 0.80
        self.LOCK_RELEASE_FRAMES = 10
        self.LOCK_IOU_MIN       = 0.15
        self._locks = []

        # ── LINHA DE DISPARO (TRIP LINE) ──────────────────────────────────────
        self.trip_line_ratio  = 0.85     # 0.0 (esquerda) … 1.0 (direita)
        self.TRIP_DELAY_MS    = 1000     # ms de espera após cruzar a linha
        self._tripped_locks   = set()    # IDs de locks que já dispararam
        self._pending_trips   = {}       # id → timestamp (s) quando cruzou
        
        # ── LOGS DE CAPTURA (NOVO) ────────────────────────────────────────────
        self.log_dir = "erros_log"
        os.makedirs(self.log_dir, exist_ok=True) # Cria a pasta automaticamente
        self._snapped_locks = set()

        # settings
        self.conf_var    = tk.DoubleVar(value=0.25)
        self.iou_var     = tk.DoubleVar(value=0.45)
        self.imgsz_var   = tk.IntVar(value=640)
        self.source_var  = tk.StringVar(value="0")
        self.show_labels = tk.BooleanVar(value=True)
        self.show_conf   = tk.BooleanVar(value=True)
        self.show_boxes  = tk.BooleanVar(value=True)
        self.device_var  = tk.StringVar(value="cpu")

        # Dimensões reais do frame capturado (atualizadas em _loop)
        self._frame_h = 480
        self._frame_w = 640

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _reset_inspection_state(self):
        """Zera a máquina de estados da inspeção e libera todos os confidence-locks."""
        self._insp_state = "IDLE"
        self._defect_name = ""
        self._consecutive_empty = 0
        self._entry_frames = 0
        self._locks = []
        self._tripped_locks = set()
        self._pending_trips = {}
        self._snapped_locks = set() # Limpa o histórico de capturas da tela

    def F(self, size, weight="normal"):
        return (self.FF, size, weight)
    
    def FB(self, size):
        return self.F(size, "bold")

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    def _build_left(self):
        left = tk.Frame(self, bg=BG_PANEL, width=300)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)

        tb = tk.Frame(left, bg=ACCENT, height=52)
        tb.pack(fill="x")
        tb.pack_propagate(False)
        inner = tk.Frame(tb, bg=ACCENT)
        inner.pack(side="left", padx=14, pady=8)
        tk.Label(inner, text="UFSC", font=self.FB(12),
                 bg=ACCENT, fg=BG_DARK).pack(anchor="w")
        tk.Label(inner, text="SISTEMA DE INSPEÇÃO DE GARRAFAS",
                 font=self.F(6), bg=ACCENT, fg=BG_DARK).pack(anchor="w")

        cv = tk.Canvas(left, bg=BG_PANEL, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner2 = tk.Frame(cv, bg=BG_PANEL)
        wid = cv.create_window((0, 0), window=inner2, anchor="nw")
        inner2.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))

        self._section_model(inner2)
        self._section_serial(inner2)
        self._section_source(inner2)
        self._section_inference(inner2)
        self._section_display(inner2)
        self._section_tripline(inner2)
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
        tk.Label(top, text=label, font=self.F(8), bg=BG_CARD, fg=TEXT_SEC).pack(side="left")
        val = tk.Label(top, text=fmt.format(var.get()), font=self.FB(8), bg=BG_CARD, fg=ACCENT)
        val.pack(side="right")
        def _upd(v, val=val, fmt=fmt):
            try: val.config(text=fmt.format(float(v)))
            except Exception: pass
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal", command=_upd).pack(fill="x", pady=2)

    def _section_model(self, p):
        card = self._card(p, "PESOS DO MODELO")
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(8, 4))
        tk.Entry(row, textvariable=self.model_path, bg=BG_ENTRY, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat", font=self.F(8), bd=4).pack(side="left", fill="x", expand=True)
        self._btn(row, "···", self._browse_model, bg=RIVET, fg=TEXT_PRI, w=4).pack(side="right", padx=(4, 0))
        self._btn(card, "CARREGAR MODELO", self._load_model, bg=ACCENT, fg=BG_DARK).pack(fill="x", padx=10, pady=(2, 6))
        dev = tk.Frame(card, bg=BG_CARD)
        dev.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(dev, text="DISPOSITIVO", font=self.F(7), bg=BG_CARD, fg=TEXT_SEC).pack(side="left", padx=(0, 8))
        for d in ("cpu", "cuda", "mps"):
            tk.Radiobutton(dev, text=d.upper(), variable=self.device_var, value=d, bg=BG_CARD, fg=TEXT_PRI,
                           selectcolor=ACCENT, activebackground=BG_CARD, font=self.F(8)).pack(side="left", padx=4)

    def _section_serial(self, p):
        card = self._card(p, "COMUNICAÇÃO SERIAL")

        row1 = tk.Frame(card, bg=BG_CARD)
        row1.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(row1, text="PORTA (ex: 5):", font=self.F(7), bg=BG_CARD, fg=TEXT_SEC).pack(side="left")
        self.port_var = tk.StringVar(value="5")
        tk.Entry(row1, textvariable=self.port_var, bg=BG_ENTRY, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat",
                 font=self.F(8), bd=4, width=6).pack(side="left", padx=(4, 10))
        self.btn_serial = self._btn(row1, "CONECTAR", self._toggle_serial, bg=ACCENT, fg=BG_DARK)
        self.btn_serial.pack(side="right", fill="x", expand=True)

        if not SERIAL_AVAILABLE:
            self.port_var.set("ERRO")
            self.btn_serial.config(state="disabled")

        tk.Label(card, text="TESTE MANUAL DE COMANDOS:", font=self.F(7), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=10, pady=(6, 0))
        row2 = tk.Frame(card, bg=BG_CARD)
        row2.pack(fill="x", padx=10, pady=(2, 10))
        self._btn(row2, "LIGA (a)", lambda: self._send_manual_cmd('a'), bg=RIVET, fg=TEXT_PRI).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self._btn(row2, "PARA (s)", lambda: self._send_manual_cmd('s'), bg=RIVET, fg=TEXT_PRI).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row2, "EIXO X (d)", lambda: self._send_manual_cmd('d'), bg=WARNING, fg=BG_DARK).pack(side="left", expand=True, fill="x", padx=(2, 0))

    def _toggle_serial(self):
        if self.serial_connected:
            if self.serial_port:
                self.serial_port.close()
            self.serial_connected = False
            self.btn_serial.config(text="CONECTAR", bg=ACCENT, fg=BG_DARK)
            print("[GUI] Porta serial desconectada.")
        else:
            port_input = self.port_var.get().strip()
            if not port_input:
                messagebox.showwarning("Porta Serial", "Digite o número ou nome da porta.")
                return
            if port_input.isdigit() and os.name == 'nt':
                port = f"COM{port_input}"
            else:
                port = port_input
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                self.serial_connected = True
                self.btn_serial.config(text="DESCONECTAR", bg=DANGER, fg=TEXT_PRI)
                print(f"[GUI] Conectado com sucesso na porta {port} a 115200 baud.")
            except Exception as e:
                messagebox.showerror("Erro Serial", f"Falha ao conectar na porta {port}:\n\n{e}")

    def _send_manual_cmd(self, char_cmd):
        if not self.serial_connected or not self.serial_port:
            messagebox.showinfo("Aviso", "Conecte a porta serial primeiro!")
            return
        try:
            self.serial_port.write(char_cmd.encode('utf-8'))
            print(f"[SERIAL MANUAL] Comando '{char_cmd}' enviado ao Arduino!")
        except Exception as e:
            print(f"[ERRO SERIAL] Falha ao enviar o comando manual: {e}")
            messagebox.showerror("Erro", f"A conexão caiu?\n{e}")

    def _section_source(self, p):
        card = self._card(p, "FONTE DE VÍDEO")
        tk.Label(card, text="0 = webcam  |  caminho  |  rtsp://",
                 font=self.F(7), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", padx=10, pady=(8, 2))
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(row, textvariable=self.source_var, bg=BG_ENTRY, fg=TEXT_PRI,
                 insertbackground=ACCENT, relief="flat", font=self.F(8), bd=4).pack(side="left", fill="x", expand=True)
        self._btn(row, "ARQUIVO", self._browse_video, bg=RIVET, fg=TEXT_PRI, w=7).pack(side="right", padx=(4, 0))

    def _section_inference(self, p):
        card = self._card(p, "CONFIGURAÇÕES DE INFERÊNCIA")
        self._scale_row(card, "Limiar de Confiança", self.conf_var, 0.01, 1.0)
        self._scale_row(card, "Limiar de IoU",       self.iou_var,  0.01, 1.0)

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=10, pady=(6, 0))
        self.lock_var = tk.DoubleVar(value=self.LOCK_THRESHOLD)
        def _on_lock_change(v):
            try:
                self.LOCK_THRESHOLD = float(v)
            except Exception:
                pass
        self._scale_row(card, "Limiar de Lock (conf ≥ → trava)", self.lock_var, 0.50, 1.0)
        self.lock_var.trace_add("write", lambda *_: _on_lock_change(self.lock_var.get()))

        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(6, 10))
        tk.Label(row, text="TAMANHO DA IMAGEM", font=self.F(7), bg=BG_CARD, fg=TEXT_SEC).pack(side="left", padx=(0, 6))
        for sz in (320, 416, 640, 1280):
            tk.Radiobutton(row, text=str(sz), variable=self.imgsz_var, value=sz, bg=BG_CARD, fg=TEXT_PRI,
                           selectcolor=ACCENT, activebackground=BG_CARD, font=self.F(8)).pack(side="left", padx=3)

    def _section_display(self, p):
        card = self._card(p, "OPÇÕES DE EXIBIÇÃO")
        for lbl, var in [("Caixas delimitadoras", self.show_boxes), ("Rótulos de classe", self.show_labels), ("Confiança", self.show_conf)]:
            tk.Checkbutton(card, text=lbl, variable=var, bg=BG_CARD, fg=TEXT_PRI, selectcolor=ACCENT,
                           activebackground=BG_CARD, activeforeground=TEXT_PRI, font=self.F(8)).pack(anchor="w", padx=12, pady=2)
        tk.Frame(card, bg=BG_CARD, height=6).pack()

    def _section_tripline(self, p):
        card = self._card(p, "LINHA DE DISPARO (TRIP LINE)")

        tk.Label(card,
                 text="Quando o lock cruzar a linha → aguarda delay → dispara 'd'",
                 font=self.F(7), bg=BG_CARD, fg=TEXT_SEC,
                 wraplength=260, justify="left").pack(anchor="w", padx=10, pady=(8, 4))

        self.trip_line_var = tk.DoubleVar(value=self.trip_line_ratio)
        def _on_trip_change(v):
            try: self.trip_line_ratio = float(v)
            except Exception: pass
        self._scale_row(card, "Posição da linha (0=esquerda  1=direita)",
                        self.trip_line_var, 0.10, 1, fmt="{:.2f}")
        self.trip_line_var.trace_add("write", lambda *_: _on_trip_change(self.trip_line_var.get()))

        delay_row = tk.Frame(card, bg=BG_CARD)
        delay_row.pack(fill="x", padx=10, pady=(6, 4))
        tk.Label(delay_row, text="Delay após cruzar (ms):", font=self.F(7),
                 bg=BG_CARD, fg=TEXT_SEC).pack(side="left")
        self.trip_delay_var = tk.IntVar(value=self.TRIP_DELAY_MS)
        spin = tk.Spinbox(delay_row, from_=0, to=10000, increment=100,
                          textvariable=self.trip_delay_var,
                          bg=BG_ENTRY, fg=TEXT_PRI, insertbackground=ACCENT,
                          buttonbackground=RIVET, relief="flat",
                          font=self.F(8), width=7)
        spin.pack(side="right")
        def _sync_delay(*_):
            try: self.TRIP_DELAY_MS = self.trip_delay_var.get()
            except Exception: pass
        self.trip_delay_var.trace_add("write", _sync_delay)

        tk.Frame(card, bg=BG_CARD, height=8).pack()

    def _section_controls(self, p):
        card = self._card(p, "CONTROLES")
        tk.Frame(card, bg=BG_CARD, height=4).pack()
        self.btn_start = self._btn(card, "▶  INICIAR", self._start_inference, bg=SUCCESS, fg=BG_DARK)
        self.btn_start.pack(fill="x", padx=10, pady=3)
        self.btn_pause = self._btn(card, "⏸  PAUSAR", self._toggle_pause, bg=WARNING, fg=BG_DARK)
        self.btn_pause.pack(fill="x", padx=10, pady=3)
        self.btn_pause.config(state="disabled")
        self.btn_stop = self._btn(card, "■  PARAR", self._stop_inference, bg=DANGER, fg=TEXT_PRI)
        self.btn_stop.pack(fill="x", padx=10, pady=3)
        self.btn_stop.config(state="disabled")
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)
        self._btn(card, "📷  CAPTURAR TELA", self._save_snapshot, bg=RIVET, fg=TEXT_PRI).pack(fill="x", padx=10, pady=(0, 10))

    def _section_stats(self, p):
        card = self._card(p, "TELEMETRIA")
        for key, var, color in [("FPS", self.fps_display, ACCENT), ("DET", self.det_count, WARNING), ("STATUS", self.status_var, SUCCESS)]:
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=5)
            tk.Label(row, text=key, font=self.FB(7), bg=BG_CARD, fg=TEXT_SEC, width=7, anchor="w").pack(side="left")
            tk.Frame(row, bg=BORDER, width=1).pack(side="left", fill="y", padx=4)
            tk.Label(row, textvariable=var, font=self.FB(9), bg=BG_CARD, fg=color, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Frame(card, bg=BG_CARD, height=8).pack()
        tk.Frame(p, bg=BG_PANEL, height=16).pack()

    def _build_right(self):
        right = tk.Frame(self, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        bar = tk.Frame(right, bg=BG_PANEL, height=44)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        tk.Label(bar, text="ZONA DE INFERÊNCIA", font=self.FB(10), bg=BG_PANEL, fg=TEXT_PRI).pack(side="left", padx=16, pady=10)
        for var, bg, fg in [(self.status_var, ACCENT, BG_DARK), (self.det_count, BG_CARD, WARNING), (self.fps_display, BG_CARD, ACCENT)]:
            tk.Label(bar, textvariable=var, font=self.FB(8), bg=bg, fg=fg, padx=10, pady=4).pack(side="right", padx=(0, 8), pady=8)

        self.feed_canvas = tk.Canvas(right, bg="#111111", highlightthickness=2, highlightbackground=BORDER)
        self.feed_canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self.feed_canvas.bind("<Configure>", lambda e: self._draw_placeholder())
        self._draw_placeholder()

    def _on_canvas_resize(self, event):
        self._canvas_w = event.width
        self._canvas_h = event.height

    def _draw_placeholder(self):
        c = self.feed_canvas
        c.update_idletasks()
        w = c.winfo_width() or 720
        h = c.winfo_height() or 520
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill="#111111", outline="")
        for x in range(0, w, 48): c.create_line(x, 0, x, h, fill="#1c1c1c")
        for y in range(0, h, 48): c.create_line(0, y, w, y, fill="#1c1c1c")
        cx, cy = w // 2, h // 2
        bl = 30
        for ox, oy, sx, sy in [(cx-100, cy-60, 1, 1), (cx+100, cy-60, -1, 1), (cx-100, cy+60, 1,-1), (cx+100, cy+60, -1,-1)]:
            c.create_line(ox, oy, ox+sx*bl, oy, fill=ACCENT, width=2)
            c.create_line(ox, oy, ox, oy+sy*bl, fill=ACCENT, width=2)
        c.create_line(cx-14, cy, cx+14, cy, fill=ACCENT2)
        c.create_line(cx, cy-14, cx, cy+14, fill=ACCENT2)
        c.create_oval(cx-5, cy-5, cx+5, cy+5, outline=ACCENT2)
        c.create_text(cx, cy+32, text="SEM SINAL", font=self.FB(14), fill=TEXT_SEC)

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
        return "#{:02x}{:02x}{:02x}".format(min(255,int(r+(255-r)*f)), min(255,int(g+(255-g)*f)), min(255,int(b+(255-b)*f)))

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select YOLO weights", filetypes=[("YOLO weights","*.pt *.onnx *.torchscript"), ("All files","*.*")])
        if p: self.model_path.set(p)

    def _browse_video(self):
        p = filedialog.askopenfilename(title="Select video", filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.webm"), ("All files","*.*")])
        if p: self.source_var.set(p)

    def _load_model(self):
        if not YOLO_AVAILABLE: messagebox.showerror("Pacote ausente", "pip install ultralytics"); return
        path = self.model_path.get().strip()
        self.status_var.set("CARREGANDO…"); self.update_idletasks()
        try:
            self.model = YOLO(path)
            self.status_var.set("MODELO OK")
        except Exception as e:
            messagebox.showerror("Erro no modelo", str(e))
            self.status_var.set("ERRO")

    def _start_inference(self):
        if not YOLO_AVAILABLE: messagebox.showerror("Pacote ausente", "pip install ultralytics"); return
        if self.model is None: messagebox.showwarning("Sem modelo", "Carregue um modelo primeiro."); return
        if self.running: return

        src_txt = self.source_var.get().strip()
        try:
            src = int(src_txt)
            self.source_is_camera = True
        except ValueError:
            src = src_txt
            self.source_is_camera = False

        if os.name == "nt" and self.source_is_camera:
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            messagebox.showerror("Erro na fonte", f"Não foi possível abrir: {src}")
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.source_is_camera:
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        while not self._frame_queue.empty():
            try: self._frame_queue.get_nowait()
            except queue.Empty: break

        self.running = True
        self.paused = False
        self._reset_inspection_state()
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.status_var.set("EXECUTANDO")

        self._cached_imgsz  = self.imgsz_var.get()
        self._cached_device = self.device_var.get()
        self.feed_canvas.update_idletasks()
        self._canvas_w = self.feed_canvas.winfo_width()
        self._canvas_h = self.feed_canvas.winfo_height()
        self.feed_canvas.bind("<Configure>", self._on_canvas_resize)

        threading.Thread(target=self._loop, daemon=True).start()
        self.after(15, self._process_frame_queue)

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶  RETOMAR")
            self.status_var.set("PAUSADO")
        else:
            self.btn_pause.config(text="⏸  PAUSAR")
            self.status_var.set("EXECUTANDO")

    def _stop_inference(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        while not self._frame_queue.empty():
            try: self._frame_queue.get_nowait()
            except queue.Empty: break
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸  PAUSAR")
        self.btn_stop.config(state="disabled")
        self.fps_display.set("FPS: --")
        self.det_count.set("DET: 0")
        self._reset_inspection_state()
        self.status_var.set("AGUARDANDO")
        self.after(120, self._draw_placeholder)

    def _loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.04)
                continue

            if self.cap is None:
                break

            ret, frame = self.cap.read()
            if not ret:
                if not self.source_is_camera and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.02)
                continue

            t0 = time.time()

            fh, fw = frame.shape[:2]
            self._frame_h = fh
            self._frame_w = fw

            conf   = self.conf_var.get()
            iou    = self.iou_var.get()
            imgsz  = self._cached_imgsz
            device = self._cached_device

            show_boxes  = self.show_boxes.get()
            show_labels = self.show_labels.get()
            show_conf   = self.show_conf.get()

            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False
            )
            result = results[0]
            n_det  = len(result.boxes)

            ann = self._annotate(frame, result, show_boxes, show_labels, show_conf)
            self._update_confidence_lock(result)

            trip_x = int(fw * self.trip_line_ratio)
            ann = self._draw_trip_line(ann, trip_x)

            ann = self._annotate_lock(ann, trip_x)
            self._handle_serial_trigger(result, n_det)

            # ── Verifica locks que cruzaram a linha ───────────────────────────
            self._check_trip_line(trip_x)
            
            # ── NOVO: Captura tela ao cruzar linha invisível no meio ──────────
            snap_x = int(fw * 0.50)
            self._check_snap_line(ann, snap_x)

            cw = self._canvas_w
            ch = self._canvas_h
            if cw > 1 and ch > 1:
                scale  = min(cw / fw, ch / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                rgb    = cv2.cvtColor(cv2.resize(ann, (nw, nh)), cv2.COLOR_BGR2RGB)
                photo  = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                x0, y0 = (cw - nw) // 2, (ch - nh) // 2
            else:
                photo  = None
                x0, y0 = 0, 0

            fps = 1.0 / max(time.time() - t0, 1e-6)
            self._enqueue_frame(photo, fps, n_det, x0, y0, cw, ch)

    def _is_defect_class(self, cls_name):
        name = cls_name.lower().strip()
        defect_keywords = (
            "tampa", "lacre",
            "nivel", "nível",
            "liquido", "líquido",
            "level", "cap", "seal",
            "rotulo", "rótulo",
            "etiqueta",
            "label",
            "logo",
            "sticker",
        )
        return any(keyword in name for keyword in defect_keywords)

    def _handle_serial_trigger(self, result, n_det):
        names_dict = result.names or {}

        non_ignored = [
            box for box in result.boxes
            if not self._is_ignored_class(str(names_dict.get(int(box.cls[0]), int(box.cls[0]))))
        ]
        object_on_screen = len(non_ignored) > 0

        defect_now = False
        defect_name_now = ""
        for box in non_ignored:
            cls_id = int(box.cls[0])
            cls_name = str(names_dict.get(cls_id, cls_id))
            if self._is_defect_class(cls_name):
                defect_now = True
                defect_name_now = cls_name
                break

        if self._insp_state == "IDLE":
            if defect_now:
                self._insp_state = "ENTERING"
                self._defect_name = defect_name_now
                self._entry_frames = 1
                self.status_var.set("VERIFICANDO...")
                print(f"[INSPEÇÃO] Possível defeito '{defect_name_now}' → ENTERING (1/{self.ENTRY_CONFIRM})")

        elif self._insp_state == "ENTERING":
            if defect_now:
                self._entry_frames += 1
                self.status_var.set(f"VERIFICANDO... ({self._entry_frames}/{self.ENTRY_CONFIRM})")
                if self._entry_frames >= self.ENTRY_CONFIRM:
                    self._insp_state = "DETECTING"
                    self._consecutive_empty = 0
                    self.status_var.set(f"DEFEITO: {self._defect_name.upper()}")
                    print(f"[INSPEÇÃO] Defeito confirmado: '{self._defect_name}' → DETECTING")
            else:
                print(f"[INSPEÇÃO] Falso positivo descartado após {self._entry_frames} frame(s) → IDLE")
                self._reset_inspection_state()
                self.status_var.set("EXECUTANDO")

        elif self._insp_state == "DETECTING":
            if defect_now:
                self._defect_name = defect_name_now
                self._consecutive_empty = 0
                self.status_var.set(f"DEFEITO: {defect_name_now.upper()}")
            elif object_on_screen:
                self._consecutive_empty = 0
            else:
                self._consecutive_empty += 1
                if self._consecutive_empty >= self.BLINK_TOLERANCE:
                    self._insp_state = "EXITING"
                    self._consecutive_empty = 0
                    print(f"[INSPEÇÃO] Garrafa saiu (BLINK_TOLERANCE={self.BLINK_TOLERANCE}) → EXITING")

        elif self._insp_state == "EXITING":
            if object_on_screen:
                self._insp_state = "DETECTING"
                self._consecutive_empty = 0
                print("[INSPEÇÃO] Objeto voltou após pisca → DETECTING (sem disparo)")
                self.status_var.set(f"DEFEITO: {self._defect_name.upper()}")
            else:
                self._consecutive_empty += 1
                if self._consecutive_empty >= self.CONFIRM_EXIT:
                    current_time = time.time()
                    if (self.serial_connected and self.serial_port and
                            current_time - self.last_trigger_time > self.trigger_cooldown):
                        try:
                            self.serial_port.write(b'd')
                            self.last_trigger_time = current_time
                            print(f"[SERIAL] Comando 'd' enviado! Defeito: '{self._defect_name}'")
                        except Exception as e:
                            print(f"[ERRO SERIAL] Falha ao enviar: {e}")
                    else:
                        print(f"[INSPEÇÃO] Cooldown ativo ou serial desconectado — disparo ignorado.")

                    self._reset_inspection_state()
                    self.status_var.set("EXECUTANDO")

    def _draw_trip_line(self, frame, trip_x):
        h, w = frame.shape[:2]
        out = frame

        LINE_COLOR  = (0, 60, 220)
        FLASH_COLOR = (0, 220, 255)

        has_pending = bool(self._pending_trips)
        color = FLASH_COLOR if has_pending else LINE_COLOR

        cv2.line(out, (trip_x, 0), (trip_x, h), color, 2)

        label = "DISPARO"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        tx = trip_x + 6
        ty = h // 2
        if tx + tw + 8 > w:
            tx = trip_x - tw - 12
        cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
        cv2.putText(out, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _check_trip_line(self, trip_x):
        now = time.time()
        active_ids = {l["id"] for l in self._locks}

        for lock in self._locks:
            lid = lock["id"]
            x2  = lock["box"][2]
            if lid in self._tripped_locks:
                continue
            if x2 >= trip_x and lid not in self._pending_trips:
                self._pending_trips[lid] = now
                print(f"[TRIP LINE] Lock #{lid} ('{lock['name']}') cruzou a linha x={trip_x:.0f}."
                      f" Aguardando {self.TRIP_DELAY_MS} ms…")

        fired = []
        for lid, ts in self._pending_trips.items():
            if now - ts >= self.TRIP_DELAY_MS / 1000.0:
                fired.append(lid)

        for lid in fired:
            del self._pending_trips[lid]
            self._tripped_locks.add(lid)

            current_time = time.time()
            if (self.serial_connected and self.serial_port and
                    current_time - self.last_trigger_time > self.trigger_cooldown):
                try:
                    self.serial_port.write(b'd')
                    self.last_trigger_time = current_time
                    print(f"[TRIP LINE] ✔ Comando 'd' disparado pelo lock #{lid}!")
                except Exception as e:
                    print(f"[TRIP LINE] Erro serial ao disparar: {e}")
            else:
                print(f"[TRIP LINE] Lock #{lid} — cooldown ativo ou serial desconectado.")

        stale = self._tripped_locks - active_ids
        if stale:
            self._tripped_locks -= stale
            print(f"[TRIP LINE] Lock(s) {stale} removido(s) do registro (saíram de cena).")

    def _check_snap_line(self, frame_to_save, snap_x):
        """
        Verifica se um lock cruzou a linha invisível no meio da tela.
        Se for um defeito, tira print e salva na pasta.
        """
        active_ids = {l["id"] for l in self._locks}

        for lock in self._locks:
            lid = lock["id"]
            x2  = lock["box"][2]

            if lid in self._snapped_locks:
                continue

            if x2 >= snap_x:
                if self._is_defect_class(lock["name"]):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # NOVO FORMATO: Data e hora primeiro para ordenar corretamente
                    filename = os.path.join(self.log_dir, f"{timestamp}_erro_{lock['name']}_id{lid}.jpg")
                    cv2.imwrite(filename, frame_to_save)
                    print(f"[LOG ERRO] Captura de '{lock['name']}' salva em: {filename}")
                self._snapped_locks.add(lid)

        stale = self._snapped_locks - active_ids
        if stale:
            self._snapped_locks -= stale

    @staticmethod
    def _box_iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _update_confidence_lock(self, result):
        boxes = result.boxes
        names = result.names or {}

        live = []
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            name = str(names.get(cls, cls))
            if self._is_ignored_class(name):
                continue
            live.append({"box": [x1, y1, x2, y2], "conf": conf,
                         "cls": cls, "name": name, "matched": False})

        for lock in self._locks:
            best_det = None
            best_iou = self.LOCK_IOU_MIN

            for det in live:
                iou = self._box_iou(lock["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None:
                best_det["matched"] = True

                old_cx = (lock["box"][0] + lock["box"][2]) / 2
                old_cy = (lock["box"][1] + lock["box"][3]) / 2
                new_box = best_det["box"]
                new_cx  = (new_box[0] + new_box[2]) / 2
                new_cy  = (new_box[1] + new_box[3]) / 2

                alpha = 0.4
                lock["vx"]    = alpha * (new_cx - old_cx) + (1 - alpha) * lock["vx"]
                lock["vy"]    = alpha * (new_cy - old_cy) + (1 - alpha) * lock["vy"]
                lock["box"]   = new_box[:]
                lock["empty"] = 0

                if best_det["conf"] > lock["conf"]:
                    print(f"[LOCK] #{lock['id']} conf {lock['conf']:.3f} → {best_det['conf']:.3f}")
                    lock["conf"] = best_det["conf"]
                    lock["name"] = best_det["name"]
                    lock["cls"]  = best_det["cls"]
            else:
                lock["empty"] += 1
                w  = lock["box"][2] - lock["box"][0]
                h  = lock["box"][3] - lock["box"][1]
                cx = (lock["box"][0] + lock["box"][2]) / 2 + lock["vx"]
                cy = (lock["box"][1] + lock["box"][3]) / 2 + lock["vy"]
                lock["box"] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                lock["vx"] *= 0.85
                lock["vy"] *= 0.85

        before = len(self._locks)
        self._locks = [l for l in self._locks if l["empty"] < self.LOCK_RELEASE_FRAMES]
        released = before - len(self._locks)
        if released:
            print(f"[LOCK] {released} lock(s) liberado(s) por timeout.")

        for det in live:
            if det["matched"]:
                continue
            if det["conf"] < self.LOCK_THRESHOLD:
                continue
            too_close = any(
                self._box_iou(det["box"], l["box"]) >= self.LOCK_IOU_MIN
                for l in self._locks
            )
            if too_close:
                continue
            new_id = max((l["id"] for l in self._locks), default=0) + 1
            self._locks.append({
                "id":    new_id,
                "box":   det["box"][:],
                "conf":  det["conf"],
                "cls":   det["cls"],
                "name":  det["name"],
                "vx":    0.0,
                "vy":    0.0,
                "empty": 0,
            })
            print(f"[LOCK] #{new_id} criado: '{det['name']}' conf={det['conf']:.3f}")

    def _annotate_lock(self, frame, trip_x=None):
        if not self._locks:
            return frame
        out = frame

        LOCK_COLOR    = (220, 80, 255)
        LOCK_COLOR2   = (180, 40, 200)
        ALERT_COLOR   = (0, 140, 255)
        ALERT_COLOR2  = (0, 100, 200)

        for lock in self._locks:
            x1, y1, x2, y2 = [int(v) for v in lock["box"]]
            conf = lock["conf"]
            name = lock["name"].upper()
            lid  = lock["id"]

            crossed = (trip_x is not None and x2 >= trip_x and lid not in self._tripped_locks)
            c1 = ALERT_COLOR  if crossed else LOCK_COLOR
            c2 = ALERT_COLOR2 if crossed else LOCK_COLOR2

            cv2.rectangle(out, (x1-2, y1-2), (x2+2, y2+2), c2, 1)
            cv2.rectangle(out, (x1, y1), (x2, y2), c1, 2)

            cl = 16
            for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out, (px, py), (px+dx*cl, py), c1, 3)
                cv2.line(out, (px, py), (px, py+dy*cl), c1, 3)

            if lid in self._pending_trips:
                elapsed = time.time() - self._pending_trips[lid]
                ratio   = min(elapsed / max(self.TRIP_DELAY_MS / 1000.0, 0.001), 1.0)
                bar_w   = x2 - x1
                filled  = int(bar_w * ratio)
                cv2.rectangle(out, (x1, y2 + 4), (x2, y2 + 10), (50, 50, 50), -1)
                cv2.rectangle(out, (x1, y2 + 4), (x1 + filled, y2 + 10), ALERT_COLOR, -1)

            suffix = "  ⏱" if lid in self._pending_trips else ("  ✔" if lid in self._tripped_locks else "")
            label = f"[LOCK #{lid}] {name}  {conf:.2f}{suffix}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
            cv2.rectangle(out, (x1, y1-th-10), (x1+tw+8, y1), c1, -1)
            cv2.putText(out, label, (x1+4, y1-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _enqueue_frame(self, photo, fps, n_det, x0, y0, cw, ch):
        try:
            if self._frame_queue.full():
                self._frame_queue.get_nowait()
            self._frame_queue.put_nowait((photo, fps, n_det, x0, y0, cw, ch))
        except (queue.Empty, queue.Full):
            pass

    def _process_frame_queue(self):
        if not self.running:
            return
        try:
            photo, fps, n_det, x0, y0, cw, ch = self._frame_queue.get_nowait()
            self._update_feed(photo, fps, n_det, x0, y0, cw, ch)
        except queue.Empty:
            pass
        self.after(15, self._process_frame_queue)

    def _is_ignored_class(self, cls_name):
        name = cls_name.lower().strip()
        ignored_keywords = ("normal",)
        return any(keyword in name for keyword in ignored_keywords)

    def _annotate(self, frame, result, show_boxes=True, show_labels=True, show_conf=True):
        if not show_boxes:
            return frame.copy()
        out = frame.copy()
        names = result.names or {}
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = str(names.get(cls, cls))
            if self._is_ignored_class(cls_name):
                continue
            color = self._cls_color(cls)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            cl = 10
            for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out, (px, py), (px+dx*cl, py), color, 3)
                cv2.line(out, (px, py), (px, py+dy*cl), color, 3)
            parts = []
            if show_labels:
                parts.append(cls_name.upper())
            if show_conf:
                parts.append(f"{conf:.2f}")
            text = "  ".join(parts)
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                cv2.putText(out, text, (x1+3, y1-4), cv2.FONT_HERSHEY_DUPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _cls_color(cls_id):
        pal = [(232,119,34),(192,192,192),(240,192,64),(192,57,43),
               (141,182,0),(180,140,60),(100,160,200),(200,100,50),(160,160,80)]
        return pal[cls_id % len(pal)]

    def _update_feed(self, photo, fps, n_det, x0, y0, cw, ch):
        if photo is None:
            return
        c = self.feed_canvas
        self._photo = photo
        c.delete("all")
        c.create_rectangle(0, 0, cw, ch, fill="#111111", outline="")
        c.create_image(x0, y0, anchor="nw", image=photo)
        self.fps_display.set(f"FPS: {fps:.1f}")
        self.det_count.set(f"DET: {n_det}")

    def _save_snapshot(self):
        if not self._photo:
            messagebox.showinfo("Captura", "Nenhum frame ativo.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path:
            try:
                self._photo._PhotoImage__photo.write(path, format="png")
                messagebox.showinfo("Salvo", path)
            except Exception as e:
                messagebox.showerror("Erro ao salvar", str(e))

    def _on_close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self._stop_inference()
        self.destroy()


def apply_style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TScale", background=BG_CARD, troughcolor=BG_ENTRY, slidercolor=ACCENT,
                sliderlength=16, sliderthickness=16, bordercolor=BG_CARD,
                darkcolor=BG_CARD, lightcolor=BG_CARD)
    s.map("TScale", slidercolor=[("active", WARNING)])
    s.configure("TScrollbar", background=BG_PANEL, troughcolor=BG_DARK, arrowcolor=TEXT_SEC,
                bordercolor=BG_PANEL, darkcolor=BG_PANEL, lightcolor=BG_PANEL)


if __name__ == "__main__":
    poppins_ok = _ensure_poppins()
    app = YOLOApp()
    ff = _register_poppins(app) if poppins_ok else ("Segoe UI" if os.name == "nt" else "Helvetica Neue")
    app.FF = ff
    apply_style()
    app.mainloop()