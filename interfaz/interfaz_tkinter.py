
# =========================================================
# IMPORTACIONES Y CONFIGURACIÓN
# =========================================================

import tkinter as tk
from tkinter import messagebox

from PIL import (
    Image,
    ImageTk,
    ImageDraw
)

import cv2

from config_mysql import ConfigMySQL
from repositorio_mysql import RepositorioMySQL
from motor_reconocimiento import MotorReconocimientoFacial


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

cfg_mysql = ConfigMySQL()

repo = RepositorioMySQL(
    cfg_mysql
)

motor = MotorReconocimientoFacial(

    cfg_mysql=cfg_mysql,

    ruta_modelo="face_cnn.keras",

    ruta_encoder="label_encoder.npy",

    tam_imagen=(96, 96),

    carpeta_capturas="data/capturas",
)

# =========================================================
# CÁMARA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "No se pudo abrir la cámara."
    )

# =========================================================
# PALETA DE COLORES
# =========================================================

BG = "#0f172a"
PANEL = "#111c33"
CARD = "#0b1224"

BORDER = "#24324a"

FG = "#e5e7eb"
MUTED = "#a7b0c0"

ACCENT = "#3b82f6"

OK = "#22c55e"
WARN = "#f59e0b"
BAD = "#ef4444"

# =========================================================
# FUENTES
# =========================================================

FONT_H1 = (
    "Segoe UI",
    18,
    "bold"
)

FONT_H2 = (
    "Segoe UI",
    14,
    "bold"
)

FONT_TXT = (
    "Segoe UI",
    11
)

FONT_TXT_B = (
    "Segoe UI",
    11,
    "bold"
)

# =========================================================
# UTILIDADES DE COLOR
# =========================================================

def _hex_to_rgb(h):

    h = h.lstrip("#")

    return tuple(
        int(h[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def _rgb_to_hex(rgb):

    return (
        "#%02x%02x%02x" % rgb
    )


def _lerp(a, b, t):

    return int(
        a + (b - a) * t
    )


def _mix(c1, c2, t):

    r1, g1, b1 = _hex_to_rgb(c1)

    r2, g2, b2 = _hex_to_rgb(c2)

    return _rgb_to_hex(

        (
            _lerp(r1, r2, t),

            _lerp(g1, g2, t),

            _lerp(b1, b2, t)
        )
    )


def _status_color(estatus: str) -> str:

    e = (
        estatus or ""
    ).lower()

    if "reconoc" in e:
        return OK

    if "sin modelo" in e:
        return WARN

    if "no detect" in e:
        return MUTED

    if "no registr" in e:
        return WARN

    return MUTED

# =========================================================
# CARD PERSONALIZADA
# =========================================================

class RoundedCard(tk.Frame):

    def __init__(
        self,
        master,
        title: str = "",
        **kw
    ):

        super().__init__(

            master,

            bg=PANEL,

            highlightbackground=BORDER,

            highlightthickness=1,

            bd=0,

            **kw
        )

        self.inner = tk.Frame(
            self,
            bg=CARD
        )

        self.inner.grid(

            row=0,

            column=0,

            sticky="nsew",

            padx=10,

            pady=10
        )

        self.grid_rowconfigure(
            0,
            weight=1
        )

        self.grid_columnconfigure(
            0,
            weight=1
        )

        self.inner.grid_columnconfigure(
            0,
            weight=1
        )

        self.inner.grid_rowconfigure(
            1,
            weight=1
        )

        self.content = tk.Frame(
            self.inner,
            bg=CARD
        )

        self.content.grid(

            row=1,

            column=0,

            sticky="nsew"
        )

        if title:

            tk.Label(

                self.inner,

                text=title,

                bg=CARD,

                fg=FG,

                font=FONT_H2

            ).grid(

                row=0,

                column=0,

                sticky="w",

                pady=(0, 10)
            )

# =========================================================
# VENTANA PRINCIPAL
# =========================================================

root = tk.Tk()

root.title(
    "Sistema de Reconocimiento Facial"
)

root.configure(
    bg=BG
)

root.grid_rowconfigure(
    0,
    weight=1
)

root.grid_columnconfigure(
    0,
    weight=3
)

root.grid_columnconfigure(
    1,
    weight=1
)

# =========================================================
# PANEL IZQUIERDO
# =========================================================

left = RoundedCard(
    root,
    title="Cámara Web en Vivo"
)

left.grid(

    row=0,

    column=0,

    sticky="nsew",

    padx=12,

    pady=12
)

left.content.grid_rowconfigure(
    0,
    weight=1
)

left.content.grid_columnconfigure(
    0,
    weight=1
)

label_cam = tk.Label(
    left.content,
    bg="#000000"
)

label_cam.grid(

    row=0,

    column=0,

    sticky="nsew"
)

# =========================================================
# PANEL DERECHO
# =========================================================

right = tk.Frame(
    root,
    bg=BG
)

right.grid(

    row=0,

    column=1,

    sticky="nsew",

    padx=(0, 12),

    pady=12
)

right.grid_columnconfigure(
    0,
    weight=1
)

right.grid_rowconfigure(
    2,
    weight=1
)

tk.Label(

    right,

    text="Panel de Resultados",

    bg=BG,

    fg=FG,

    font=FONT_H1

).grid(

    row=0,

    column=0,

    sticky="w",

    pady=(0, 10)
  )
# =========================================================
# CARD RESULTADOS
# =========================================================

card_res = RoundedCard(
    right
)

card_res.grid(

    row=1,

    column=0,

    sticky="ew",

    pady=(0, 12)
)

card_res.content.grid_columnconfigure(
    0,
    weight=1
)

top_row = tk.Frame(
    card_res.content,
    bg=CARD
)

top_row.grid(

    row=0,

    column=0,

    sticky="ew"
)

top_row.grid_columnconfigure(
    0,
    weight=1
)

lbl_nombre = tk.Label(

    top_row,

    text="---",

    bg=CARD,

    fg=FG,

    font=(
        "Segoe UI",
        16,
        "bold"
    )
)

lbl_nombre.grid(

    row=0,

    column=0,

    sticky="w"
)

lbl_badge = tk.Label(

    top_row,

    text="---",

    bg=_status_color("---"),

    fg="#0b1224",

    font=FONT_TXT_B,

    padx=10,

    pady=4
)

lbl_badge.grid(

    row=0,

    column=1,

    sticky="e"
)

grid_info = tk.Frame(
    card_res.content,
    bg=CARD
)

grid_info.grid(

    row=1,

    column=0,

    sticky="ew",

    pady=(10, 0)
)

grid_info.grid_columnconfigure(
    1,
    weight=1
)

var_matricula = tk.StringVar(
    value="---"
)

var_inscrito = tk.StringVar(
    value="---"
)

var_carrera = tk.StringVar(
    value="---"
)

var_conf = tk.StringVar(
    value="0.00"
)
