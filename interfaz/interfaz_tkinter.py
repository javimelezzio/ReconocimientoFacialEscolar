import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2

from config_mysql import ConfigMySQL
from repositorio_mysql import RepositorioMySQL
from motor_reconocimiento import MotorReconocimientoFacial

cfg_mysql = ConfigMySQL()
repo = RepositorioMySQL(cfg_mysql)

motor = MotorReconocimientoFacial(
    cfg_mysql=cfg_mysql,
    ruta_modelo="face_cnn.keras",
    ruta_encoder="label_encoder.npy",
    tam_imagen=(96, 96),
    carpeta_capturas="data/capturas",
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

BG = "#0b1220"
SIDEBAR = "#07101c"
PANEL = "#111c33"
CARD = "#0f1a2b"
BORDER = "#24324a"
FG = "#e8eefc"
MUTED = "#9aa7bd"
ACCENT = "#20d090"
OK = "#22c55e"
WARN = "#f59e0b"
BAD = "#ef4444"

FONT_H2 = ("Segoe UI", 13, "bold")
FONT_TXT = ("Segoe UI", 11)
FONT_TXT_B = ("Segoe UI", 11, "bold")


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def _lerp(a, b, t):
    return int(a + (b - a) * t)


def _mix(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))


def _status_color(estatus: str) -> str:
    e = (estatus or "").lower()
    if "reconoc" in e:
        return OK
    if "sin modelo" in e:
        return WARN
    if "no detect" in e:
        return MUTED
    if "no registr" in e:
        return WARN
    return MUTED


def _get_lanczos_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _resize_keep_aspect(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    iw, ih = pil_img.size
    if iw <= 0 or ih <= 0:
        return pil_img
    scale = min(max_w / iw, max_h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    return pil_img.resize((nw, nh), _get_lanczos_resample())


class RoundedCard(tk.Frame):
    def __init__(self, master, title: str = "", **kw):
        super().__init__(master, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, bd=0, **kw)
        self.inner = tk.Frame(self, bg=CARD)
        self.inner.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.inner.grid_columnconfigure(0, weight=1)
        self.inner.grid_rowconfigure(1, weight=1)
        if title:
            tk.Label(self.inner, text=title, bg=CARD, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.content = tk.Frame(self.inner, bg=CARD)
        self.content.grid(row=1, column=0, sticky="nsew")


def _entry(parent):
    e = tk.Entry(
        parent,
        font=FONT_TXT,
        fg=FG,
        bg=_mix(CARD, "#ffffff", 0.05),
        insertbackground=FG,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        bd=0,
    )
    return e


def _button(parent, text, cmd, primary=False):
    bg = "#1c6fd6" if primary else _mix(CARD, "#ffffff", 0.06)
    fg = "#ffffff" if primary else FG
    return tk.Button(
        parent,
        text=text,
        command=cmd,
        font=("Segoe UI", 11, "bold"),
        bg=bg,
        fg=fg,
        activebackground=_mix(bg, "#ffffff", 0.10),
        activeforeground=fg,
        relief="flat",
        bd=0,
        padx=12,
        pady=10,
        highlightthickness=1,
        highlightbackground=BORDER,
    )


root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")
root.configure(bg=BG)
root.geometry("1250x720")
root.minsize(1100, 650)
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)


def _popup_text(title, header, body, w=820, h=560):
    win = tk.Toplevel(root)
    win.title(title)
    win.configure(bg=BG)
    win.geometry(f"{w}x{h}")
    win.transient(root)
    win.grab_set()

    cont = RoundedCard(win, title=header)
    cont.pack(fill="both", expand=True, padx=16, pady=16)

    text = tk.Text(
        cont.content,
        bg=CARD,
        fg=FG,
        insertbackground=FG,
        relief="flat",
        bd=0,
        wrap="word",
        font=("Segoe UI", 11),
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    text.pack(fill="both", expand=True)
    text.insert("1.0", body)
    text.configure(state="disabled")
    _button(cont.content, "Cerrar", win.destroy, primary=True).pack(fill="x", pady=(12, 0))


def mostrar_roles_finales():
    body = (
        "@luzbel999 – AGUILAR MÉRIDA ERIC ALEXANDER\n"
        "Encargado del diseño, trabajando en conjunto con el responsable principal para definir la estructura funcional del proyecto.\n\n"
        "CASTAÑEDA FARFÁN ITALY VALERIA\n"
        "Apoyo en el área de diseño, contribuyendo a la definición estética y visual del proyecto, así como en la propuesta de mejoras para la interfaz.\n\n"
        "@1237-the – DE JESÚS MENDOZA FRANCISCO JAVIER\n"
        "Responsable de la implementación y adaptación de la base de datos MySQL, asegurando la correcta migración y el manejo eficiente de la información.\n\n"
        "@javimelezzio – HERNÁNDEZ MELECIO ROBERTO JAVIER\n"
        "Responsable de la gestión integral del proyecto en GitHub, asegurando la correcta organización de los archivos y versiones. "
        "Encargado de realizar pruebas de funcionamiento y validar la integración entre la base de datos, la interfaz y los módulos de procesamiento.\n\n"
        "@jessicaasc60-wq – JESSICA MARÍN ASCENCIO\n"
        "Líder en diseño. Se encargó de estructurar la interfaz gráfica y coordinar la integración visual con la lógica del sistema, garantizando una experiencia de usuario clara y funcional.\n\n"
        "@angeljuarezcruz87-cloud – JUÁREZ CRUZ ÁNGEL DAVID\n"
        "Colaborador en diseño y pruebas. Participó en la validación del sistema y en la retroalimentación sobre la experiencia de usuario, además de apoyar en tareas de diseño complementarias.\n\n"
        "@pinmode17 – SÁNCHEZ FLORES JOSÉ MANUEL\n"
        "Responsable del diseño, creación e implementación de la base de datos. Colaboró en la definición de esquemas y relaciones, asegurando la consistencia y escalabilidad del sistema de almacenamiento.\n"
    )
    _popup_text("Roles finales", "ROLES FINALES", body, 820, 560)


def mostrar_agradecimientos():
    body = (
        "Gracias a todas las personas que apoyaron en pruebas, retroalimentación y revisión del proyecto.\n\n"
        "Agradecemos especialmente el apoyo brindado durante el desarrollo e integración del sistema.\n"
    )
    _popup_text("Agradecimientos", "AGRADECIMIENTOS", body, 700, 420)


sidebar = tk.Frame(root, bg=SIDEBAR, width=200)
sidebar.grid(row=0, column=0, sticky="nsw")
sidebar.grid_propagate(False)
sidebar.grid_rowconfigure(10, weight=1)

def _nav_btn_icon(text, r, cmd=None):
    b = tk.Button(
        sidebar,
        text=text,
        font=("Segoe UI", 16),
        bg=_mix(SIDEBAR, "#ffffff", 0.08),
        fg=FG,
        activebackground=_mix(SIDEBAR, "#ffffff", 0.12),
        activeforeground=FG,
        relief="flat",
        bd=0,
        padx=10,
        pady=12,
        command=cmd,
    )
    b.grid(row=r, column=0, padx=12, pady=10, sticky="ew")
    return b

_nav_btn_icon("🏠", 0)
_nav_btn_icon("👤", 1, cmd=mostrar_roles_finales)

lbl_agr = tk.Label(sidebar, text="Agradecimientos", bg=SIDEBAR, fg=FG, font=("Segoe UI", 13, "bold"))
lbl_agr.grid(row=2, column=0, padx=12, pady=(18, 8), sticky="ew")

btn_agr = tk.Button(
    sidebar,
    text="Ver",
    font=("Segoe UI", 11, "bold"),
    bg=_mix(SIDEBAR, "#ffffff", 0.08),
    fg=FG,
    activebackground=_mix(SIDEBAR, "#ffffff", 0.12),
    activeforeground=FG,
    relief="flat",
    bd=0,
    padx=10,
    pady=10,
    command=mostrar_agradecimientos,
)
btn_agr.grid(row=3, column=0, padx=12, pady=(0, 10), sticky="ew")

btn_salir = tk.Button(
    sidebar,
    text="Salir",
    font=("Segoe UI", 10, "bold"),
    bg=_mix(SIDEBAR, "#ffffff", 0.08),
    fg=FG,
    activebackground=_mix(SIDEBAR, "#ffffff", 0.12),
    activeforeground=FG,
    relief="flat",
    bd=0,
    padx=10,
    pady=8,
    command=root.destroy,
)
btn_salir.grid(row=11, column=0, padx=12, pady=(0, 12), sticky="ew")


main = tk.Frame(root, bg=BG)
main.grid(row=0, column=1, sticky="nsew")
main.grid_rowconfigure(0, weight=1)
main.grid_columnconfigure(0, weight=3)
main.grid_columnconfigure(1, weight=2)


card_cam = RoundedCard(main, title="")
card_cam.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(16, 16))
card_cam.content.grid_rowconfigure(1, weight=1)
card_cam.content.grid_columnconfigure(0, weight=1)

cam_header = tk.Frame(card_cam.content, bg=CARD)
cam_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
tk.Label(cam_header, text="CÁMARA EN VIVO", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")

label_cam = tk.Label(card_cam.content, bg="#000000")
label_cam.grid(row=1, column=0, sticky="nsew")


right = tk.Frame(main, bg=BG)
right.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=(16, 16))
right.grid_columnconfigure(0, weight=1)
right.grid_rowconfigure(0, weight=1)
right.grid_rowconfigure(1, weight=1)


card_res = RoundedCard(right, title="RESULTADO DE RECONOCIMIENTO")
card_res.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
card_res.content.grid_columnconfigure(0, weight=1)

top_row = tk.Frame(card_res.content, bg=CARD)
top_row.grid(row=0, column=0, sticky="ew")
top_row.grid_columnconfigure(0, weight=1)

lbl_nombre = tk.Label(top_row, text="---", bg=CARD, fg=FG, font=("Segoe UI", 18, "bold"))
lbl_nombre.grid(row=0, column=0, sticky="w")

lbl_badge = tk.Label(top_row, text="---", bg=_status_color("---"), fg="#0b1220", font=FONT_TXT_B, padx=10, pady=4)
lbl_badge.grid(row=0, column=1, sticky="e")

grid_info = tk.Frame(card_res.content, bg=CARD)
grid_info.grid(row=1, column=0, sticky="ew", pady=(10, 0))
grid_info.grid_columnconfigure(1, weight=1)

var_matricula = tk.StringVar(value="---")
var_inscrito = tk.StringVar(value="---")
var_carrera = tk.StringVar(value="---")
var_conf = tk.StringVar(value="0.00")

def _info_row(r, label, value_var):
    tk.Label(grid_info, text=label, bg=CARD, fg=MUTED, font=FONT_TXT).grid(row=r, column=0, sticky="w", pady=3)
    tk.Label(grid_info, textvariable=value_var, bg=CARD, fg=FG, font=FONT_TXT_B).grid(row=r, column=1, sticky="w", pady=3)

_info_row(0, "Matrícula:", var_matricula)
_info_row(1, "Inscrito:", var_inscrito)
_info_row(2, "Carrera:", var_carrera)
_info_row(3, "Precisión:", var_conf)

bar = tk.Canvas(card_res.content, height=10, bg=CARD, highlightthickness=0)
bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))

def _draw_conf_bar(conf: float):
    bar.delete("all")
    w = bar.winfo_width() or 280
    h = 10
    bar.create_rectangle(0, 0, w, h, fill=_mix(CARD, "#ffffff", 0.06), outline="")
    conf = max(0.0, min(1.0, float(conf)))
    if conf < 0.5:
        c = _mix(BAD, WARN, conf / 0.5)
    else:
        c = _mix(WARN, OK, (conf - 0.5) / 0.5)
    bar.create_rectangle(0, 0, int(w * conf), h, fill=c, outline="")

card_reg = RoundedCard(right, title="REGISTRO DEL ESTUDIANTE")
card_reg.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
card_reg.content.grid_columnconfigure(1, weight=1)

tk.Label(card_reg.content, text="Matrícula:", bg=CARD, fg=FG, font=FONT_TXT).grid(row=0, column=0, sticky="w", pady=6, padx=(0, 8))
matricula_entry = _entry(card_reg.content)
matricula_entry.grid(row=0, column=1, sticky="ew", pady=6)

tk.Label(card_reg.content, text="Nombre:", bg=CARD, fg=FG, font=FONT_TXT).grid(row=1, column=0, sticky="w", pady=6, padx=(0, 8))
nombre_entry = _entry(card_reg.content)
nombre_entry.grid(row=1, column=1, sticky="ew", pady=6)

tk.Label(card_reg.content, text="Inscrito (SI/NO):", bg=CARD, fg=FG, font=FONT_TXT).grid(row=2, column=0, sticky="w", pady=6, padx=(0, 8))
inscrito_entry = _entry(card_reg.content)
inscrito_entry.insert(0, "SI")
inscrito_entry.grid(row=2, column=1, sticky="ew", pady=6)

tk.Label(card_reg.content, text="Carrera:", bg=CARD, fg=FG, font=FONT_TXT).grid(row=3, column=0, sticky="w", pady=6, padx=(0, 8))
carrera_entry = _entry(card_reg.content)
carrera_entry.grid(row=3, column=1, sticky="ew", pady=6)

foquito_canvas = tk.Canvas(card_reg.content, width=16, height=16, bg=CARD, highlightthickness=0)
foquito_canvas.grid(row=4, column=1, sticky="e", pady=(10, 0))
foquito = foquito_canvas.create_oval(2, 2, 14, 14, fill=_mix(CARD, "#ffffff", 0.08), outline=BORDER)
_foquito_after_id = None

def _foquito_set(color_hex: str):
    foquito_canvas.itemconfig(foquito, fill=color_hex)

def _foquito_off():
    _foquito_set(_mix(CARD, "#ffffff", 0.08))

def _foquito_ok_pulse(ms=1500):
    global _foquito_after_id
    if _foquito_after_id is not None:
        try:
            root.after_cancel(_foquito_after_id)
        except Exception:
            pass
        _foquito_after_id = None
    _foquito_set(OK)
    _foquito_after_id = root.after(ms, _foquito_off)

def registrar_o_actualizar_alumno():
    try:
        repo.guardar_o_actualizar_alumno(
            matricula=matricula_entry.get().strip(),
            nombre=nombre_entry.get().strip(),
            inscrito=inscrito_entry.get().strip(),
            carrera=carrera_entry.get().strip(),
        )
        messagebox.showinfo("OK", "Alumno registrado/actualizado.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def guardar_imagen_para_matricula():
    matricula = matricula_entry.get().strip()
    if not matricula:
        messagebox.showwarning("Falta matrícula", "Escribe la matrícula para asociar la imagen.")
        return
    try:
        motor.guardar_cara_detectada(matricula, notas="captura desde interfaz")
        _foquito_ok_pulse(1500)
    except Exception as e:
        messagebox.showerror("Error", str(e))

_button(card_reg.content, "Guardar / Actualizar alumno", registrar_o_actualizar_alumno, primary=True).grid(row=4, column=0, sticky="ew", pady=(14, 8))
_button(card_reg.content, "Guardar imagen (cara detectada)", guardar_imagen_para_matricula, primary=False).grid(row=5, column=0, columnspan=2, sticky="ew")

_target_w = 640
_target_h = 480

def _update_target_size(event=None):
    global _target_w, _target_h
    w = label_cam.winfo_width()
    h = label_cam.winfo_height()
    if w > 50 and h > 50:
        _target_w, _target_h = w, h

label_cam.bind("<Configure>", _update_target_size)

def actualizar_panel_resultados(r: dict):
    matricula = r.get("matricula_predicha") or "---"
    nombre = r.get("nombre") or "---"
    inscrito = r.get("inscrito") or "---"
    carrera = r.get("carrera") or "---"
    estatus = r.get("estatus", "---")
    conf = float(r.get("confianza", 0.0))

    lbl_nombre.configure(text=nombre)
    lbl_badge.configure(text=estatus, bg=_status_color(estatus))

    var_matricula.set(str(matricula))
    var_inscrito.set(str(inscrito))
    var_carrera.set(str(carrera))
    var_conf.set("%.2f" % conf)

    _draw_conf_bar(conf)

def actualizar_frame():
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        root.after(30, actualizar_frame)
        return

    r = motor.predecir(frame_bgr)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    if r.get("caja") is not None:
        x, y, w, h = r["caja"]
        draw.rectangle([x, y, x + w, y + h], outline=_mix(OK, ACCENT, 0.35), width=4)

    actualizar_panel_resultados(r)

    img2 = _resize_keep_aspect(img, _target_w, _target_h)
    imgtk = ImageTk.PhotoImage(img2)
    label_cam.imgtk = imgtk
    label_cam.configure(image=imgtk)

    root.after(30, actualizar_frame)

def al_cerrar():
    try:
        cap.release()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", al_cerrar)
actualizar_frame()
root.mainloop()
