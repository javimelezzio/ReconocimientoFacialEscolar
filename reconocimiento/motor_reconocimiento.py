import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Deque
from collections import deque

import cv2
import numpy as np

from config_mysql import ConfigMySQL
from repositorio_mysql import RepositorioMySQL


class MotorReconocimientoFacial:

    # =========================
    # CONFIGURACIÓN GENERAL
    # =========================

    UMBRAL_CONFIANZA = 0.80
    SUAVIZADO_FRAMES = 8

    def __init__(

        self,

        cfg_mysql: ConfigMySQL,

        ruta_modelo: str = "face_cnn.keras",

        ruta_encoder: str = "label_encoder.npy",

        tam_imagen: Tuple[int, int] = (96, 96),

        carpeta_capturas: str = os.path.join(
            "data",
            "capturas"
        ),
    ):

        self.repo = RepositorioMySQL(cfg_mysql)

        self.tam_imagen = tam_imagen

        base_dir = Path(__file__).resolve().parent

        self.ruta_modelo = str(
            (base_dir / ruta_modelo).resolve()
        )

        self.ruta_encoder = str(
            (base_dir / ruta_encoder).resolve()
        )

        self.carpeta_capturas = str(
            (base_dir / carpeta_capturas).resolve()
        )

        os.makedirs(
            self.carpeta_capturas,
            exist_ok=True
        )

        # =========================
        # DETECTOR DE ROSTROS
        # =========================

        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # =========================
        # VARIABLES INTERNAS
        # =========================

        self._ultimo_frame_bgr = None

        self._ultima_caja = None
        # (x, y, w, h)

        self.modelo = None
        self.encoder = None

        self._intentar_cargar_modelo()

        # Historial de probabilidades
        # para suavizado temporal
        self._probs_hist: Deque[np.ndarray] = deque(
            maxlen=self.SUAVIZADO_FRAMES
        )

    # =========================
    # CARGA DEL MODELO
    # =========================

    def _intentar_cargar_modelo(self) -> None:

        try:

            import tensorflow as tf
            from sklearn.preprocessing import LabelEncoder

            if not os.path.exists(self.ruta_modelo):

                raise FileNotFoundError(
                    f"No existe el modelo: "
                    f"{self.ruta_modelo}"
                )

            if not os.path.exists(self.ruta_encoder):

                raise FileNotFoundError(
                    f"No existe el encoder: "
                    f"{self.ruta_encoder}"
                )

            # Cargar CNN
            self.modelo = tf.keras.models.load_model(
                self.ruta_modelo
            )

            # Cargar clases
            clases = np.load(
                self.ruta_encoder,
                allow_pickle=True
            )

            enc = LabelEncoder()
            enc.fit(clases)

            self.encoder = enc

            print(
                "Modelo cargado OK:",
                self.ruta_modelo
            )

        except Exception as e:

            self.modelo = None
            self.encoder = None

            print(
                "AVISO: "
                "No se cargó el modelo/encoder."
            )

            print(
                "Modo SOLO CAPTURA."
            )

            print("Detalle:", e)

    # =========================
    # DETECCIÓN DE CARAS
    # =========================

    def detectar_caras(
        self,
        frame_bgr: np.ndarray
    ):

        gris = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY
        )

        caras = self.detector.detectMultiScale(

            gris,

            scaleFactor=1.2,

            minNeighbors=5,

            minSize=(60, 60)
        )

        return caras

    # =========================
    # PREPROCESAMIENTO
    # =========================

    def _preprocesar_cara(
        self,
        cara_bgr: np.ndarray
    ) -> np.ndarray:

        gris = cv2.cvtColor(
            cara_bgr,
            cv2.COLOR_BGR2GRAY
        )

        gris = cv2.resize(

            gris,

            self.tam_imagen,

            interpolation=cv2.INTER_AREA
        )

        gris = gris.astype("float32") / 255.0

        # (H, W, 1)
        x = np.expand_dims(
            gris,
            axis=-1
        )

        # (1, H, W, 1)
        x = np.expand_dims(
            x,
            axis=0
        )

        return x

    # =========================
    # SUAVIZADO TEMPORAL
    # =========================

    def _promedio_probs(
        self,
        probs_actual: np.ndarray
    ) -> np.ndarray:

        self._probs_hist.append(
            probs_actual.astype(np.float32)
        )

        arr = np.stack(
            list(self._probs_hist),
            axis=0
        )

        return arr.mean(axis=0)

    # =========================
    # PREDICCIÓN
    # =========================

    def predecir(
        self,
        frame_bgr: np.ndarray
    ) -> Dict[str, Any]:

        self._ultimo_frame_bgr = frame_bgr

        caras = self.detectar_caras(
            frame_bgr
        )

        # =========================
        # NO HAY ROSTROS
        # =========================

        if len(caras) == 0:

            self._ultima_caja = None

            self._probs_hist.clear()

            return {

                "caja": None,

                "matricula_predicha": None,

                "confianza": 0.0,

                "nombre": None,

                "inscrito": None,

                "carrera": None,

                "estatus": "No detectado",
            }

        # Tomar la primera cara detectada
        x, y, w, h = [
            int(v)
            for v in caras[0]
        ]

        self._ultima_caja = (
            x,
            y,
            w,
            h
        )

        # =========================
        # MODO SOLO CAPTURA
        # =========================

        if self.modelo is None or self.encoder is None:

            self._probs_hist.clear()

            return {

                "caja": self._ultima_caja,

                "matricula_predicha": None,

                "confianza": 0.0,

                "nombre": None,

                "inscrito": None,

                "carrera": None,

                "estatus": (
                    "Sin modelo "
                    "(solo captura)"
                ),
            }

        # =========================
        # RECORTE DE ROSTRO
        # =========================

        cara_roi = frame_bgr[
            y:y + h,
            x:x + w
        ]

        xin = self._preprocesar_cara(
            cara_roi
        )

        # =========================
        # PREDICCIÓN CNN
        # =========================

        probs = self.modelo.predict(
            xin,
            verbose=0
        )[0]

        probs_suav = self._promedio_probs(
            probs
        )

        idx = int(np.argmax(probs_suav))

        confianza = float(
            probs_suav[idx]
        )

        # =========================
        # NO CONFIABLE
        # =========================

        if confianza < self.UMBRAL_CONFIANZA:

            return {

                "caja": self._ultima_caja,

                "matricula_predicha": None,

                "confianza": confianza,

                "nombre": "Desconocido",

                "inscrito": "---",

                "carrera": "---",

                "estatus": (
                    f"No seguro "
                    f"(<{self.UMBRAL_CONFIANZA:.2f})"
                ),
            }

        # =========================
        # MATRÍCULA PREDICHA
        # =========================

        matricula = self.encoder.inverse_transform(
            [idx]
        )[0]

        alumno = self.repo.obtener_alumno(
            matricula
        )

        # =========================
        # CONSULTAR BD
        # =========================

        if alumno is None:

            nombre = "Desconocido"
            inscrito = "---"
            carrera = "---"

            estatus = "No registrado"

        else:

            nombre = getattr(
                alumno,
                "nombre",
                "Desconocido"
            )

            inscrito = getattr(
                alumno,
                "inscrito",
                "---"
            )

            carrera = getattr(
                alumno,
                "carrera",
                "---"
            )

            estatus = "Reconocido"

        return {

            "caja": self._ultima_caja,

            "matricula_predicha": matricula,

            "confianza": confianza,

            "nombre": nombre,

            "inscrito": inscrito,

            "carrera": carrera,

            "estatus": estatus,
        }

    # =========================
    # GUARDAR CAPTURA
    # =========================

    def guardar_cara_detectada(

        self,

        matricula: str,

        notas: Optional[str] = None

    ) -> str:

        matricula = (
            matricula or ""
        ).strip()

        if not matricula:

            raise ValueError(
                "La matrícula "
                "no puede estar vacía."
            )

        if (

            self._ultimo_frame_bgr is None

            or

            self._ultima_caja is None
        ):

            raise RuntimeError(

                "No hay cara detectada aún.\n"

                "Primero apunta la cámara "
                "a un rostro."
            )

        x, y, w, h = self._ultima_caja

        cara_roi = self._ultimo_frame_bgr[
            y:y + h,
            x:x + w
        ]

        # Timestamp
        ts = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        nombre_archivo = (
            f"{matricula}_{ts}.jpg"
        )

        ruta = os.path.join(
            self.carpeta_capturas,
            nombre_archivo
        )

        # Guardar imagen
        ok = cv2.imwrite(
            ruta,
            cara_roi
        )

        if not ok:

            raise RuntimeError(
                "No se pudo guardar "
                "la imagen en disco."
            )

        self.repo.registrar_imagen_de_alumno(

            matricula=matricula,

            ruta_imagen=ruta,

            ancho=w,

            alto=h,

            notas=notas,
        )

        return ruta
