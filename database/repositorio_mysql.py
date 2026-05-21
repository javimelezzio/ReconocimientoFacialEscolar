import mysql.connector
from mysql.connector import Error
from dataclasses import dataclass
from typing import Optional, List
from config_mysql import ConfigMySQL


@dataclass
class Alumno:
    matricula: str
    nombre: str
    inscrito: str
    carrera: str


@dataclass
class ImagenAlumno:
    id: int
    matricula: str
    ruta_imagen: str
    capturada_en: str
    ancho: int
    alto: int
    notas: Optional[str]


class RepositorioMySQL:

    def __init__(self, cfg: ConfigMySQL):
        self.cfg = cfg

    def _conectar(self):
        try:
            return mysql.connector.connect(
                host=self.cfg.host,
                port=self.cfg.port,
                user=self.cfg.user,
                password=self.cfg.password,
                database=self.cfg.database,
                autocommit=True,
            )
        except Error as e:
            raise ConnectionError(f"Error conectando a MySQL: {e}")

    # =========================
    # ALUMNOS
    # =========================

    def guardar_o_actualizar_alumno(
        self,
        matricula: str,
        nombre: str,
        inscrito: str,
        carrera: str
    ) -> None:

        inscrito = inscrito.strip().upper()

        if inscrito not in ("SI", "NO"):
            raise ValueError(
                "El campo 'inscrito' debe ser 'SI' o 'NO'."
            )

        sql = """
        INSERT INTO alumnos (
            matricula,
            nombre,
            inscrito,
            carrera
        )
        VALUES (%s, %s, %s, %s)

        ON DUPLICATE KEY UPDATE
            nombre = VALUES(nombre),
            inscrito = VALUES(inscrito),
            carrera = VALUES(carrera)
        """

        with self._conectar() as cnx:
            with cnx.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        matricula,
                        nombre,
                        inscrito,
                        carrera
                    )
                )

    def obtener_alumno(
        self,
        matricula: str
    ) -> Optional[Alumno]:

        sql = """
        SELECT
            matricula,
            nombre,
            inscrito,
            carrera
        FROM alumnos
        WHERE matricula = %s
        """

        with self._conectar() as cnx:
            with cnx.cursor() as cur:
                cur.execute(sql, (matricula,))
                row = cur.fetchone()

        return Alumno(*row) if row else None

    def listar_alumnos(self) -> List[Alumno]:

        sql = """
        SELECT
            matricula,
            nombre,
            inscrito,
            carrera
        FROM alumnos
        ORDER BY matricula
        """

        with self._conectar() as cnx:
            with cnx.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        return [Alumno(*r) for r in rows]

    # =========================
    # IMÁGENES
    # =========================

    def registrar_imagen_de_alumno(
        self,
        matricula: str,
        ruta_imagen: str,
        ancho: int,
        alto: int,
        notas: Optional[str] = None
    ) -> None:

        sql = """
        INSERT INTO imagenes_alumno (
            matricula,
            ruta_imagen,
            ancho,
            alto,
            notas
        )
        VALUES (%s, %s, %s, %s, %s)
        """

        with self._conectar() as cnx:
            with cnx.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        matricula,
                        ruta_imagen,
                        int(ancho),
                        int(alto),
                        notas
                    )
                )

    def obtener_imagenes_de_alumno(
        self,
        matricula: str
    ) -> List[ImagenAlumno]:

        sql = """
        SELECT
            id,
            matricula,
            ruta_imagen,
            capturada_en,
            ancho,
            alto,
            notas
        FROM imagenes_alumno
        WHERE matricula = %s
        ORDER BY capturada_en DESC
        """

        with self._conectar() as cnx:
            with cnx.cursor() as cur:
                cur.execute(sql, (matricula,))
                rows = cur.fetchall()

        return [ImagenAlumno(*r) for r in rows]
