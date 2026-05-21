from dataclasses import dataclass

@dataclass(frozen=True)
class ConfigMySQL:
    # cambiar por credenciales reales en producción

    host: str = "localhost"
    port: int = 3306
    user: str = "usuario_demo"
    password: str = "password_demo"
    database: str = "reconocimiento_escolar"
