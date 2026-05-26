
import argparse
import re
import shutil
from pathlib import Path


# =========================
# DIRECTORIOS
# =========================

CAPTURAS_DIR = Path(
    "data/capturas"
)

TRAIN_DIR = Path(
    "data/train"
)

# =========================
# PATRÓN:
# matricula_archivo.jpg
# =========================

PATRON = re.compile(
    r"^([^_]+)_"
)


def obtener_imagenes():

    extensiones_validas = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    imagenes = []

    for p in CAPTURAS_DIR.iterdir():

        if (
            p.is_file()
            and
            p.suffix.lower() in extensiones_validas
        ):

            imagenes.append(p)

    return imagenes


def procesar_imagen(
    img_path: Path,
    mover: bool
) -> bool:

    m = PATRON.match(
        img_path.name
    )

    if not m:
        return False

    matricula = m.group(1).strip()

    if not matricula:
        return False

    # =========================
    # CARPETA DESTINO
    # data/train/A001/
    # =========================

    dst_dir = TRAIN_DIR / matricula

    dst_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dst_path = dst_dir / img_path.name

    # =========================
    # MOVER O COPIAR
    # =========================

    if mover:

        shutil.move(
            str(img_path),
            str(dst_path)
        )

    else:

        shutil.copy2(
            str(img_path),
            str(dst_path)
        )

    return True


def main(mover: bool):

    # =========================
    # VALIDAR DIRECTORIO
    # =========================

    if not CAPTURAS_DIR.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{CAPTURAS_DIR.resolve()}"
        )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    imgs = obtener_imagenes()

    if not imgs:

        print(
            "No encontré imágenes en:"
        )

        print(
            CAPTURAS_DIR.resolve()
        )

        return

    total = 0
    sin_matricula = 0

    print("\nProcesando imágenes...\n")

    for img_path in imgs:

        ok = procesar_imagen(
            img_path,
            mover
        )

        if ok:

            total += 1

            print(
                f"[OK] {img_path.name}"
            )

        else:

            sin_matricula += 1

            print(
                f"[IGNORADA] "
                f"{img_path.name}"
            )

    # =========================
    # RESUMEN FINAL
    # =========================

    print("\n=========================")
    print("PROCESO FINALIZADO")
    print("=========================")

    print(
        "Imágenes procesadas:",
        total
    )

    print(
        "Sin matrícula válida:",
        sin_matricula
    )

    print(
        "Dataset generado en:"
    )

    print(
        TRAIN_DIR.resolve()
    )

    print(
        "\nEjemplo estructura:"
    )

    print(
        "data/train/A001/*.jpg"
    )

    if mover:

        print(
            "\nModo usado: MOVER"
        )

    else:

        print(
            "\nModo usado: COPIAR"
        )

if __name__ == "__main__":

    ap = argparse.ArgumentParser(

        description=(
            "Organiza capturas "
            "faciales por matrícula."
        )
    )

    ap.add_argument(

        "--mover",

        action="store_true",

        help=(
            "Mueve imágenes "
            "en vez de copiarlas"
        )
    )

    args = ap.parse_args()

    main(
        mover=args.mover
  )
