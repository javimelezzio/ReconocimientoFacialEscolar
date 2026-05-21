
import os
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (96, 96)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "capturas"

MODEL_OUT = BASE_DIR / "face_cnn.keras"
ENCODER_OUT = BASE_DIR / "label_encoder.npy"

# Mínimo de imágenes por matrícula
MIN_IMGS_PER_CLASS = 2


def extraer_label_desde_nombre(nombre_archivo: str) -> str | None:

    if "_" not in nombre_archivo:
        return None

    return nombre_archivo.split("_", 1)[0].strip() or None


def cargar_dataset(data_dir: Path, img_size=(96, 96)):

    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {data_dir.resolve()}"
        )

    X = []
    y = []

    for img_path in data_dir.rglob("*"):

        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ):
            continue

        label = extraer_label_desde_nombre(img_path.name)

        if not label:
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        try:

            # Escala de grises
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            # Redimensionar
            gray = cv2.resize(
                gray,
                img_size,
                interpolation=cv2.INTER_AREA
            )

        except Exception:
            continue

        X.append(gray)
        y.append(label)

    if len(X) < 10:
        raise RuntimeError(
            f"Muy pocas imágenes ({len(X)}). "
            f"Junta más capturas."
        )

    X = np.array(X, dtype=np.float32) / 255.0
    X = np.expand_dims(X, axis=-1)

    y = np.array(y, dtype=object)

    return X, y


def construir_modelo(
    num_clases: int,
    input_shape=(96, 96, 1)
):

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=input_shape),

        data_augmentation,

        tf.keras.layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            num_clases,
            activation="softmax"
        ),
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],
    )

    return model


def main():

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print(
        "Leyendo capturas desde:",
        DATA_DIR.resolve()
    )

    X, y_labels = cargar_dataset(
        DATA_DIR,
        IMG_SIZE
    )

    conteo = Counter(y_labels.tolist())

    print("\nImágenes por matrícula:")

    for k, v in sorted(
        conteo.items(),
        key=lambda kv: (-kv[1], kv[0])
    ):

        print(f"  {k}: {v}")

    # Filtrar clases con pocas imágenes
    clases_validas = {
        c
        for c, n in conteo.items()
        if n >= MIN_IMGS_PER_CLASS
    }

    if len(clases_validas) < 2:

        raise RuntimeError(
            f"No hay suficientes clases "
            f"con >= {MIN_IMGS_PER_CLASS} imágenes.\n"
            f"Clases válidas: "
            f"{sorted(clases_validas)}"
        )

    mask = np.array([
        lbl in clases_validas
        for lbl in y_labels
    ], dtype=bool)

    X = X[mask]
    y_labels = y_labels[mask]

    print(
        f"\nUsando {len(X)} imágenes "
        f"tras filtrar clases."
    )

    conteo2 = Counter(y_labels.tolist())

    print(
        "Clases finales:",
        sorted(conteo2.keys())
    )

    # Encoder
    clases = sorted(conteo2.keys())

    class_to_idx = {
        c: i
        for i, c in enumerate(clases)
    }

    y = np.array([
        class_to_idx[c]
        for c in y_labels
    ], dtype=np.int64)

    np.save(
        str(ENCODER_OUT),
        np.array(clases, dtype=object),
        allow_pickle=True
    )

    # División entrenamiento / validación
    X_train, X_val, y_train, y_val = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    model = construir_modelo(

        num_clases=len(clases),

        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            1
        ),
    )

    model.summary()

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_accuracy",
            save_best_only=True
        ),
    ]

    # Entrenamiento
    model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=30,

        batch_size=32,

        callbacks=callbacks,

        verbose=1,
    )

    # Evaluación final
    loss, acc = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    print(f"\nAccuracy validación: {acc:.4f}")
    print(f"Loss validación: {loss:.4f}")

    # Guardar modelo
    model.save(str(MODEL_OUT))

    print("\n================================")
    print("ENTRENAMIENTO FINALIZADO")
    print("================================")

    print(
        "Modelo guardado en:",
        MODEL_OUT.resolve()
    )

    print(
        "Encoder guardado en:",
        ENCODER_OUT.resolve()
    )

    print("Clases:", clases)


if __name__ == "__main__":
    main()
