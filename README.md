# ReconocimientoFacialEscolar

En este repositorio **presentamos** un sistema de **reconocimiento facial** desarrollado en **Python**, orientado a un entorno escolar. El objetivo del proyecto es permitir la **captura y detección de rostros**, el **registro de alumnos**, la **persistencia de información en MySQL** y la **identificación en tiempo real** desde una interfaz gráfica construida con **Tkinter**.

El sistema integra, principalmente:

- Captura desde cámara con OpenCV.
- Detección de rostros (Haar Cascade).
- Registro y actualización de alumnos.
- Persistencia en base de datos **MySQL** (alumnos e imágenes capturadas).
- Entrenamiento de un modelo CNN con TensorFlow/Keras para reconocer matrícula.
- Interfaz gráfica (Tkinter) para operar el flujo completo desde una sola aplicación.

---

## Requisitos

### Software / Entorno
- **Python 3.x**
- **MySQL Server** (requerido para almacenar alumnos e imágenes)
- **Tkinter** (normalmente viene con Python; depende del sistema operativo/instalación)
- Cámara web accesible por OpenCV

### Dependencias Python
En el repositorio se incluye el archivo `requiments.txt` (así está nombrado), donde se listan dependencias y componentes del entorno. En forma resumida, utilizamos:

- `opencv-python`
- `numpy`
- `Pillow`
- `mysql-connector-python`
- `tensorflow`
- `scikit-learn`

---

## Instalación (recomendado con entorno virtual)

Clonamos el repositorio, creamos un entorno virtual y luego instalamos dependencias:

```bash
git clone https://github.com/javimelezzio/ReconocimientoFacialEscolar.git
cd ReconocimientoFacialEscolar

python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install opencv-python numpy pillow mysql-connector-python tensorflow scikit-learn
```

> Nota: si en alguna rama se agrega un `requirements.txt` estándar, podemos instalar con `pip install -r requirements.txt`. Por ahora, el repositorio documenta dependencias en `requiments.txt`.

---

## Configuración de MySQL

Para que el sistema funcione correctamente con persistencia, primero preparamos la base de datos y configuramos el acceso:

1. **Crear el esquema en MySQL**  
   En el README del repositorio se referencia el script `01_esquema_mysql.sql` (ver la carpeta `database/`).

2. **Configurar credenciales de conexión**  
   Ajustamos los parámetros en `config_mysql.py` (host, puerto, usuario, contraseña y nombre de base de datos).

El esquema considera (según la documentación del repositorio):
- Base `reconocimiento_escolar`
- Tabla `alumnos`
- Tabla `imagenes_alumno` (con relación FK hacia `alumnos`)

---

## Ejecución (flujo recomendado)

A continuación describimos el flujo recomendado para operar el sistema desde cero (captura → dataset → entrenamiento → reconocimiento):

### 1) Ejecutar la aplicación (interfaz) y capturar rostros

Podemos iniciar la aplicación de cualquiera de estas dos formas:

**Opción A (entrypoint del repositorio):**
```bash
python main.py
```

**Opción B (ejecución directa de interfaz, si aplica):**
```bash
python interfaz_tkinter.py
```

Desde la interfaz:
- Abrimos la cámara y visualizamos el video en vivo.
- El sistema detecta el rostro y lo encuadra.
- Registramos o actualizamos al alumno (tabla `alumnos`).
- Guardamos la imagen (cara detectada) en `data/capturas/` y registramos la captura en MySQL (tabla `imagenes_alumno`).

---

### 2) (Opcional) Preparar dataset de entrenamiento

Si deseamos entrenar un modelo con base en las capturas, preparamos el dataset separándolo por matrícula:

```bash
python armar_dataset_desde_capturas.py
```

Si preferimos **mover** las imágenes en lugar de copiarlas:
```bash
python armar_dataset_desde_capturas.py --mover
```

Estructura esperada:
- `data/train/<matricula>/...`

---

### 3) Entrenar el modelo

Ejecutamos el entrenamiento del modelo CNN:

```bash
python entrenar_modelo.py
```

Al finalizar, el entrenamiento genera los siguientes archivos:

- `face_cnn.keras` (modelo entrenado)
- `label_encoder.npy` (codificador de clases / matrículas)

---

### 4) Reconocimiento en vivo

Una vez que existen `face_cnn.keras` y `label_encoder.npy`, iniciamos nuevamente la aplicación para habilitar predicción. En este modo, el sistema:

- Predice la **matrícula** con el modelo CNN.
- Calcula y expone la confianza.
- Consulta los datos del alumno en MySQL para mostrarlos en la interfaz.

Ejecutamos:
```bash
python main.py
```

---

## Descripción de componentes principales

### `main.py`
Es el punto de entrada del proyecto. Desde ahí iniciamos la aplicación llamando a `iniciar_app()` (ubicada en el módulo de interfaz).

### Interfaz (`interfaz_tkinter.py` / carpeta `interfaz/`)
Aquí se implementa la interfaz gráfica con Tkinter. La interfaz:
- Abre la cámara con OpenCV (`cv2.VideoCapture(0)`).
- Muestra el video en vivo y dibuja un rectángulo sobre el rostro detectado.
- Integra el motor de reconocimiento para predicción (cuando el modelo está disponible).
- Permite registrar/actualizar alumnos y guardar capturas.

### Motor (`motor_reconocimiento.py`)
Es el motor central de captura y reconocimiento:
- Detecta rostros con Haar Cascade.
- Preprocesa el rostro (escala de grises, `resize` a 96x96 y normalización).
- Si existen `face_cnn.keras` y `label_encoder.npy`, carga el modelo y predice la matrícula.
- Guarda imágenes detectadas (por ejemplo en `data/capturas/<matricula>_<timestamp>.jpg`) y registra la captura en MySQL.

### Persistencia (`repositorio_mysql.py`)
Implementa la capa de acceso a datos MySQL. Incluye estructuras (dataclasses) y funciones para:
- Guardar o actualizar un alumno.
- Consultar un alumno por matrícula.
- Listar alumnos.
- Registrar imágenes capturadas asociadas a un alumno.

### Configuración (`config_mysql.py`)
Contiene la configuración de conexión a MySQL (host, puerto, usuario, contraseña y base de datos).

### `database/`, `entrenamiento/`, `reconocimiento/`
Carpetas del repositorio que agrupan scripts SQL y módulos relacionados con entrenamiento y reconocimiento.

---

## Archivos generados (runtime)

Para utilizar reconocimiento (no solo captura), generamos y dejamos disponibles en el proyecto:

- `face_cnn.keras`
- `label_encoder.npy`

---

## Notas y recomendaciones

- En Windows, TensorFlow normalmente se ejecuta en CPU; esto es esperado.
- Para obtener un reconocimiento más estable, recomendamos contar con **30 o más capturas por matrícula**.
- Si los archivos `face_cnn.keras` y `label_encoder.npy` no existen, el sistema puede operar en modo de **captura**, pero no realizará reconocimiento.

---

## Equipo de desarrollo y responsabilidades

Este proyecto fue desarrollado en equipo, distribuyendo las cargas de trabajo en tres áreas principales: gestión/integración, base de datos y diseño de interfaz.

### Gestión, Integración y Pruebas
* [@javimelezzio](https://github.com/javimelezzio) – **HERNÁNDEZ MELECIO ROBERTO JAVIER**
    * Líder y administrador del repositorio en GitHub (control de versiones y ramas).
    * Encargado del despliegue, pruebas de caja negra y validación de la integración del sistema (interfaz + motor CNN + MySQL).

### Base de Datos 
* [@pinmode17](https://github.com/pinmode17) – **SÁNCHEZ FLORES JOSÉ MANUEL**
    * Diseño del modelo entidad-relación y creación del esquema (`database/01_esquema_mysql.sql`).
    * Definición de las estructuras de datos y llaves foráneas para optimizar el almacenamiento de imágenes binarias.
* [@1237-the](https://github.com/1237-the) – **DE JESÚS MENDOZA FRANCISCO JAVIER**
    * Implementación de la capa de datos en Python (`repositorio_mysql.py`).
    * Desarrollo de scripts de conexión, consultas de matrículas y control de excepciones en `config_mysql.py`.

### Diseño de Interfaz (GUI) y Experiencia de Usuario
* [@jessicaasc60-wq](https://github.com/jessicaasc60-wq) – **JESSICA MARÍN ASCENCIO**
    * Líder del equipo de diseño gráfico y maquetación de ventanas.
    * Desarrollo principal de la vista en `interfaz_tkinter.py` y renderizado del flujo de la cámara en tiempo real.
* [@luzbel999](https://github.com/luzbel999) – **AGUILAR MÉRIDA ERIC ALEXANDER**
    * Co-diseñador de la interfaz; encargado de la lógica de los botones para el registro, actualización de alumnos y control de eventos de la GUI.
* [@angeljuarezcruz87-cloud](https://github.com/angeljuarezcruz87-cloud) – **JUÁREZ CRUZ ÁNGEL DAVID**
    * Diseño visual complementario y encargado de las pruebas de usabilidad de la interfaz gráfica.
* **CASTAÑEDA FARFÁN ITALY VALERIA**
    * Maquetación de la propuesta visual estética y revisión de la consistencia de colores y tipografías en la aplicación de escritorio.

## Licencia

Este repositorio incluye un archivo `LICENSE`. Para más detalle, consultar dicho archivo.
