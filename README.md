# Proyecto ReconocimientoFacial

Sistema de **reconocimiento facial** hecho en **Python** que integra:

- Captura desde cámara
- Detección de rostros
- Registro de alumnos
- Persistencia en base de datos MySQL
- Interfaz gráfica con Tkinter

---

## ¿Cómo se ejecuta? (flujo recomendado)

1) **Crear la base de datos** en MySQL ejecutando:
   - `01_esquema_mysql.sql`

2) **Configurar credenciales** de MySQL en:
   - `config_mysql.py`

3) **Capturar caras** desde la interfaz (se guardan en `data/capturas`):
   - Ejecuta `interfaz_tkinter.py`
   - Ingresa una matrícula y usa **Guardar imagen (cara detectada)**.

4) **(Opcional) Armar dataset de entrenamiento** desde `data/capturas` a `data/train/<matricula>/...`:
   - Ejecuta `armar_dataset_desde_capturas.py`

5) **Entrenar el modelo** para generar:
   - `face_cnn.keras`
   - `label_encoder.npy`

   Ejecuta `entrenar_modelo.py`.

6) **Ejecutar el reconocimiento en vivo**:
   - Ejecuta `interfaz_tkinter.py`

---

## Descripción de archivos (qué hace cada uno)

### `interfaz_tkinter.py`
Interfaz gráfica principal con Tkinter.

- Abre la cámara con OpenCV (`cv2.VideoCapture(0)`).
- Muestra video en vivo y dibuja un rectángulo sobre el rostro detectado.
- Llama al motor (`MotorReconocimientoFacial.predecir`) para obtener:
  - matrícula predicha
  - confianza
  - datos del alumno desde MySQL (nombre, inscrito, carrera)
- Permite:
  - **Registrar / Actualizar alumno** (tabla `alumnos`).
  - **Guardar imagen (cara detectada)** (archivo en `data/capturas` + registro en MySQL).

**Archivo a ejecutar para usar el sistema:** `python interfaz_tkinter.py`

---

### `motor_reconocimiento.py`
Motor central de reconocimiento / captura.

- Detecta rostros con **OpenCV Haar Cascade**.
- Preprocesa el rostro (grises, resize 96x96, normalización).
- Si existen `face_cnn.keras` y `label_encoder.npy`:
  - carga el modelo CNN (TensorFlow/Keras)
  - predice la **matrícula**
  - consulta el alumno con `RepositorioMySQL.obtener_alumno`
- Si el modelo no existe, trabaja en modo **"solo captura"**.
- Guarda imágenes detectadas con `guardar_cara_detectada()`:
  - escribe el JPG en `data/capturas/<matricula>_<timestamp>.jpg`
  - registra la imagen en MySQL (`imagenes_alumno`).

---

### `repositorio_mysql.py`
Capa de persistencia en MySQL.

Incluye dos dataclasses:
- `Alumno`: (matricula, nombre, inscrito, carrera)
- `ImagenAlumno`: metadatos de una captura

Funciones principales:
- `guardar_o_actualizar_alumno(...)`: inserta o actualiza alumno.
- `obtener_alumno(matricula)`: devuelve un `Alumno` o `None`.
- `listar_alumnos()`: lista todos los alumnos.
- `registrar_imagen_de_alumno(...)`: inserta registro de captura.

---

### `config_mysql.py`
Configuración de conexión a MySQL.

- Define `ConfigMySQL` con: host, port, user, password, database.
- Ajusta estos valores según tu instalación.

---

### `01_esquema_mysql.sql`
Script SQL para inicializar la base de datos.

Crea:
- Base `reconocimiento_escolar`
- Tabla `alumnos`
- Tabla `imagenes_alumno` (relación FK hacia `alumnos`)

---

### `armar_dataset_desde_capturas.py`
Utilidad para preparar datos de entrenamiento.

- Lee imágenes en `data/capturas`.
- Extrae matrícula del nombre `MATRICULA_...jpg`.
- Copia (o mueve con `--mover`) cada imagen a:
  - `data/train/<matricula>/<archivo>.jpg`

Uso:
- Copiar: `python armar_dataset_desde_capturas.py`
- Mover: `python armar_dataset_desde_capturas.py --mover`

---

### `entrenar_modelo.py`
Entrenamiento del modelo CNN.

- Lee imágenes desde `data/capturas` (por nombre `MATRICULA_...jpg`).
- Filtra matrículas con menos de `MIN_IMGS_PER_CLASS` imágenes.
- Entrena una CNN sencilla (Conv2D + MaxPool + Dense).
- Guarda:
  - `face_cnn.keras` (modelo)
  - `label_encoder.npy` (clases/matrículas)

Uso:
- `python entrenar_modelo.py`

---

## Notas

- En Windows, TensorFlow normalmente corre en CPU (eso es esperado).
- Para un reconocimiento estable, se recomienda tener **30+ capturas por matrícula**.

---
# Roles finales

- [@luzbel999](https://github.com/luzbel999) – **AGUILAR MÉRIDA ERIC ALEXANDER**  
  Encargado del diseño, trabajando en conjunto con el responsable principal para definir la estructura funcional del proyecto. 
- []() – **CASTAÑEDA FARFÁN ITALY VALERIA**  
  Apoyo en el área de diseño, contribuyendo a la definición estética y visual del proyecto, así como en la propuesta de mejoras para la interfaz.  

- [@1237-the](https://github.com/1237-the) – **DE JESÚS MENDOZA FRANCISCO JAVIER**  
  Responsable de la implementación y adaptación de la base de datos MySQL, asegurando la correcta migración y el manejo eficiente de la información.  

- [@javimelezzio](https://github.com/javimelezzio) – **HERNÁNDEZ MELECIO ROBERTO JAVIER**  
  Responsable de la gestión integral del proyecto en GitHub, asegurando la correcta organización de los archivos y versiones. Encargado de realizar pruebas de funcionamiento y validar la integración entre la base de datos, la interfaz y los módulos de procesamiento.  

- [@jessicaasc60-wq](https://github.com/jessicaasc60-wq) – **JESSICA MARÍN ASCENCIO**  
  Líder en diseño. Se encargó de estructurar la interfaz gráfica y coordinar la integración visual con la lógica del sistema, garantizando una experiencia de usuario clara y funcional.  

- [@angeljuarezcruz87-cloud](https://github.com/angeljuarezcruz87-cloud) – **JUÁREZ CRUZ ÁNGEL DAVID**  
  Colaborador en diseño y pruebas. Participó en la validación del sistema y en la retroalimentación sobre la experiencia de usuario, además de apoyar en tareas de diseño complementarias.  

- [@pinmode17](https://github.com/pinmode17) – **SÁNCHEZ FLORES JOSÉ MANUEL**  
  Responsable del diseño, creación e implementación de la base de datos. Colaboró en la definición de esquemas y relaciones, asegurando la consistencia y escalabilidad del sistema de almacenamiento.  adapta este a eso
---
