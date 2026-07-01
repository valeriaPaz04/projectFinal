---
title: sistemaSalud
emoji: 🖐️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
python_version: "3.10"
---

# SeñaSalud – Comunicación en Lenguaje de Señas para Consultas Médicas

Sistema de visión por computadora que reconoce el alfabeto de la Lengua de Señas Colombiana (LSC) letra por letra en tiempo real, a través de una cámara web, y va armando palabras y frases completas — incluyendo gestos dedicados para **espacio** y **borrar** — para facilitar la comunicación de personas sordas o con discapacidad del habla dentro de un centro de salud.

## Sector y contexto del caso (ficticio)

**Sector:** Salud.

La **Clínica San Rafael** (institución ficticia) atiende a diario pacientes con discapacidad auditiva o del habla. Actualmente, cuando no hay un intérprete de lengua de señas disponible —algo frecuente fuera de horario administrativo o en turnos de urgencias—, la comunicación entre el paciente y el personal médico se vuelve lenta, confusa y propensa a errores durante el triaje y la consulta.

**SeñaSalud** propone un asistente de comunicación por cámara, disponible en cada consultorio, que permite al paciente deletrear en LSC lo que necesita decir (síntomas, molestias, solicitudes) y ver la frase construida en pantalla en tiempo real, para que el personal de salud la lea directamente, sin depender de un intérprete presencial.

## Modelo de visión por computadora utilizado

- **MediaPipe Hands** (Google) — modelo preentrenado de estimación de pose de manos, que detecta 21 puntos clave (landmarks) de la mano en tiempo real a partir de la imagen de la cámara.
- **Clasificador propio** (RandomForest, entrenado sobre los landmarks normalizados de MediaPipe) que traduce esos puntos clave en la letra o gesto correspondiente (26 letras del alfabeto + gesto de `espacio` + gesto de `borrar`).

**Justificación:** MediaPipe Hands es liviano, corre en tiempo real sin necesidad de GPU y es un estándar robusto y ampliamente usado para estimación de pose de manos, lo que lo hace ideal para una aplicación que debe funcionar de forma fluida en un consultorio con hardware modesto. En lugar de entrenar un modelo de visión desde cero, se aprovecha MediaPipe como extractor de características y se entrena un clasificador simple y rápido sobre esos landmarks, lo que reduce drásticamente el tiempo y los datos necesarios para reconocer los gestos de la LSC.

## Funcionalidades

- Reconocimiento en tiempo real de las 26 letras del alfabeto en LSC mediante la cámara.
- Gesto de **espacio**: inserta un espacio en la frase que se está construyendo.
- Gesto de **borrar**: elimina la última letra escrita (para corregir errores).
- Suavizado temporal por votación (evita que ruido de un solo cuadro cambie la letra detectada).
- Panel en pantalla con la frase construida en tiempo real, historial de los últimos gestos detectados y nivel de confianza de cada predicción.
- Respaldo por teclado (`BACKSPACE` para borrar, `barra espaciadora` para espacio) por si el gesto no se detecta bien.

## Estado actual del proyecto

- ✅ Captura de dataset propio de gestos (`1-capturar_gestos.py`), entrenamiento del clasificador (`2-entrenar_modelo.py`) y reconocimiento en tiempo real (`3-reconocer_gestos.py`) funcionando localmente como aplicación de escritorio (OpenCV).
- ✅ Aplicación web (`app.py`) construida con **Streamlit** y **streamlit-webrtc**, que reutiliza la misma lógica de reconocimiento y captura la cámara directamente desde el navegador. Probada localmente.
- ✅ Desplegada en **Hugging Face Spaces**, con sincronización automática desde este repositorio mediante GitHub Actions (`.github/workflows/sync-to-hf.yml`) en cada push a `main`.
- ⚠️ Limitación conocida: las letras **F** y **Z** tienen una tasa de reconocimiento más baja que el resto del alfabeto (uso poco frecuente, pendiente de mejora con más datos de entrenamiento).

## Enlace a la aplicación desplegada

🔗 https://huggingface.co/spaces/valeriaPaz04/sistemaSalud

## Instalación y ejecución local (versión actual)

1. Clonar el repositorio y entrar a la carpeta del proyecto:
   ```bash
   git clone https://github.com/valeriaPaz04/projectFinal.git
   cd projectFinal
   ```
2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Linux / macOS
   source venv/bin/activate
   ```
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. **Opción A — Aplicación web (Streamlit, recomendada):**
   ```bash
   streamlit run app.py
   ```
   Se abre en el navegador en `http://localhost:8501`. Presiona **START** para activar la cámara.

   **Opción B — Script de escritorio (ventana OpenCV):**
   ```bash
   pip install opencv-python   # variante con interfaz gráfica (no la headless de requirements.txt)
   python 3-reconocer_gestos.py
   ```
   - `ESC` para salir.
   - Gesto `espacio` o barra espaciadora → agrega un espacio.
   - Gesto `borrar` o `BACKSPACE` → borra la última letra.

### (Opcional) Recapturar datos y reentrenar el modelo

```bash
python 1-capturar_gestos.py    # captura muestras de un gesto (letra, "espacio" o "borrar")
python 2-entrenar_modelo.py    # reentrena el clasificador con el dataset actualizado
```

## Estructura del proyecto

```
projectFinal/
├── app.py                   # Aplicación web (Streamlit + streamlit-webrtc)
├── 1-capturar_gestos.py     # Captura de muestras de gestos (landmarks de MediaPipe)
├── 2-entrenar_modelo.py     # Entrenamiento y evaluación del clasificador
├── 3-reconocer_gestos.py    # Versión de escritorio (OpenCV) del reconocimiento en tiempo real
├── requirements.txt         # Dependencias de Python (para local y despliegue)
├── packages.txt             # Dependencias del sistema (apt) para el despliegue
└── models/
    ├── modelo_gestos.pkl       # Clasificador entrenado
    ├── scaler_gestos.pkl       # Escalador de características
    ├── info_modelo.txt         # Métricas y metadatos del entrenamiento
    └── diagnostico_modelo.png  # Gráficas de desempeño del modelo
```

## Integrantes del equipo y roles

| Integrante | Rol |
|---|---|
| Valeria Paz Arana | Desarrollo integral: captura de datos, entrenamiento del modelo, integración de MediaPipe, lógica de reconocimiento y armado de frases, documentación y despliegue |
| Yeferson Andrés Moreno Granda | Ideación y definición del caso de uso del proyecto |
| Santiago Suárez Duran | Investigación y definición del sector/contexto del caso ficticio |
| Sebastián Valencia Pérez | Pruebas del sistema (validación del reconocimiento de gestos) |
