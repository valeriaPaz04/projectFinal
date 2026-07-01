import cv2
import mediapipe as mp
import csv
import numpy as np
import os
import time
from datetime import datetime

# ── CONFIGURACIÓN DE COLORES (más visibles) ──────────────────────────────
COLOR_TEXTO_PRINCIPAL = (0, 0, 0)        # Negro (para texto principal)
COLOR_TEXTO_SECUNDARIO = (50, 50, 50)    # Gris oscuro
COLOR_TEXTO_HUD = (200, 200, 200)        # Gris claro (para HUD)
COLOR_TEXTO_EXITO = (0, 200, 255)        # Naranja (para mensajes de éxito)
COLOR_BARRA = (0, 200, 80)               # Verde oscuro (barra de progreso)
COLOR_BARRA_COMPLETA = (0, 255, 0)       # Verde (barra completa)
COLOR_BORDE_BARRA = (100, 100, 100)      # Gris oscuro

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ── Crear carpetas necesarias ──────────────────────────────────────────────
CARPETA_DATASET = "dataset"
if not os.path.exists(CARPETA_DATASET):
    os.makedirs(CARPETA_DATASET)
    print(f"📁 Carpeta '{CARPETA_DATASET}' creada")

CARPETA_FOTOS = "fotos_gestos"
if not os.path.exists(CARPETA_FOTOS):
    os.makedirs(CARPETA_FOTOS)
    print(f"📁 Carpeta '{CARPETA_FOTOS}' creada")

# ── Archivo de dataset ──────────────────────────────────────────────────────
ARCHIVO = os.path.join(CARPETA_DATASET, "gestos.csv")
archivo_existe = os.path.isfile(ARCHIVO)

archivo = open(ARCHIVO, "a", newline="")
writer = csv.writer(archivo)

if not archivo_existe:
    headers = []
    for i in range(21):
        headers.append(f"x{i}")
    for i in range(21):
        headers.append(f"y{i}")
    for i in range(21):
        headers.append(f"z{i}")
    headers += ["is_right", "is_left", "gesto"]
    writer.writerow(headers)
    print(f"📄 Archivo '{ARCHIVO}' creado con encabezados")

gesto = input("Nombre del gesto: ")
objetivo = int(input("Número de muestras (recomendado 300): ") or 300)

contador = 0
ultimo_guardado = 0
COOLDOWN = 0.05
modo_auto = False

# ── Variable para controlar que solo se guarde UNA foto ──────────────────
foto_ya_guardada = False

def normalizar_puntos(landmarks):
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origen = puntos[0].copy()
    puntos -= origen
    escala = np.max(np.linalg.norm(puntos, axis=1))
    if escala > 0:
        puntos /= escala
    return puntos.flatten().tolist()

def dibujar_barra_progreso(frame, contador, objetivo):
    h, w = frame.shape[:2]
    pct = contador / objetivo
    bw = w - 40
    bh = 18
    by = h - 40

    # Fondo barra
    cv2.rectangle(frame, (20, by), (20 + bw, by + bh), (60, 60, 60), -1)
    # Relleno progreso
    color = COLOR_BARRA if pct < 0.8 else COLOR_BARRA_COMPLETA
    cv2.rectangle(frame, (20, by), (20 + int(bw * pct), by + bh), color, -1)
    # Borde
    cv2.rectangle(frame, (20, by), (20 + bw, by + bh), COLOR_BORDE_BARRA, 1)
    # Texto - EN NEGRO para que se vea mejor
    texto = f"{contador}/{objetivo} muestras"
    cv2.putText(frame, texto, (20, by - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 2, cv2.LINE_AA)

def guardar_foto_unica(frame, gesto, handedness):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{gesto}_{handedness}_referencia_{timestamp}.jpg"
    ruta_completa = os.path.join(CARPETA_FOTOS, nombre_archivo)
    cv2.imwrite(ruta_completa, frame)
    return ruta_completa

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    ahora = time.time()
    foto_guardada = False

    if result.multi_hand_landmarks:
        for i, hand in enumerate(result.multi_hand_landmarks):

            puntos_norm = normalizar_puntos(hand.landmark)
            handedness = result.multi_handedness[i].classification[0].label

            # Mostrar instrucciones - AHORA EN NEGRO
            modo_txt = "AUTO" if modo_auto else "tecla S"
            estado_foto = "✅ Foto guardada" if foto_ya_guardada else "📷 Foto pendiente"
            
            # Texto principal en NEGRO con fondo claro para visibilidad
            texto_info = f"[{handedness}] Gesto: {gesto} | Guardar: {modo_txt} | {estado_foto}"
            
            # Fondo semitransparente para el texto (para que se vea mejor)
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (len(texto_info) * 14 + 10, 45), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Texto en NEGRO y más grande
            cv2.putText(frame, texto_info,
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Guardar muestra
            guardar = False
            if modo_auto and (ahora - ultimo_guardado) > COOLDOWN:
                guardar = True
            elif cv2.waitKey(1) & 0xFF == ord("s"):
                guardar = True

            if guardar and contador < objetivo:
                lado = [1, 0] if handedness == "Right" else [0, 1]
                writer.writerow(puntos_norm + lado + [gesto])
                contador += 1
                ultimo_guardado = ahora
                
                if not foto_ya_guardada:
                    try:
                        ruta_foto = guardar_foto_unica(frame, gesto, handedness)
                        foto_ya_guardada = True
                        foto_guardada = True
                        print(f"📸 Foto de referencia guardada: {ruta_foto}")
                    except Exception as e:
                        print(f"❌ Error al guardar foto: {e}")

            # Dibujar landmarks
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=2)
            )

    # Mostrar mensaje si se guardó foto - EN NEGRO
    if foto_guardada:
        cv2.putText(frame, "📸 FOTO DE REFERENCIA GUARDADA", 
                    (frame.shape[1] - 320, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD superior - EN NEGRO
    texto_hud = f"A: modo auto | S: guardar | ESC: salir | Fotos en: {CARPETA_FOTOS}"
    cv2.putText(frame, texto_hud,
                (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    dibujar_barra_progreso(frame, contador, objetivo)

    cv2.imshow("Captura de gestos", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord("a"):
        modo_auto = not modo_auto
        print(f"Modo auto: {'ON' if modo_auto else 'OFF'}")

    if contador >= objetivo:
        print(f"✓ Dataset completo: {contador} muestras de '{gesto}'")
        if foto_ya_guardada:
            print(f"📸 Foto de referencia guardada en: {CARPETA_FOTOS}/")
        else:
            print(f"⚠️  No se guardó foto de referencia")
        
        cv2.putText(frame, "¡Dataset completo! Presiona ESC",
                    (80, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Captura de gestos", frame)
        cv2.waitKey(2000)
        break

cap.release()
archivo.close()
cv2.destroyAllWindows()

total_lineas = sum(1 for _ in open(ARCHIVO))
print(f"\n📊 Resumen final:")
print(f"  - Dataset: {total_lineas} muestras en '{ARCHIVO}'")
print(f"  - Fotos: {'1 foto de referencia' if foto_ya_guardada else '0 fotos'} en '{CARPETA_FOTOS}/'")
print(f"  - Gesto: '{gesto}'")
print(f"  - Modo auto: {'ON' if modo_auto else 'OFF'}")