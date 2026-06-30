import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
import time

# ── Cargar modelo ─────────────────────────────────────────────────────────────
modelo = joblib.load("models/modelo_gestos.pkl")
clases = modelo.classes_

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ── Parámetros ────────────────────────────────────────────────────────────────
UMBRAL = 0.75           # confianza mínima
BUFFER_SIZE = 12        # frames para votación por mayoría (suavizado temporal)
HISTORIAL_MAX = 5       # gestos recientes mostrados

buffers_prediccion = {}  # un buffer por mano detectada
historial_gestos = deque(maxlen=HISTORIAL_MAX)
ultimo_gesto = ""

fps_contador = 0
fps_valor = 0.0
t_fps = time.time()


def normalizar_puntos(landmarks):
    """Misma normalización que en la captura."""
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origen = puntos[0].copy()
    puntos -= origen
    escala = np.max(np.linalg.norm(puntos, axis=1))
    if escala > 0:
        puntos /= escala
    return puntos.flatten().tolist()


def dibujar_barra_confianza(frame, x, y, confianza, label, color):
    """Barra horizontal de confianza debajo de la etiqueta del gesto."""
    bw = 200
    bh = 10
    filled = int(bw * confianza)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bh), color, -1)
    cv2.putText(frame, f"{confianza*100:.0f}%", (x + bw + 6, y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def dibujar_panel_gesto(frame, gesto, confianza, x_off=20, y_off=20):
    """Caja semitransparente con nombre del gesto y barra de confianza."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_off, y_off), (x_off + 280, y_off + 80), (25, 25, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    if confianza > 0.90:
        color = (0, 255, 120)
    elif confianza > 0.80:
        color = (0, 200, 255)
    else:
        color = (0, 160, 255)

    cv2.putText(frame, gesto, (x_off + 14, y_off + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

    dibujar_barra_confianza(frame, x_off + 14, y_off + 58, confianza, gesto, color)


def dibujar_historial(frame, historial):
    """Muestra los últimos gestos detectados en la esquina inferior."""
    h, w = frame.shape[:2]
    y_base = h - 20
    cv2.putText(frame, "Historial:", (10, y_base - len(historial) * 20 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    for i, g in enumerate(reversed(historial)):
        alpha = 1.0 - (i * 0.18)
        color = tuple(int(c * alpha) for c in (160, 210, 255))
        cv2.putText(frame, g, (10, y_base - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ── Bucle principal ───────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # FPS
    fps_contador += 1
    if time.time() - t_fps >= 1.0:
        fps_valor = fps_contador / (time.time() - t_fps)
        fps_contador = 0
        t_fps = time.time()
    cv2.putText(frame, f"FPS: {fps_valor:.0f}", (frame.shape[1] - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    if result.multi_hand_landmarks:
        for i, hand in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[i].classification[0].label
            key = f"hand_{i}_{handedness}"

            if key not in buffers_prediccion:
                buffers_prediccion[key] = deque(maxlen=BUFFER_SIZE)

            puntos_norm = normalizar_puntos(hand.landmark)
            lado = [1, 0] if handedness == "Right" else [0, 1]
            features = puntos_norm + lado

            # Predicción frame actual
            probs = modelo.predict_proba([features])[0]
            idx = np.argmax(probs)
            confianza_frame = probs[idx]

            if confianza_frame > UMBRAL:
                buffers_prediccion[key].append(clases[idx])

            # ── Votación por mayoría (suavizado temporal) ──────────────────
            if len(buffers_prediccion[key]) >= BUFFER_SIZE // 2:
                votos = Counter(buffers_prediccion[key])
                gesto_final, votos_ganador = votos.most_common(1)[0]
                confianza_suavizada = votos_ganador / len(buffers_prediccion[key])

                if confianza_suavizada >= 0.6:
                    y_panel = 20 + i * 110
                    dibujar_panel_gesto(frame, f"{handedness}: {gesto_final}",
                                        confianza_suavizada, y_off=y_panel)

                    # Añadir al historial cuando cambia el gesto
                    etiqueta = f"{handedness[0]}: {gesto_final}"
                    if etiqueta != ultimo_gesto:
                        historial_gestos.append(etiqueta)
                        ultimo_gesto = etiqueta

            # Dibujar mano
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(100, 180, 255), thickness=2)
            )

    else:
        # Sin manos → limpiar buffers
        buffers_prediccion.clear()

    dibujar_historial(frame, historial_gestos)

    cv2.putText(frame, "ESC para salir",
                (frame.shape[1] - 150, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.imshow("Reconocimiento de gestos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()