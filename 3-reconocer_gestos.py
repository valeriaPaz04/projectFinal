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
UMBRAL = 0.6            # confianza mínima por cuadro
BUFFER_SIZE = 12        # frames para votación por mayoría (suavizado temporal)
MIN_BUFFER = 4          # cuadros válidos mínimos antes de intentar votar
CONFIANZA_CONFIRMAR = 0.55  # % de acuerdo en el buffer para confirmar el gesto
HISTORIAL_MAX = 5       # gestos recientes mostrados
PANEL_FRASE_ALTURA = 50  # alto de la barra inferior con la frase

GESTO_ESPACIO = "espacio"
GESTO_BORRAR = "borrar"

buffers_prediccion = {}  # un buffer por mano detectada
historial_gestos = deque(maxlen=HISTORIAL_MAX)
ultimo_gesto_por_mano = {}  # último gesto confirmado por cada mano (evita repetir mientras se sostiene)
frase = ""

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
    """Muestra los últimos gestos detectados encima de la barra de la frase."""
    h, w = frame.shape[:2]
    y_base = h - PANEL_FRASE_ALTURA - 16
    cv2.putText(frame, "Historial:", (10, y_base - len(historial) * 20 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    for i, g in enumerate(reversed(historial)):
        alpha = 1.0 - (i * 0.18)
        color = tuple(int(c * alpha) for c in (160, 210, 255))
        cv2.putText(frame, g, (10, y_base - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def dibujar_panel_frase(frame, frase):
    """Barra inferior con la frase que se va armando letra por letra."""
    h, w = frame.shape[:2]
    y0 = h - PANEL_FRASE_ALTURA
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    texto = f"Frase: {frase}_"
    cv2.putText(frame, texto, (16, y0 + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


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

            # Lectura cruda por cuadro (para diagnóstico), junto a la muñeca
            muneca = hand.landmark[0]
            x_px = int(muneca.x * frame.shape[1])
            y_px = int(muneca.y * frame.shape[0])
            cv2.putText(frame, f"{clases[idx]} {confianza_frame*100:.0f}%",
                        (x_px, y_px + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # ── Votación por mayoría (suavizado temporal) ──────────────────
            if len(buffers_prediccion[key]) >= MIN_BUFFER:
                votos = Counter(buffers_prediccion[key])
                gesto_final, votos_ganador = votos.most_common(1)[0]
                confianza_suavizada = votos_ganador / len(buffers_prediccion[key])

                if confianza_suavizada >= CONFIANZA_CONFIRMAR:
                    y_panel = 20 + i * 110
                    dibujar_panel_gesto(frame, f"{handedness}: {gesto_final}",
                                        confianza_suavizada, y_off=y_panel)

                    # Confirmar el gesto solo si es nuevo para esta mano
                    # (cambia de letra, o se soltó y volvió a mostrar la misma)
                    if gesto_final != ultimo_gesto_por_mano.get(key):
                        ultimo_gesto_por_mano[key] = gesto_final

                        etiqueta = f"{handedness[0]}: {gesto_final}"
                        historial_gestos.append(etiqueta)

                        if gesto_final == GESTO_ESPACIO:
                            frase += " "
                        elif gesto_final == GESTO_BORRAR:
                            frase = frase[:-1]
                        else:
                            frase += gesto_final

            # Dibujar mano
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(100, 180, 255), thickness=2)
            )

    else:
        # Sin manos → limpiar buffers y permitir reconfirmar el mismo gesto
        buffers_prediccion.clear()
        ultimo_gesto_por_mano.clear()

    dibujar_historial(frame, historial_gestos)
    dibujar_panel_frase(frame, frase)

    cv2.putText(frame, "ESC: salir | SUPR: borrar | ESPACIO: espacio",
                (frame.shape[1] - 340, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.imshow("Reconocimiento de gestos", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:        # ESC
        break
    elif tecla == 8:        # BACKSPACE (respaldo de teclado)
        frase = frase[:-1]
    elif tecla == 32:        # SPACE (respaldo de teclado)
        frase += " "

cap.release()
cv2.destroyAllWindows()