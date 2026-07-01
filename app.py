import threading
from collections import Counter, deque

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(page_title="SeñaSalud", page_icon="🖐️", layout="wide")

# ── Parámetros (mismos que la versión de escritorio) ───────────────────────────
UMBRAL = 0.6                # confianza mínima por cuadro
BUFFER_SIZE = 12            # frames para votación por mayoría (suavizado temporal)
MIN_BUFFER = 4              # cuadros válidos mínimos antes de intentar votar
CONFIANZA_CONFIRMAR = 0.55  # % de acuerdo en el buffer para confirmar el gesto
HISTORIAL_MAX = 5           # gestos recientes mostrados
PANEL_FRASE_ALTURA = 50     # alto de la barra inferior con la frase

GESTO_ESPACIO = "espacio"
GESTO_BORRAR = "borrar"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("models/modelo_gestos.pkl")
    return modelo, modelo.classes_


modelo, clases = cargar_modelo()


def normalizar_puntos(landmarks):
    """Misma normalización que en la captura."""
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origen = puntos[0].copy()
    puntos -= origen
    escala = np.max(np.linalg.norm(puntos, axis=1))
    if escala > 0:
        puntos /= escala
    return puntos.flatten().tolist()


def dibujar_barra_confianza(frame, x, y, confianza, color):
    bw, bh = 200, 10
    filled = int(bw * confianza)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bh), color, -1)
    cv2.putText(frame, f"{confianza*100:.0f}%", (x + bw + 6, y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def dibujar_panel_gesto(frame, gesto, confianza, x_off=20, y_off=20):
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
    dibujar_barra_confianza(frame, x_off + 14, y_off + 58, confianza, color)


def dibujar_historial(frame, historial):
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
    h, w = frame.shape[:2]
    y0 = h - PANEL_FRASE_ALTURA
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.putText(frame, f"Frase: {frase}_", (16, y0 + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


class ProcesadorGestos(VideoProcessorBase):
    """Reconoce gestos de LSC cuadro a cuadro y arma la frase, igual que la versión de escritorio."""

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.buffers_prediccion = {}
        self.ultimo_gesto_por_mano = {}
        self.historial_gestos = deque(maxlen=HISTORIAL_MAX)
        self.frase = ""
        self.lock = threading.Lock()

    def reiniciar(self):
        with self.lock:
            self.frase = ""
            self.historial_gestos.clear()
            self.ultimo_gesto_por_mano.clear()
            self.buffers_prediccion.clear()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        with self.lock:
            if result.multi_hand_landmarks:
                for i, hand in enumerate(result.multi_hand_landmarks):
                    handedness = result.multi_handedness[i].classification[0].label
                    key = f"hand_{i}_{handedness}"

                    if key not in self.buffers_prediccion:
                        self.buffers_prediccion[key] = deque(maxlen=BUFFER_SIZE)

                    puntos_norm = normalizar_puntos(hand.landmark)
                    lado = [1, 0] if handedness == "Right" else [0, 1]
                    features = puntos_norm + lado

                    probs = modelo.predict_proba([features])[0]
                    idx = np.argmax(probs)
                    confianza_frame = probs[idx]

                    if confianza_frame > UMBRAL:
                        self.buffers_prediccion[key].append(clases[idx])

                    # Lectura cruda por cuadro (diagnóstico), junto a la muñeca
                    muneca = hand.landmark[0]
                    x_px = int(muneca.x * img.shape[1])
                    y_px = int(muneca.y * img.shape[0])
                    cv2.putText(img, f"{clases[idx]} {confianza_frame*100:.0f}%",
                                (x_px, y_px + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)

                    if len(self.buffers_prediccion[key]) >= MIN_BUFFER:
                        votos = Counter(self.buffers_prediccion[key])
                        gesto_final, votos_ganador = votos.most_common(1)[0]
                        confianza_suavizada = votos_ganador / len(self.buffers_prediccion[key])

                        if confianza_suavizada >= CONFIANZA_CONFIRMAR:
                            y_panel = 20 + i * 110
                            dibujar_panel_gesto(img, f"{handedness}: {gesto_final}",
                                                confianza_suavizada, y_off=y_panel)

                            # Confirmar el gesto solo si es nuevo para esta mano
                            if gesto_final != self.ultimo_gesto_por_mano.get(key):
                                self.ultimo_gesto_por_mano[key] = gesto_final

                                etiqueta = f"{handedness[0]}: {gesto_final}"
                                self.historial_gestos.append(etiqueta)

                                if gesto_final == GESTO_ESPACIO:
                                    self.frase += " "
                                elif gesto_final == GESTO_BORRAR:
                                    self.frase = self.frase[:-1]
                                else:
                                    self.frase += gesto_final

                    mp_drawing.draw_landmarks(
                        img, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(100, 180, 255), thickness=2),
                    )
            else:
                # Sin manos → limpiar buffers y permitir reconfirmar el mismo gesto
                self.buffers_prediccion.clear()
                self.ultimo_gesto_por_mano.clear()

            dibujar_historial(img, self.historial_gestos)
            dibujar_panel_frase(img, self.frase)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Interfaz ────────────────────────────────────────────────────────────────────
st.title("🖐️ SeñaSalud")
st.caption("Comunicación en Lengua de Señas Colombiana (LSC) para consultas médicas")

col_video, col_info = st.columns([3, 1])

with col_video:
    ctx = webrtc_streamer(
        key="senasalud",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ProcesadorGestos,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("Instrucciones")
    st.markdown(
        "1. Presiona **START** y permite el acceso a la cámara.\n"
        "2. Deletrea letra por letra frente a la cámara.\n"
        "3. Gesto **espacio** → agrega un espacio.\n"
        "4. Gesto **borrar** → elimina la última letra.\n"
        "5. Baja la mano un instante para repetir la misma letra dos veces seguidas.\n\n"
        "La frase construida se muestra en la barra inferior del video."
    )
    if st.button("🔄 Reiniciar frase"):
        if ctx.video_processor:
            ctx.video_processor.reiniciar()
