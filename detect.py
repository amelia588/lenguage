# Importamos las librerías
import cv2
import mediapipe as mp
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
import torch
import numpy as np
import os

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

# Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Cargamos el modelo YOLOv5 para reconocimiento adicional
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'C:/Users/SOPORTE/Desktop/VIDEO/modelo/best.pt'
model = DetectMultiBackend(model_path, device=device)

# Inicializamos la cámara
cap = cv2.VideoCapture(0)

# Bucle principal
while True:
    # Leer fotogramas
    ret, frame = cap.read()

    # Verificar si se capturó el fotograma correctamente
    if not ret:
        print("No se pudo capturar el fotograma.")
        break

    # Volteamos el fotograma horizontalmente para corregir el reflejo
    frame = cv2.flip(frame, 1)

    # Convertimos el fotograma a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección de manos con MediaPipe
    resultados = hands.process(rgb_frame)

    # Dibujamos las manos detectadas y obtenemos regiones de interés
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            # Dibujamos las conexiones de la mano
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Calculamos el bounding box aproximado de la mano
            h, w, _ = frame.shape
            x_min = min([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_min = min([int(landmark.y * h) for landmark in hand_landmarks.landmark])
            x_max = max([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_max = max([int(landmark.y * h) for landmark in hand_landmarks.landmark])

            # Aseguramos que el ROI esté dentro de los límites
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Extraemos el ROI de la mano
            roi = frame[y_min:y_max, x_min:x_max]

            # Procesamos el ROI con YOLO si es suficientemente grande
            if roi.size > 0:
                # Redimensionar el ROI para que coincida con el tamaño esperado por YOLOv5
                img = cv2.resize(roi, (640, 640))
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device).float() / 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)  # Reorganizamos el tensor para el modelo

                # Predicción con YOLO
                yolo_resultados = model(img, augment=False, visualize=False)
                yolo_resultados = non_max_suppression(yolo_resultados, conf_thres=0.4, iou_thres=0.45)

                # Dibujamos anotaciones de YOLO dentro del ROI
                for result in yolo_resultados[0]:
                    if result is not None:
                        x1, y1, x2, y2, conf, cls = result
                        cv2.rectangle(roi, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                frame[y_min:y_max, x_min:x_max] = roi

    # Mostramos los fotogramas con las detecciones
    cv2.imshow("DETECCION", frame)

    # Salir del programa con la tecla ESC (31)
    if cv2.waitKey(1) == 31:
        break

# Liberamos recursos
cap.release()
cv2.destroyAllWindows()