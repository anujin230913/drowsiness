import cv2              # Камер, зураг боловсруулах
import time             # Секунд тоолох
import numpy as np      # Тооцоолол
import mediapipe as mp  # Нүүр, нүд, ам таних AI
import winsound         # Дуут дохио (Windows)

# -------- MediaPipe тохиргоо --------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# -------- Камер асаах --------
cap = cv2.VideoCapture(0)

# -------- Face landmark индексүүд --------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]

# -------- Туслах функцууд --------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye):
    v1 = distance(landmarks[eye[1]], landmarks[eye[5]])
    v2 = distance(landmarks[eye[2]], landmarks[eye[4]])
    h = distance(landmarks[eye[0]], landmarks[eye[3]])
    return (v1 + v2) / (2.0 * h)

def play_alert():
    # winsound.SND_LOOP ашиглан дууг тасралтгүй тоглуулна
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP)

def stop_alert():
    winsound.PlaySound(None, winsound.SND_PURGE)

# -------- ХЯНАЛТЫН ХУВЬСАГЧИД --------
eye_close_start = None
yawn_count = 0
last_yawn_time = time.time()
alert_playing = False 

# ================== ҮНДСЭН ДАВТАЛТ ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    danger = False

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # ---- НҮД ШАЛГАХ ----
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            # ---- АМ ШАЛГАХ ----
            mouth_open = distance(landmarks[MOUTH[0]], landmarks[MOUTH[1]])

            current_time = time.time()

            # 👁 НҮДНИЙ ЛОГИК (0.2-оос бага бол аниастай)
            if ear < 0.20:
                if eye_close_start is None:
                    eye_close_start = current_time
                elif current_time - eye_close_start >= 4: # 4 секундээс дээш
                    danger = True
            else:
                eye_close_start = None

            # 👄 ЭВШЭЭХ ЛОГИК
            if mouth_open > 30: # Амны зай
                if current_time - last_yawn_time > 2:
                    yawn_count += 1
                    last_yawn_time = current_time

            if yawn_count >= 4:
                danger = True

            # 🚨 АНХААРУУЛГА ХАРУУЛАХ
            if danger:
                cv2.putText(frame, "ANHAAR! TA YDARSAN BNA!", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                if not alert_playing:
                    play_alert()
                    alert_playing = True

            # ---- ДЭЛГЭЦ ДЭЭР МЭДЭЭЛЭЛ ХАРУУЛАХ ----
            cv2.putText(frame, f"EAR (Nud): {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Evshilt: {yawn_count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Stop Alert: 's' | Exit: 'q'", (30, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Товчлуур шалгах
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):      # 's' дарвал дуу зогсоно
        stop_alert()
        alert_playing = False
        yawn_count = 0       # Тоолуурыг дахин эхлүүлэх
    elif key == ord('q'):    # 'q' дарвал гарна
        break

    cv2.imshow("Joloochiin Ayulgui Baidal", frame)

cap.release()
cv2.destroyAllWindows()
stop_alert()