import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(2)

PINCH_RESISTANCE = 0.025
MIDDLE_RESISTANCE = 0.025
BAGUE_RESISTANCE = 0.025
PETIT_THREDSHOLD = 0.025

event_cooldown = 2
last_event_time = 0
screenshot_count = 0

recording_enabled = False
mouth_prev_open = False

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def pouce_et_fuck():
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'shift', 'd') # shortcut pour mettre en sourdine sur discord, tu doit avoir l'app en grand au premier plan
    print("Mute casque")

def pouce_et_petit():
    print("petit doigt et pouce ")

def pouce_et_bague():
    print("pouce et bague  ")

def is_mouth_open(landmarks, h):
    upper_lip_y = landmarks[13].y * h
    lower_lip_y = landmarks[14].y * h
    return abs(lower_lip_y - upper_lip_y) > 15  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(image_rgb)
    mouth_open = False
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        if is_mouth_open(face_landmarks.landmark, h):
            mouth_open = True

    if mouth_open and not mouth_prev_open:
        recording_enabled = not recording_enabled
        print("System ON" if recording_enabled else "System OFF")
        time.sleep(0.5)  
    mouth_prev_open = mouth_open

    if recording_enabled:
        results = hands.process(image_rgb)

        current_time = time.time()
        if results.multi_hand_landmarks and (current_time - last_event_time > event_cooldown):
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                middle = hand_landmarks.landmark[12]
                bague = hand_landmarks.landmark[16]
                petit = hand_landmarks.landmark[20]

                thumb_coords = (thumb.x * w, thumb.y * h)
                index_coords = (index.x * w, index.y * h)
                middle_coords = (middle.x * w, middle.y * h)
                bague_coords = (bague.x * w, bague.y * h)
                petit_coords = (petit.x * w, petit.y * h)

                dist_thumb_index = get_distance(thumb_coords, index_coords)
                dist_thumb_middle = get_distance(thumb_coords, middle_coords)
                dist_thumb_bague = get_distance(thumb_coords, bague_coords)
                dist_thumb_petit = get_distance(thumb_coords, petit_coords)

                if dist_thumb_middle < MIDDLE_RESISTANCE * w:
                    pouce_et_fuck()
                    last_event_time = current_time
                    break

                if dist_thumb_petit < PETIT_THREDSHOLD * w:
                    pouce_et_petit()
                    last_event_time = current_time
                    break

                if dist_thumb_bague < BAGUE_RESISTANCE * w:
                    pouce_et_bague()
                    last_event_time = current_time
                    break

                if dist_thumb_index < PINCH_RESISTANCE * w:
                    screenshot_count += 1
                    filename = f"screen_{screenshot_count}.png"
                    cv2.imwrite(filename, frame)
                    print(f"screenshoted : {filename}")
                    last_event_time = current_time
                    break

    cv2.imshow('finger detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
