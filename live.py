#Import libraries
import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
import pygame
import time
import winsound
import numpy as np
import mediapipe as mp

#Initialize sound
pygame.mixer.init()

#Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

#Eye and mouth landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_LIPS = [61, 291, 81, 178, 13, 14, 17, 84, 311]

#EAR and MAR functions
def compute_ear(landmarks, eye):
    A = np.linalg.norm(np.array(landmarks[eye[1]]) - np.array(landmarks[eye[5]]))
    B = np.linalg.norm(np.array(landmarks[eye[2]]) - np.array(landmarks[eye[4]]))
    C = np.linalg.norm(np.array(landmarks[eye[0]]) - np.array(landmarks[eye[3]]))
    return (A + B) / (2.0 * C)

def compute_mar(landmarks):
    def to_xy(p): return np.array([p.x, p.y])
    A = np.linalg.norm(to_xy(landmarks[13]) - to_xy(landmarks[14]))
    B = np.linalg.norm(to_xy(landmarks[61]) - to_xy(landmarks[291]))
    return A / B

#Load and modify ResNet18 model for 3 classes
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("StayAwake.pth", map_location=torch.device('cpu')))
model.eval()

#Image transformations used during training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Class labels
labels = ["awake", "drowsy", "sleepy"]

#Open the webcam
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')


blink_start, yawn_start = None, None
sound_played = False


cv2.namedWindow("Stay Awake", cv2.WINDOW_NORMAL)

while True:
    state = "Awake"
    color = (0, 255, 0)

    ret, frame = cap.read()
    if not ret:
        break

    #Convert image to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img).unsqueeze(0)

    #Predict the state using the smart model
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        state = labels[predicted.item()]  # تحويل الرقم إلى اسم الحالة من خلال قائمة أو قاموس labels

        if state == 'awake':  # تحقق من اسم الحالة وليس من الرقم
            color = (0, 255, 0)  # أخضر في حالة الاستيقاظ
        elif state == 'drowsy':
            color = (0,255,255)
        else:
            color = (0, 0, 255)  # على سبيل المثال: أزرق أو لون آخر في حالة النعاس



    #Calculate EAR and MAR
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

            left_ear = compute_ear(landmarks, LEFT_EYE)
            right_ear = compute_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            mar = compute_mar(face_landmarks.landmark)

            #Draw landmarks
            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[i], 2, (0, 0, 255), -1)
            for i in OUTER_LIPS:
                cv2.circle(frame, (int(face_landmarks.landmark[i].x * w),
                                   int(face_landmarks.landmark[i].y * h)), 2, (255, 0, 0), -1)

            #Eye-based drowsiness logic
            if ear < 0.2:
                if blink_start is None:
                    blink_start = time.time()
                elif time.time() - blink_start > 2:
                    state = "sleepy"
                    color = (0, 0, 255)
                    #Red screen overlay
                    overlay = frame.copy()
                    red_overlay = np.full_like(frame, (0, 0, 255))
                    alpha = 0.4
                    cv2.addWeighted(red_overlay, alpha, overlay, 1 - alpha, 0, overlay)
                    frame = overlay
                    #Play alert sound
                    if not sound_played:
                        pygame.mixer.music.load("alert.mp3")
                        pygame.mixer.music.play()
                        sound_played = True
                  

            else:
                blink_start = None
                sound_played = False
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                state = "awake"
                color = (0, 255, 0)

            if mar > 0.2:
                if yawn_start is None:
                    yawn_start = time.time()
                elif time.time() - yawn_start > 1.5:
                    state = "drowsy"
                    color = (0, 255, 255)
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    sound_played = False
                    if not pygame.mixer.music.get_busy():
                        winsound.Beep(1000, 500)
            else:
                yawn_start = None

            #Display EAR, MAR, and State
            cv2.putText(frame, f'EAR: {ear:.2f}', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(frame, f'State: {state}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Get window size
    _, _, win_w, win_h = cv2.getWindowImageRect("Stay Awake")

    # Resize frame to fit window size
    resized_frame = cv2.resize(frame, (win_w, win_h))

    # Show the resized frame
    cv2.imshow("Stay Awake", resized_frame)

    #Show the frame
   # cv2.imshow("Stay Awake", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Stay Awake", cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
