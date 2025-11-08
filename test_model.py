#Import libraries
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import mediapipe as mp

#Setup mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

#Eye and mouth landmark indices
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

#Load image
image_path = "img2.jpg"
image = cv2.imread(image_path)
h, w = image.shape[:2]
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Extract face landmarks
result = face_mesh.process(rgb)

#Define EAR and MAR functions
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

#Load the model StayAwake.pth
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("StayAwake.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

labels = ["awake", "drowsy", "sleepy"]

#Predict state
input_tensor = transform(rgb).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    state = labels[predicted.item()]

#Calculate EAR and MAR
if result.multi_face_landmarks:
    face_landmarks = result.multi_face_landmarks[0]
    pixel_landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

    left_ear = compute_ear(pixel_landmarks, LEFT_EYE)
    right_ear = compute_ear(pixel_landmarks, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    mar = compute_mar(face_landmarks.landmark)

    #Draw eye landmarks
    for idx in LEFT_EYE + RIGHT_EYE:
        x, y = pixel_landmarks[idx]
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  
    #Draw mouth landmarks
    mouth_points = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146,
        91, 181, 84, 17, 314, 405, 321, 375, 291, 308
    ]
    for idx in mouth_points:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1) 
    #Annotate image with results
    cv2.putText(image, f"State: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, f"EAR: {ear:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, f"MAR: {mar:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #Save image
    output_path = "output_with_ear_mar.jpg"
    cv2.imwrite(output_path, image)
    print(f"Save img in : {output_path}")
    print(f"Predicted State: {state}")
    print(f"EAR: {ear:.2f}")
    print(f"MAR: {mar:.2f}")
else:
    print("No image !!")
