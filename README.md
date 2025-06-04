import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import datetime
import os

# Setup folders
os.makedirs("logs", exist_ok=True)
os.makedirs("dataset/fire", exist_ok=True)
os.makedirs("dataset/fight", exist_ok=True)
os.makedirs("dataset/normal", exist_ok=True)

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dummy models (Replace with trained classifiers for fire and fight)
fire_model = resnet18(pretrained=True)
fight_model = resnet18(pretrained=True)

# Modify for binary classification
fire_model.fc = torch.nn.Linear(fire_model.fc.in_features, 2)
fight_model.fc = torch.nn.Linear(fight_model.fc.in_features, 2)

# Set to evaluation
fire_model.eval()
fight_model.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Logging function
def log_event(event_type, frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"logs/{event_type}_{timestamp}.jpg"
    cv2.imwrite(path, frame)
    print(f"[LOGGED] {event_type} at {timestamp}")

# Prediction helper
def predict(model, frame):
    with torch.no_grad():
        img_tensor = transform(frame).unsqueeze(0)
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        return pred.item()

# Real-time loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        face = frame[y:y+h, x:x+w]

    # Fire Detection
    fire_pred = predict(fire_model, frame)
    if fire_pred == 1:
        log_event("fire", frame)

    # Fight Detection
    fight_pred = predict(fight_model, frame)
    if fight_pred == 1:
        log_event("fight", frame)

    # Display
    cv2.imshow("AI Security", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
