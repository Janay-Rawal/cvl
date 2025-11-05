# Experiment 1: Human Pose Estimation using YOLO + MediaPipe

import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load models
model = YOLO('yolov8n.pt')                # YOLO for person detection
pose = mp.solutions.pose.Pose()           # MediaPipe for pose
draw = mp.solutions.drawing_utils         # Drawing utility

cap = cv2.VideoCapture('Walk_2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])   # Detect people
    for box in results[0].boxes.xyxy:     # For each person
        x1, y1, x2, y2 = map(int, box)
        person = frame[y1:y2, x1:x2]       # Crop
        rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
        output = pose.process(rgb)

        if output.pose_landmarks:
            draw.draw_landmarks(
                person, output.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
            frame[y1:y2, x1:x2] = person

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()