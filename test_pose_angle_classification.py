import numpy as np
import math
import torch
import cv2
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F
import logging
import json
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Predefined "selfie" pose angles for comparison
selfie_angles = [
    {"connection": "RIGHT_FOOT_INDEX-RIGHT_KNEE-RIGHT_HIP", "angle": 177.3},
    {"connection": "RIGHT_SHOULDER-RIGHT_ELBOW-RIGHT_WRIST", "angle": 18.97},
    {"connection": "RIGHT_ELBOW-RIGHT_WRIST-RIGHT_THUMB", "angle": 168.38},
    {"connection": "LEFT_FOOT_INDEX-LEFT_KNEE-LEFT_HIP", "angle": 178.48},
    {"connection": "LEFT_SHOULDER-LEFT_ELBOW-LEFT_WRIST", "angle": 24.81},
    {"connection": "LEFT_ELBOW-LEFT_WRIST-LEFT_THUMB", "angle": 175.66},
]

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

device_config = config.get('device', 'auto')
device = torch.device('cuda' if torch.cuda.is_available() and device_config == 'auto' else device_config)
logging.info(f'Using device: {device}')

video_path = config['video_path']
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    logging.error(f"Failed to open video: {video_path}")
    exit(1)

frame_height, frame_width = cap.read()[1].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def classify_pose(landmarks, frame, angle_tolerance=30):
    matching_angles = 0
    # Calculate and compare angles
    for selfie_angle in selfie_angles:
        points = [landmarks[getattr(mp_pose.PoseLandmark, conn).value] for conn in selfie_angle["connection"].split('-')]
        current_angle = calculate_angle(points[0], points[1], points[2])
        if abs(selfie_angle["angle"] - current_angle) <= angle_tolerance: matching_angles += 1
    # Check if matching angles are at least 90% of selfie_angles
    if matching_angles / len(selfie_angles) >= 0.9:
        cv2.putText(frame, "Selfie Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Unknown Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            frame = classify_pose(landmarks, frame)
        cv2.imshow('Frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()