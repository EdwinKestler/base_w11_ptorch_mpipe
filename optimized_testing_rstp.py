import os
import numpy as np
import torch
import cv2
import torchvision.transforms.functional as F
import logging
import json
import mediapipe as mp
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)

# Initialize counters and configuration
selfie_pose_counter = 0
frame_count = 0
last_pose_was_selfie = False

# Configuration and logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
device_config = config.get('device', 'auto')
device = torch.device('cuda' if torch.cuda.is_available() and device_config == 'auto' else device_config)
logging.info(f'Using device: {device}')

# Predefined "selfie" pose angles for comparison
selfie_angles = [
    {"connection": "RIGHT_FOOT_INDEX-RIGHT_KNEE-RIGHT_HIP", "angle": 177.3},
    {"connection": "RIGHT_SHOULDER-RIGHT_ELBOW-RIGHT_WRIST", "angle": 18.97},
    {"connection": "RIGHT_ELBOW-RIGHT_WRIST-RIGHT_THUMB", "angle": 168.38},
    {"connection": "LEFT_FOOT_INDEX-LEFT_KNEE-LEFT_HIP", "angle": 178.48},
    {"connection": "LEFT_SHOULDER-LEFT_ELBOW-LEFT_WRIST", "angle": 24.81},
    {"connection": "LEFT_ELBOW-LEFT_WRIST-LEFT_THUMB", "angle": 175.66},
]

# MediaPipe and model initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
kweights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
mweights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=kweights).eval().to(device)
mmodel = maskrcnn_resnet50_fpn(weights=mweights).eval().to(device)
transform_fn = kweights.transforms()

# RTSP stream setup
RTSP_URL = 'rtsp://admin:FBx!admin2023@192.168.1.108:554'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    logging.error("Failed to open RTSP stream")
    exit(-1)

frame_height, frame_width = cap.read()[1].shape[:2]

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def classify_pose(landmarks, frame, angle_tolerance, counter, last_pose_was_selfie):
    matching_angles = 0
    # Calculate and compare angles
    for selfie_angle in selfie_angles:
        points = [landmarks[getattr(mp_pose.PoseLandmark, conn).value] for conn in selfie_angle["connection"].split('-')]
        current_angle = calculate_angle(points[0], points[1], points[2])
        if abs(selfie_angle["angle"] - current_angle) <= angle_tolerance: matching_angles += 1
    
    is_selfie_now = matching_angles / len(selfie_angles) >= 0.9
    if is_selfie_now and not last_pose_was_selfie:
        counter += 1  # Increment the counter only if transitioning to a "Selfie Pose"
        cv2.putText(frame, "Selfie Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif not is_selfie_now:
        cv2.putText(frame, "Unknown Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, counter, is_selfie_now

# Frame processing optimization
frame_skip = 5  # Process every 5th frame
frames_since_last_selfie = float('inf')  # Tracks frames since last "Selfie Pose"
# Before the try block, prepare the transformation function to avoid repeated calls to kweights.transforms()
transform_fn = kweights.transforms()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        resized_frame = cv2.resize(frame, (640, 480))  # Resizing for processing
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).to(device)

        with torch.no_grad():
            frame_transformed = transform_fn(frame_tensor).unsqueeze(0).to(device)
            outputs = model(frame_transformed)

            human_detected = False  # Reset detection flag for each frame

            # Process model outputs to check for human detection
            for output in outputs:
                # Assuming 'outputs' is a list of dictionaries for detected objects
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    if score > 0.99 and label == 1:  # Example threshold and label check
                        human_detected = True
                        break  # Break if at least one human is detected
                if human_detected:
                    break  # Break the outer loop as well if a human is detected

            if human_detected:
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                    _, selfie_pose_counter, last_pose_was_selfie = classify_pose(
                    landmarks, resized_frame, 30, selfie_pose_counter, last_pose_was_selfie)  # Adjusted to match function signature

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break  # Exit the loop if 'x' is pressed

except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()

    with open('selfie_pose_counter.json', 'w') as file:
        json.dump({'selfie_pose_counter': selfie_pose_counter}, file)
    logging.info(f"Selfie poses detected: {selfie_pose_counter}")