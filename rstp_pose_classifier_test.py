import os
from screeninfo import get_monitors
import numpy as np
import math
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

selfie_pose_counter = 0  # Counts unique "Selfie Pose" detection events
frame_count = 0  # Counts all processed frames
last_pose_was_selfie = False  # Tracks if the last pose detected was a "Selfie Pose"

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

# Model loading optimization
# Load models only once when the script starts
global kweights, mweights, model, mmodel  # Declare globally for model persistence
try:
    kweights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    mweights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=kweights).eval().to(device)
    mmodel = maskrcnn_resnet50_fpn(weights=mweights).eval().to(device)
except:  # In case of first run exception, load models explicitly
    with torch.no_grad():
        kweights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        mweights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=kweights).eval().to(device)
        mmodel = maskrcnn_resnet50_fpn(weights=mweights).eval().to(device)
        
# Use the RTSP feed instead of a video file
RTSP_URL = 'rtsp://admin:FBx!admin2023@192.168.1.108:554'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    logging.error("Failed to open RTSP stream")
    exit(-1)

# Get screen size from the first monitor
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

# Calculate desired window size as 1/4 of the screen size
desired_width = screen_width // 3
desired_height = screen_height // 3

frame_height, frame_width = cap.read()[1].shape[:2]

# Define the codec and create VideoWriter object, trying a different codec
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Change to 'avc1' for H.264, if 'mp4v' is not working
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (desired_width, desired_height))


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
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames based on the frame_skip value

        # Resize frame to desired dimensions only once before any processing
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert the resized frame to RGB only once for both MediaPipe and model inference
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Convert the RGB frame to a tensor only once for model inference
        frame_tensor = F.to_tensor(frame_rgb).to(device)
        
        # Now, the frame_rgb is used for MediaPipe processing and frame_tensor for PyTorch model inference
        with torch.no_grad():
            # Apply the transformation directly to the frame tensor for model inference
            frame_transformed = transform_fn(frame_tensor).unsqueeze(0).to(device)
            outputs, moutputs = model(frame_transformed), mmodel(frame_transformed)

            human_detected = False
        
            for output in outputs:
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    if score > 0.99 and label == 1:
                        human_detected = True
                        box = box.cpu().numpy().astype(int)
                        cv2.rectangle(resized_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        label_text = f'Person: {score:.2f}'
                        cv2.putText(resized_frame, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break  # Exit the loop after the first detected person
            
        # Continue using frame_rgb for MediaPipe processing without reconversion
        if human_detected:
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                resized_frame, selfie_pose_counter, last_pose_was_selfie = classify_pose(
                landmarks, resized_frame, 30, selfie_pose_counter, last_pose_was_selfie)

        # The rest of the loop continues without re-converting the frame
        cv2.putText(resized_frame, f'Selfie Poses: {selfie_pose_counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('RTSP stream', resized_frame)
        out.write(resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('RTSP stream', cv2.WND_PROP_VISIBLE) < 1:
            break

except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
    
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()