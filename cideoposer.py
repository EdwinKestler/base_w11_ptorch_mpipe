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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    
# Load predefined selfie pose landmarks
with open('selfie_pose_landmarks.json', 'r') as file:
    selfie_landmarks = json.load(file)

device_config = config.get('device', 'auto')
device = torch.device('cuda' if torch.cuda.is_available() and device_config == 'auto' else device_config)
logging.info(f'Using device: {device}')

mask_color = (0, 255, 0)  # Define mask color here

# Load models
with torch.no_grad():
    kweights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    mweights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=kweights).eval().to(device)
    mmodel = maskrcnn_resnet50_fpn(weights=mweights).eval().to(device)

video_path = config['video_path']  # Ensure your config.json contains the path to the video
cap = cv2.VideoCapture(video_path)

# Check if video was opened successfully
if not cap.isOpened():
    logging.error(f"Failed to open video: {video_path}")
    exit(1)

# Read the first frame to get video properties
ret, first_frame = cap.read()
if not ret:
    logging.error("Failed to read the first frame from the video.")
    cap.release()
    exit(1)

# Get the dimensions of the frame, which will be used for the output video
frame_height, frame_width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the codec and create VideoWriter object to save the output video (if needed)
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

def calculate_angle(a, b, c):
    """Calculate the angle between three points. Points are in (x, y) format."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_selfie_pose(current_angles, predefined_angles, angle_tolerance=10):
    """Check if the current pose matches the 'selfie' pose based on angles."""
    matching_angles = 0
    for pre_angle in predefined_angles:
        connection = pre_angle['connection']
        pre_angle_value = pre_angle['angle']
        # Find the corresponding current angle
        current_angle_value = next((angle['angle'] for angle in current_angles if angle['connection'] == connection), None)
        if current_angle_value is not None and abs(current_angle_value - pre_angle_value) <= angle_tolerance:
            matching_angles += 1
    # Determine if the pose matches based on a threshold
    if matching_angles / len(predefined_angles) >= 0.8:  # Adjust the threshold as needed
        return True
    return False

# Assuming you have a list of current pose angles calculated as follows:
current_angles = []  # This should be filled with current pose angles similar to predefined angles


def calculateAngle(landmark1, landmark2, landmark3):
    # Calculate angle between three points
    # Assuming landmark format: (x, y, z)
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).to(device)

        # Preprocess and model inference
        frame_transformed = kweights.transforms()(frame_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs, moutputs = model(frame_transformed), mmodel(frame_transformed)

        human_detected = False

        for output in outputs:
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                if score > 0.99 and label == 1:
                    human_detected = True
                    box = box.cpu().numpy().astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    label_text = f'Person: {score:.2f}'
                    cv2.putText(frame, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

        if human_detected:
            for moutput in moutputs:
                masks = moutput['masks']
                scores = moutput['scores']
                for mask, score in zip(masks, scores):
                    if score > config['score_threshold']:
                        mask = mask.squeeze().cpu().numpy()
                        mask = (mask > 0.5).astype(np.uint8)
                        frame[mask == 1] = cv2.addWeighted(frame[mask == 1], 0.5, np.full_like(frame[mask == 1], mask_color), 0.5, 0)
            
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = [np.array([lm.x, lm.y, lm.z]) for lm in results.pose_landmarks.landmark]
                
                # Example of integrating selfie detection
                if human_detected and results.pose_landmarks:
                    current_landmarks = [{  # Simplified example
                        'x': landmark.x * frame_width,
                        'y': landmark.y * frame_height,
                        'z': landmark.z
                    } for landmark in results.pose_landmarks.landmark]
                    
                    # Check if the current pose matches the "selfie" pose
                    if is_selfie_pose(current_landmarks, selfie_landmarks):
                        cv2.putText(frame, "Selfie", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('Frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")

cap.release()
out.release()
cv2.destroyAllWindows()