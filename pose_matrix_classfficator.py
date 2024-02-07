import cv2
import mediapipe as mp
import numpy as np
import math
import json

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load and process image
image_path = './selfie_pose_landmark.jpg'
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

pose_landmarks = {}

if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    landmarks = results.pose_landmarks.landmark
    pose_landmarks['landmarks'] = []
    pose_landmarks['angles'] = []

    # Example for calculating and storing multiple angles
    # Adjusted connections for calculating specific angles
    connections = [
        # Right side angles
        (mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB),
        
        # Left side angles
        (mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB),
    ]

    pose_landmarks['angles'] = []

    for connection in connections:
        a = [landmarks[connection[0].value].x * image_width, landmarks[connection[0].value].y * image_height]
        b = [landmarks[connection[1].value].x * image_width, landmarks[connection[1].value].y * image_height]
        c = [landmarks[connection[2].value].x * image_width, landmarks[connection[2].value].y * image_height]
        
        angle = calculate_angle(a, b, c)
        pose_landmarks['angles'].append({
            'connection': f'{connection[0].name}-{connection[1].name}-{connection[2].name}',
            'angle': angle
        })

        # Optionally, store landmark positions
        for landmark in mp_pose.PoseLandmark:
            pose_landmarks['landmarks'].append({
                'landmark': landmark.name,
                'x': landmarks[landmark.value].x,
                'y': landmarks[landmark.value].y,
                'z': landmarks[landmark.value].z,
                'visibility': landmarks[landmark.value].visibility,
            })

# Save to JSON
with open('selfie_pose_landmarks.json', 'w') as f:
    json.dump(pose_landmarks, f, indent=4)

# Display image
cv2.imshow("Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()