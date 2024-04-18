import cv2
import mediapipe as mp
import numpy as np
import ffmpeg

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load video
video_path = './TheFigen_vid1.mp4'
probe = ffmpeg.probe(video_path)
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
frame_size = (int(video_stream['width']), int(video_stream['height']))
fps = eval(video_stream['avg_frame_rate'])

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'avc1')
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

total_triangles = 0  # Initialize total squares count

# Process video and detect poses
stream = ffmpeg.input(video_path)
stream = ffmpeg.filter(stream, 'fps', fps=fps)
stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='bgr24')
process = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=False)

# Initialize the tracker
tracker = cv2.legacy.TrackerCSRT_create()
tracking_started = False
tracking_box = None

while True:
    in_bytes = process.stdout.read(frame_size[0] * frame_size[1] * 3)
    if not in_bytes:
        break
    
    frame = np.frombuffer(in_bytes, np.uint8).reshape(frame_size[1], frame_size[0], 3)

    # Convert the frame to RGB format as expected by MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply MediaPipe Pose on the frame
    results = mp_pose.process(frame_rgb)
    
    # Check if any landmarks are detected
    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get the RIGHT_INDEX landmark
        right_index = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
        
        # Convert normalized coordinates to pixel values
        x = int(right_index.x * frame_size[0])
        y = int(right_index.y * frame_size[1])
        
        # Define the size of the bounding box (this can be adjusted as needed)
        box_size = 50
        
        # Crop the region around the RIGHT_INDEX
        cropped_region = frame[max(y - box_size, 0):min(y + box_size, frame_size[1]),
                               max(x - box_size, 0):min(x + box_size, frame_size[0])]
        
        # Create a copy of the cropped region
        cropped_region_copy = cropped_region.copy()
        
        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and aspect ratio to identify squares
        triangles = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.225 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check for triangles based on the number of vertices (3)
            if len(approx) == 3:
                triangles.append(contour)
        
        # Create a copy of the frame before drawing the landmarks
        frame_copy = frame.copy()
        
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            frame_copy, 
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Check if segmentation mask is available
        if results.segmentation_mask is not None:
            # Create a red mask image
            segmentation_mask = (results.segmentation_mask * 255).astype(np.uint8)
            red_mask = cv2.merge([segmentation_mask]*3)  # Create a 3-channel red mask
            red_mask[:, :, 1:] = 0  # Set green and blue channels to 0 (only red channel active)

            # Blend the red mask with the original frame
            alpha = 0.9  # Transparency factor for the red mask
            frame_copy = cv2.addWeighted(frame_copy, 1, red_mask, alpha, 0)

        # Draw contours on the cropped region copy
        cv2.drawContours(cropped_region_copy, triangles, -1, (0, 0, 255), 2)
        
        # Count and print the number of detected squares
        num_triangles = len(triangles)
        total_triangles += num_triangles  # Update the total count
        
        # Overlay the count on the frame
        cv2.putText(frame_copy, f'cajas corregidas: {num_triangles}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_copy, f'Trabajo efectivo 420: {total_triangles}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Place the processed cropped region back into the frame
        frame_copy[max(y - box_size, 0):min(y + box_size, frame_size[1]),
                    max(x - box_size, 0):min(x + box_size, frame_size[0])] = cropped_region_copy

        # Update the tracker
        if not tracking_started:
            # Initialize the tracker
            tracker.init(frame_copy, (x - box_size, y - box_size, box_size * 2, box_size * 2))
            tracking_started = True
            tracking_box = (x - box_size, y - box_size, box_size * 2, box_size * 2)
        else:
            # Update the tracker
            success, tracking_box = tracker.update(frame_copy)
            if success:
                # Draw the tracking box
                x, y, w, h = [int(v) for v in tracking_box]
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Write the frame to the output video
    out.write(frame_copy)

# Cleanup
process.stdout.close()
process.wait()
out.release()