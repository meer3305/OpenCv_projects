import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # For audio feedback (optional)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for rep counting
counter = 0
stage = None
shoulder_press_threshold = 0.15  # Threshold for shoulder press movement

# Initialize text-to-speech engine (optional)
engine = pyttsx3.init()

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_landmark_coordinates(landmarks, landmark_id, image_width, image_height):
    """Get normalized landmark coordinates and convert to pixel values"""
    landmark = landmarks.landmark[landmark_id]
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    return (x, y)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    results = pose.process(image)
    
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for shoulder, elbow, and wrist (right side)
        image_height, image_width, _ = image.shape
        shoulder = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_width, image_height)
        elbow = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, image_width, image_height)
        wrist = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, image_width, image_height)
        
        # Calculate angle between shoulder, elbow, and wrist
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Calculate vertical movement (shoulder to wrist)
        vertical_movement = wrist[1] - shoulder[1]
        
        # Shoulder press counter logic
        if vertical_movement < -image_height * shoulder_press_threshold:
            if stage == "down":
                counter += 1
                print(f"Rep count: {counter}")
                # Optional audio feedback
                if counter % 5 == 0:  # Announce every 5 reps
                    engine.say(f"{counter} reps")
                    engine.runAndWait()
            stage = "up"
        elif vertical_movement > -image_height * (shoulder_press_threshold * 0.5):
            stage = "down"
        
        # Visualize angle and movement
        cv2.putText(image, f"Angle: {angle:.2f}", 
                    tuple(np.add(elbow, [20, 40])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, f"Movement: {vertical_movement:.2f}", 
                    tuple(np.add(shoulder, [20, 80])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    except:
        pass
    
    # Render rep counter
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
    cv2.putText(image, 'REPS', (15, 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Render stage (up/down)
    cv2.putText(image, 'STAGE', (65, 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, stage, (60, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    
    cv2.imshow('Shoulder Press Counter', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()