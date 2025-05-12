import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for rep counting
left_counter = 0
right_counter = 0
left_stage = None
right_stage = None
curl_threshold = 160  # Angle threshold for curl detection
hysteresis = 30  # Prevents rapid toggling between states

# Initialize text-to-speech engine
engine = pyttsx3.init()

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle if angle <= 180 else 360-angle

def get_landmark_coordinates(landmarks, landmark_id, img_width, img_height):
    """Get normalized landmark coordinates and convert to pixel values"""
    landmark = landmarks.landmark[landmark_id]
    return (int(landmark.x * img_width), int(landmark.y * img_height))

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # Get image dimensions
        h, w, _ = image.shape
        
        # Right arm landmarks
        r_shoulder = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        r_elbow = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h)
        r_wrist = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h)
        
        # Left arm landmarks
        l_shoulder = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        l_elbow = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h)
        l_wrist = get_landmark_coordinates(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h)
        
        # Calculate angles
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        # Right arm logic
        if r_angle > curl_threshold:
            right_stage = "down"
        if r_angle < curl_threshold - hysteresis and right_stage == 'down':
            right_stage = "up"
            right_counter += 1
            print(f"Right arm reps: {right_counter}")
        
        # Left arm logic
        if l_angle > curl_threshold:
            left_stage = "down"
        if l_angle < curl_threshold - hysteresis and left_stage == 'down':
            left_stage = "up"
            left_counter += 1
            print(f"Left arm reps: {left_counter}")
        
        # Visual feedback for right arm
        cv2.putText(image, f"R: {r_angle:.1f}°", 
                   tuple(np.add(r_elbow, [20, 40])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Visual feedback for left arm
        cv2.putText(image, f"L: {l_angle:.1f}°", 
                   tuple(np.add(l_elbow, [20, 40])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    except Exception as e:
        print(f"Error: {e}")
        pass
    
    # Display counters
    cv2.rectangle(image, (0, 0), (w, 110), (245, 117, 16), -1)
    
    # Right arm counter
    cv2.putText(image, 'RIGHT REPS', (15, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(right_counter), (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, f"Stage: {right_stage}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Left arm counter
    cv2.putText(image, 'LEFT REPS', (w-150, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(left_counter), (w-150, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, f"Stage: {left_stage}", (w-150, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    
    cv2.imshow('Dual-Arm Bicep Curl Counter', image)
    
    # Audio feedback every 5 reps (optional)
    if right_counter % 5 == 0 or left_counter % 5 == 0:
        engine.say(f"Right {right_counter}, Left {left_counter}")
        engine.runAndWait()
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
