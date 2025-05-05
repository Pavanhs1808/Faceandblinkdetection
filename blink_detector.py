import cv2
import mediapipe as mp
import time

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Initialize OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# EAR threshold and frame counter
EAR_THRESHOLD = 0.25
CLOSED_EYES_FRAME = 3
blink_counter = 0
frame_counter = 0

# Timer for counting blinks every 30 seconds
start_time = time.time()
blink_count_in_30s = 0

# Emotion indicators
emotion = "Neutral"

# Eye landmark indices for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Capture video
cap = cv2.VideoCapture(0)

def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

def calculate_ear(landmarks, eye_indices):
    A = euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    ear = (A + B) / (2.0 * C)
    return ear

def update_emotion(blink_rate):
    if blink_rate < 0.09:
        return "Relaxed"
    elif 0.1 <= blink_rate <= 0.15:
        return "Neutral"
    else:
        return "Stressed"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw blue bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest (ROI) for face landmarks
        roi = frame[y:y + h, x:x + w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_roi)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Calculate EAR
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                # Blink detection logic
                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= CLOSED_EYES_FRAME:
                        blink_counter += 1
                        blink_count_in_30s += 1
                        print(f"Blink detected! Total blinks: {blink_counter}")
                    frame_counter = 0

    # Timer logic: Reset every 30 seconds and display the blink count in that period
    elapsed_time = time.time() - start_time
    if elapsed_time >= 30:
        start_time = time.time()  # Reset timer
        blink_rate = blink_count_in_30s / 30.0  # Calculate blink rate per second
        emotion = update_emotion(blink_rate)  # Update emotion based on blink rate
        blink_count_in_30s = 0  # Reset the blink count for the next 30 seconds
        print(f"Blink rate in last 30 seconds: {blink_rate:.2f} blinks/sec, Emotion: {emotion}")

    # Display blink count and emotion
    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Emotion: {emotion}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Eye Blink Detection with Face Box", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
