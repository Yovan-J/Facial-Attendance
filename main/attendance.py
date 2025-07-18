import cv2
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

# ------------------------ Configuration ------------------------
DEBUG_MODE = True               # Enable/disable debug logs
ABSENCE_TIMEOUT_MIN = 1         # Minutes to wait before marking student absent
DB_PATH = 'database/attendance_system.db'
FACE_MATCH_THRESHOLD = 0.6      # Cosine distance threshold for face matching

# ------------------------ Time Slots ------------------------
SLOTS = [
    (1, "08:50", "09:40"), (2, "09:40", "10:30"), (3, "10:30", "10:45"),
    (4, "10:45", "11:35"), (5, "11:35", "12:25"), (6, "12:25", "13:15"),
    (7, "13:15", "14:05"), (8, "14:05", "14:55"), (9, "14:55", "15:45"),
    (10, "15:45", "16:35"), (11, "02:47", "02:48"), (12, "02:48", "02:49")
]

# ------------------------ Load Models ------------------------
facenet = InceptionResnetV1(pretrained='vggface2').eval()
yolo_model = YOLO("yolov8n-face.pt")


# ------------------------ Utility Functions ------------------------

def get_current_slot():
    """Returns the current active slot ID based on system time."""
    now = datetime.now().strftime('%H:%M')
    for slot_id, start, end in SLOTS:
        if start <= now < end:
            return slot_id
    return 0


def detect_face(image):
    """Detects and crops the face using YOLOv8."""
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        return image[y1:y2, x1:x2]
    return None


def generate_embedding(image):
    """Generates 512-D face embedding using FaceNet."""
    face = detect_face(image)
    if face is None:
        return None
    face = cv2.resize(face, (160, 160))
    face = np.transpose(face, (2, 0, 1)) / 255.0
    face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        embedding = facenet(face_tensor).numpy().flatten()
    return embedding


def match_face(captured_embedding):
    """Matches captured embedding with database records."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT roll_no, embedding FROM students')
    for roll_no, db_embedding in cursor.fetchall():
        db_embedding = np.frombuffer(db_embedding, dtype=np.float32)
        if cosine(captured_embedding, db_embedding) < FACE_MATCH_THRESHOLD:
            conn.close()
            if DEBUG_MODE:
                print(f"[MATCH] Roll No: {roll_no}")
            return roll_no
    conn.close()
    return None


def track_attendance(roll_no, slot_id):
    """Marks student 'Present' if not already recorded for the slot."""
    date = datetime.now().strftime('%Y-%m-%d')
    entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        '''SELECT 1 FROM attendance WHERE roll_no = ? AND date = ? AND slot_id = ?''',
        (roll_no, date, slot_id)
    )

    if cursor.fetchone() is None:
        cursor.execute('''
            INSERT INTO attendance (roll_no, date, slot_id, entry_time, exit_time, status)
            VALUES (?, ?, ?, ?, NULL, "Present")
        ''', (roll_no, date, slot_id, entry_time))
        if DEBUG_MODE:
            print(f"[ENTRY] Roll No: {roll_no}, Slot: {slot_id}, Time: {entry_time}")
    else:
        if DEBUG_MODE:
            print(f"[SKIP] Already marked present for Slot {slot_id}: {roll_no}")

    conn.commit()
    conn.close()


def update_status(roll_no, slot_id):
    """Marks exit time when student leaves."""
    exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT entry_time FROM attendance 
        WHERE roll_no = ? AND slot_id = ? AND exit_time IS NULL
        ORDER BY entry_time DESC LIMIT 1
    ''', (roll_no, slot_id))

    if cursor.fetchone():
        cursor.execute('''
            UPDATE attendance SET exit_time = ? 
            WHERE roll_no = ? AND slot_id = ? AND exit_time IS NULL
        ''', (exit_time, roll_no, slot_id))
        if DEBUG_MODE:
            print(f"[EXIT] Roll No: {roll_no}, Slot: {slot_id}, Time: {exit_time}")

    conn.commit()
    conn.close()


# ------------------------ Main Recognition Loop ------------------------

def recognize_face_from_webcam():
    """Main loop for real-time webcam recognition and attendance marking."""
    cap = cv2.VideoCapture(0)
    student_in_class = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_slot = get_current_slot()
        if current_slot == 0:
            continue  # No active slot, skip frame

        embedding = generate_embedding(frame)
        if embedding is not None:
            roll_no = match_face(embedding)
            if roll_no:
                if roll_no not in student_in_class:
                    student_in_class[roll_no] = {
                        "entry_time": datetime.now(),
                        "last_seen": datetime.now(),
                        "slot_id": current_slot,
                        "remaining_time": ABSENCE_TIMEOUT_MIN * 60
                    }
                    track_attendance(roll_no, current_slot)
                else:
                    student_in_class[roll_no]["last_seen"] = datetime.now()

                    # If slot changed while student is still present
                    if student_in_class[roll_no]["slot_id"] != current_slot:
                        track_attendance(roll_no, current_slot)
                        student_in_class[roll_no]["slot_id"] = current_slot

        # Handle exit timeout logic
        for roll_no, data in list(student_in_class.items()):
            time_since_last_seen = datetime.now() - data["last_seen"]
            remaining_time = ABSENCE_TIMEOUT_MIN * 60 - time_since_last_seen.total_seconds()

            if remaining_time <= 0:
                update_status(roll_no, current_slot)
                del student_in_class[roll_no]
            else:
                student_in_class[roll_no]["remaining_time"] = remaining_time
                if DEBUG_MODE:
                    print(f"[TIMEOUT] {roll_no}: {remaining_time/60:.2f} min remaining")

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------ Entry Point ------------------------

if __name__ == "__main__":
    recognize_face_from_webcam()
