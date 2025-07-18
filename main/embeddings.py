import numpy as np
import sqlite3
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import cv2

# Load models
yolo_model = YOLO("yolov8n-face.pt")  # YOLOv8 lightweight face detector
facenet = InceptionResnetV1(pretrained="vggface2").eval()  # Face embedding model


def detect_face(image):
    """
    Detects and crops the first face found in the image using YOLOv8.
    """
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        return image[y1:y2, x1:x2]
    return None


def generate_embedding(image):
    """
    Generates a 512-d face embedding for a detected face using FaceNet.
    Returns None if no face is detected.
    """
    face = detect_face(image)
    if face is None:
        return None

    face = cv2.resize(face, (160, 160))
    face = np.transpose(face, (2, 0, 1)) / 255.0  # Normalize to [0, 1]
    face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        embedding = facenet(face_tensor).numpy().flatten()

    return embedding


def save_to_db(roll_no, name, avg_embedding):
    """
    Saves the student's roll number, name, and average face embedding into the database.
    """
    with sqlite3.connect("database/attendance_system.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (roll_no, name, embedding) VALUES (?, ?, ?)",
            (roll_no, name, avg_embedding.tobytes())
        )
        conn.commit()


def process_and_store(roll_no, name, images):
    """
    Processes a list of face images for a student, generates embeddings,
    averages them, and stores the result in the database.
    """
    embeddings_list = []
    for img in images:
        embedding = generate_embedding(img)
        if embedding is not None:
            embeddings_list.append(embedding)

    print(f"✅ Valid face images selected: {len(embeddings_list)}/{len(images)}")

    if len(embeddings_list) == 0:
        print("❌ Error: No valid faces detected. Registration failed.")
        return

    avg_embedding = np.mean(embeddings_list, axis=0)
    save_to_db(roll_no, name, avg_embedding)
    print(f"✅ Registration successful for {name} ({roll_no})")
