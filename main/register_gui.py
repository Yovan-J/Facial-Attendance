import sys
import cv2
import numpy as np
import sqlite3
import torch
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QVBoxLayout, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import QSize, QTimer
import embeddings  # Ensure embeddings.py exists and is in same folder

# Restrict script execution to admin_login.py
if len(sys.argv) < 2 or sys.argv[1] != "admin_access":
    print("Unauthorized access! This script must be run from admin_login.py.")
    sys.exit(1)

# Load YOLOv8 face detector
model = YOLO("yolov8n-face.pt")  # Ensure the file exists in working directory

class RegisterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Face Registration")
        self.setGeometry(400, 200, 600, 500)

        # --- GUI Layout ---
        layout = QVBoxLayout()

        self.roll_label = QLabel("📌 Roll No:")
        self.roll_input = QLineEdit()
        self.roll_input.setPlaceholderText("Enter Roll Number")
        layout.addWidget(self.roll_label)
        layout.addWidget(self.roll_input)

        self.name_label = QLabel("📌 Name:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        self.start_camera_btn = QPushButton("📹 Start Capturing")
        self.start_camera_btn.clicked.connect(self.start_camera)
        layout.addWidget(self.start_camera_btn)

        self.capture_btn = QPushButton("📸 Capture Image")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_btn)

        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(150, 150))
        self.image_list.setFixedHeight(320)
        self.image_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.image_list)

        self.save_btn = QPushButton("✅ Save Selected Images")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_embeddings)
        layout.addWidget(self.save_btn)

        self.delete_btn = QPushButton("🗑 Clear Captured Images")
        self.delete_btn.clicked.connect(self.clear_images)
        layout.addWidget(self.delete_btn)

        self.setLayout(layout)

        # --- Variables ---
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_image)
        self.captured_images = []
        self.max_images = 6

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Failed to open camera.")
            return

        self.capture_btn.setEnabled(True)
        self.timer.start(3000)  # Capture every 3 seconds

    def capture_image(self):
        if len(self.captured_images) >= self.max_images:
            QMessageBox.warning(self, "Limit Reached", f"Maximum of {self.max_images} images allowed.")
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Capture Error", "Could not read from camera.")
            return

        results = model(frame)
        if len(results[0].boxes) == 0:
            QMessageBox.warning(self, "No Face", "No face detected. Ensure your face is visible.")
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.captured_images.append(rgb_image)
        item = QListWidgetItem(QIcon(pixmap), "")
        item.setSizeHint(QSize(150, 150))
        self.image_list.addItem(item)

        if len(self.captured_images) >= 3:
            self.save_btn.setEnabled(True)

    def save_embeddings(self):
        selected_items = self.image_list.selectedItems()
        if len(selected_items) != 5:
            QMessageBox.warning(self, "Selection Error", "Please select exactly 5 images.")
            return

        selected_images = [self.captured_images[self.image_list.row(item)] for item in selected_items]
        valid_faces = sum(1 for img in selected_images if len(model(img)[0].boxes) > 0)

        if valid_faces < 5:
            QMessageBox.warning(self, "Invalid Images", "Ensure all selected images contain clear faces.")
            return

        embeddings_list = []
        for img in selected_images:
            emb = embeddings.generate_embedding(img)
            if emb is not None:
                embeddings_list.append(emb)

        if len(embeddings_list) < 5:
            QMessageBox.warning(self, "Embedding Error", "Failed to extract all embeddings.")
            return

        avg_embedding = np.mean(embeddings_list, axis=0)
        roll_no = self.roll_input.text().strip()
        name = self.name_input.text().strip()

        if not roll_no or not name:
            QMessageBox.warning(self, "Missing Data", "Roll No and Name are required.")
            return

        self.store_embedding(roll_no, name, avg_embedding)
        QMessageBox.information(self, "Success", "Embeddings saved successfully!")
        self.clear_images()

    def store_embedding(self, roll_no, name, embedding):
        try:
            conn = sqlite3.connect("database/attendance_system.db")
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO students (roll_no, name, embedding) VALUES (?, ?, ?)''',
                (roll_no, name, embedding.tobytes())
            )
            conn.commit()
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to save: {e}")
        finally:
            conn.close()

    def clear_images(self):
        self.image_list.clear()
        self.captured_images.clear()
        self.save_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegisterGUI()
    window.show()
    sys.exit(app.exec())
