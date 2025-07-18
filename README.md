# Facial Recognition-Based Attendance System

This project implements a facial recognition-based attendance system using Python, OpenCV, PyTorch, and a YOLOv8 face detection model. The system captures face data, extracts embeddings, and maintains attendance records securely in a SQLite database.

## Features

- Admin-only access to registration interface  
- Face detection using YOLOv8  
- Facial embeddings generation using a deep learning model  
- Image capture with preview and face verification  
- SQLite database integration for student data and attendance logs  
- Attendance marking via facial recognition and cosine similarity  
- GUI built using PyQt6  

## Project Structure

```
AttendanceSystem/
│
├── admin_login.py           # Entry point for admin to register students
├── attendance.py            # Facial recognition and attendance marking
├── db_init.py               # Initializes the SQLite database schema
├── embeddings.py            # Embedding extraction using deep learning model
├── register_gui.py          # GUI for capturing and saving face data
├── database/
│   └── attendance_system.db # SQLite database (created at runtime)
├── models/
│   └── yolov8n-face.pt      # Pre-trained YOLOv8 face detection model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Initialize Database

```bash
python db_init.py
```

### 2. Register Students

```bash
python admin_login.py
```

> Only admin can access the registration window. It captures 5 face images per student, generates an average embedding, and stores it along with name and roll number.

### 3. Mark Attendance

```bash
python attendance.py
```

> The system opens the camera, detects faces, compares with stored embeddings, and logs attendance if similarity exceeds the threshold.

## Notes

- Ensure `yolov8n-face.pt` is present in the `models/` directory.  
- Captured face images are not permanently stored; only embeddings are retained.  
- Attendance entries are timestamped and stored in the database.

## License

This project is intended for educational use. Modify and extend as needed.
