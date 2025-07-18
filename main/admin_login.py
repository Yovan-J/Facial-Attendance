import sys
import os
import subprocess
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox
)
from PyQt6.QtGui import QFont

# Hardcoded admin credentials (could later be replaced with hashed DB or config)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

# Token file to grant temporary admin access
TOKEN_FILE = "admin_access.token"

class AdminLogin(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Login")
        self.setGeometry(500, 300, 400, 250)
        self.setStyleSheet("background-color: #2c3e50; color: white;")

        layout = QVBoxLayout()

        self.title = QLabel("🔒 Admin Login")
        self.title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.title.setStyleSheet("color: #f39c12;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        # Username Input
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter Username")
        self.username_input.setFont(QFont("Arial", 12))
        self.username_input.setStyleSheet(self.input_style())
        layout.addWidget(self.username_input)

        # Password Input
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFont(QFont("Arial", 12))
        self.password_input.setStyleSheet(self.input_style())
        layout.addWidget(self.password_input)

        # Login Button
        self.login_button = QPushButton("Login")
        self.login_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.login_button.setStyleSheet(self.button_style())
        self.login_button.clicked.connect(self.check_credentials)
        layout.addWidget(self.login_button)

        # Trigger login on pressing Enter in password field
        self.password_input.returnPressed.connect(self.check_credentials)

        self.setLayout(layout)

    def input_style(self):
        return """
            QLineEdit {
                border: 2px solid #f39c12;
                border-radius: 10px;
                padding: 8px;
                background-color: #ecf0f1;
                color: black;
            }
        """

    def button_style(self):
        return """
            QPushButton {
                background-color: #f39c12;
                color: white;
                border-radius: 10px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            QMessageBox.information(self, "Login Successful", "Welcome, Admin!")
            self.generate_access_token()
            self.open_face_registration()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password!")

    def generate_access_token(self):
        """Creates a temporary token file to verify admin access"""
        with open(TOKEN_FILE, "w") as f:
            f.write("AUTHORIZED")

    def open_face_registration(self):
        """Launches the registration interface"""
        subprocess.Popen([sys.executable, "register_gui.py", "admin_access"])
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_window = AdminLogin()
    login_window.show()
    sys.exit(app.exec())
