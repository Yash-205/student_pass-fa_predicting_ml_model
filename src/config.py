"""
Configuration module for the Student Study Coach.
Defines project-wide constants, paths, and feature names.
"""

from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the raw student dataset
DATA_PATH = BASE_DIR / "data" / "student_data.csv"

# Directory where trained model artifacts are stored
MODELS_DIR = BASE_DIR / "models"

# List of numeric features used for both training and inference.
# Any changes here require retraining the model.
FEATURES = [
    "parental_education_level",
    "daily_study_hours",
    "attendance_rate",
    "sleep_hours",
    "stress_level",
    "motivation_score",
    "math_score",
    "reading_score",
    "writing_score",
]

