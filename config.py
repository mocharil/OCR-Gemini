import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
CREDENTIALS_FILE_PATH = os.getenv("CREDENTIALS_FILE_PATH")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
