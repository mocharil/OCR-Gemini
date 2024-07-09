from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

from config import PROJECT_ID, CREDENTIALS_FILE_PATH, GEMINI_MODEL

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE_PATH)
vertexai.init(project=PROJECT_ID, credentials=credentials)
multimodal_model = GenerativeModel(GEMINI_MODEL)