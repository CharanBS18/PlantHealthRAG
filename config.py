import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
try:
    load_dotenv()
except Exception:
    # Continue without .env file (for deployment environments)
    pass

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-base")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "Salesforce/blip-image-captioning-base")
HISTORY_PATH = os.getenv("HISTORY_PATH", "data/analysis_history.json")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "English")
ENABLE_TTS = os.getenv("ENABLE_TTS", "false").lower() == "true"
