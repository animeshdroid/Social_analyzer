import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    
    # App Settings
    APP_NAME = os.getenv("APP_NAME", "Social Analyzer Pro")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8501))
    
    # Model Settings
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    MAX_LENGTH = 512
    
    # UI Settings
    PAGE_TITLE = "🌟 Social Analyzer Pro"
    PAGE_ICON = "🌟"

