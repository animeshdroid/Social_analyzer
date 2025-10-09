import re
from datetime import datetime
from typing import Optional, Tuple
import streamlit as st

def extract_social_url_info(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract platform and post ID from social media URL"""
    
    # Twitter/X patterns
    twitter_patterns = [
        r'(?:twitter|x)\.com/[^/]+/status/(\d+)',
        r't\.co/(\w+)',
        r'twitter\.com/i/web/status/(\d+)'
    ]
    
    for pattern in twitter_patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return 'twitter', match.group(1)
    
    # Reddit patterns
    reddit_patterns = [
        r'reddit\.com/r/\w+/comments/([a-z0-9]+)',
        r'redd\.it/([a-z0-9]+)',
        r'reddit\.com/comments/([a-z0-9]+)'
    ]
    
    for pattern in reddit_patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return 'reddit', match.group(1)
    
    return None, None

def clean_text(text: str) -> str:
    """Clean text for analysis"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def format_number(num: int) -> str:
    """Format number with K/M suffixes"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def get_time_ago(timestamp: int) -> str:
    """Convert timestamp to time ago string"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return "Unknown"

@st.cache_data
def load_models():
    """Load and cache HuggingFace models"""
    try:
        from app.models.sentiment_model import SentimentAnalyzer
        from app.models.emotion_detector import EmotionDetector
        
        sentiment_analyzer = SentimentAnalyzer()
        emotion_detector = EmotionDetector()
        
        return sentiment_analyzer, emotion_detector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

