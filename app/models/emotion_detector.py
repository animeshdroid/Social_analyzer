from transformers import pipeline
import torch
from typing import Dict, Any
from app.config import Config
from app.utils.constants import EMOTION_EMOJIS

class EmotionDetector:
    def __init__(self):
        self.model_name = Config.EMOTION_MODEL
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_length = getattr(Config, "MAX_LENGTH", 128)
        try:
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                top_k=None,
                truncation=True,
                max_length=self.max_length
            )
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
            self.pipeline = None

    def detect_emotions(self, text: str) -> Dict[str, Any]:
        if not self.pipeline:
            return self._default_result("Pipeline missing")
        if not text.strip():
            return self._default_result("Blank input")
        try:
            results = self.pipeline(text)
            # Unwrap if results is a list of lists
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]
            if (isinstance(results, list) and len(results) > 0 
                and isinstance(results[0], dict) and 'label' in results[0] and 'score' in results[0]):
                sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)
                dominant = sorted_emotions[0]
                emotion_name = dominant['label'].lower()
                confidence = float(dominant['score'])
                emotion_scores = {e['label'].lower(): float(e['score']) for e in results if 'label' in e and 'score' in e}
                top_3 = [
                    {
                        'emotion': e['label'].lower(),
                        'score': float(e['score']),
                        'emoji': EMOTION_EMOJIS.get(e['label'].lower(), '😐')
                    }
                    for e in sorted_emotions[:3]
                ]
                return {
                    'dominant_emotion': emotion_name,
                    'confidence': confidence,
                    'emoji': EMOTION_EMOJIS.get(emotion_name, '😐'),
                    'all_emotions': emotion_scores,
                    'top_3_emotions': top_3
                }
            return self._default_result("No valid results")
        except Exception as e:
            print(f"❌ Error in emotion detection: {e}")
            return self._default_result(str(e))

    def _default_result(self, msg=""):
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'emoji': EMOTION_EMOJIS.get('neutral', '😐'),
            'all_emotions': {'neutral': 1.0},
            'top_3_emotions': [
                {'emotion': 'neutral', 'score': 1.0, 'emoji': EMOTION_EMOJIS.get('neutral', '😐')}
            ]
        }
