from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import Dict, Any
from app.config import Config
from app.utils.constants import SENTIMENT_EMOJIS

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = Config.SENTIMENT_MODEL
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                top_k=None,
                truncation=True,
                max_length=getattr(Config, "MAX_LENGTH", 128)
            )
        except Exception as e:
            print(f"❌ Error loading sentiment model: {e}")
            self.pipeline = None

        self.label_mapping = {
            'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE',
            'NEGATIVE': 'NEGATIVE', 'NEUTRAL': 'NEUTRAL', 'POSITIVE': 'POSITIVE'
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        if not self.pipeline:
            return self._default_result("Pipeline missing")
        if not text.strip():
            return self._default_result("Blank input")
        try:
            results = self.pipeline(text)
            # UNWRAP if nested [[...]]
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]
            if (isinstance(results, list) and len(results) > 0 
                and isinstance(results[0], dict) and 'label' in results[0] and 'score' in results[0]):
                sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
                top_result = sorted_results[0]
                sentiment_raw = top_result['label']
                sentiment = self.label_mapping.get(sentiment_raw, sentiment_raw)
                confidence = float(top_result['score'])
                score_map = {self.label_mapping.get(r['label'], r['label']): float(r['score']) for r in results if 'label' in r and 'score' in r}
                polarity = self._calculate_polarity(score_map)
                emoji = SENTIMENT_EMOJIS.get(sentiment, '😐')
                return {
                    'sentiment': sentiment,
                    'confidence': round(confidence, 3),
                    'emoji': emoji,
                    'all_scores': score_map,
                    'polarity': round(polarity, 3)
                }
            return self._default_result("No valid results")
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
            return self._default_result(str(e))

    def _calculate_polarity(self, scores: Dict[str, float]) -> float:
        pos_score = scores.get('POSITIVE', 0.0)
        neg_score = scores.get('NEGATIVE', 0.0)
        return pos_score - neg_score

    def _default_result(self, msg=""):
        return {
            'sentiment': 'NEUTRAL',
            'confidence': 0.0,
            'emoji': '😐',
            'all_scores': {},
            'polarity': 0.0
        }
