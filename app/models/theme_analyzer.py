from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from typing import Dict, List, Any
from app.utils.constants import THEME_CATEGORIES
from app.utils.helpers import clean_text

class ThemeAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.kmeans = None
        self.theme_keywords = {
            'support': ['great', 'amazing', 'awesome', 'love', 'perfect', 'excellent', 'fantastic', 'wonderful'],
            'criticism': ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid'],
            'question': ['how', 'what', 'why', 'when', 'where', 'which', 'can', 'could', 'would'],
            'technical': ['code', 'api', 'bug', 'feature', 'update', 'version', 'system', 'error'],
            'personal': ['i', 'me', 'my', 'myself', 'experience', 'story', 'happened', 'feel'],
            'humor': ['lol', 'haha', 'funny', 'joke', 'hilarious', '😂', '🤣', 'laugh'],
            'news': ['breaking', 'report', 'announced', 'update', 'news', 'official', 'confirmed'],
            'debate': ['disagree', 'wrong', 'argue', 'debate', 'opinion', 'think', 'believe'],
            'suggestion': ['should', 'could', 'suggest', 'recommend', 'idea', 'proposal', 'maybe']
        }
    
    def analyze_themes(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group comments by themes using clustering and keyword analysis"""
        if not comments or len(comments) < 2:
            return self._create_single_theme(comments)
        
        # Prepare texts
        texts = []
        valid_comments = []
        
        for comment in comments:
            text = comment.get('body', '') or comment.get('text', '')
            cleaned_text = clean_text(text)
            
            if len(cleaned_text.split()) >= 3:  # Filter very short comments
                texts.append(cleaned_text)
                valid_comments.append(comment)
        
        if len(texts) < 2:
            return self._create_single_theme(valid_comments)
        
        try:
            # Use both clustering and keyword-based classification
            clustered_themes = self._cluster_comments(texts, valid_comments)
            keyword_themes = self._classify_by_keywords(texts, valid_comments)
            
            # Merge and prioritize keyword-based classification
            merged_themes = self._merge_themes(clustered_themes, keyword_themes)
            
            return merged_themes
            
        except Exception as e:
            print(f"Error in theme analysis: {str(e)}")
            return self._create_single_theme(valid_comments)
    
    def _cluster_comments(self, texts: List[str], comments: List[Dict]) -> Dict[str, Any]:
        """Cluster comments using TF-IDF and K-means"""
        try:
            # TF-IDF Vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Determine optimal cluster count
            n_clusters = min(max(2, len(texts) // 3), 6)
            
            # K-means clustering
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.kmeans.fit_predict(tfidf_matrix)
            
            # Group comments by clusters
            themes = {}
            for i, cluster_id in enumerate(clusters):
                theme_name = f"Theme {cluster_id + 1}"
                
                if theme_name not in themes:
                    themes[theme_name] = {
                        'comments': [],
                        'keywords': [],
                        'sentiment_summary': {'positive': 0, 'negative': 0, 'neutral': 0},
                        'avg_score': 0
                    }
                
                themes[theme_name]['comments'].append(comments[i])
            
            # Extract keywords for each theme
            feature_names = self.vectorizer.get_feature_names_out()
            for i, theme_name in enumerate(themes.keys()):
                cluster_center = self.kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                themes[theme_name]['keywords'] = [feature_names[idx] for idx in top_indices]
            
            return themes
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            return {}
    
    def _classify_by_keywords(self, texts: List[str], comments: List[Dict]) -> Dict[str, Any]:
        """Classify comments using keyword matching"""
        themes = {}
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            best_theme = 'general'
            best_score = 0
            
            # Check against keyword categories
            for theme_type, keywords in self.theme_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > best_score:
                    best_score = score
                    best_theme = theme_type
            
            # Create theme if doesn't exist
            if best_theme not in themes:
                themes[best_theme] = {
                    'comments': [],
                    'keywords': self.theme_keywords.get(best_theme, []),
                    'sentiment_summary': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'avg_score': 0
                }
            
            themes[best_theme]['comments'].append(comments[i])
        
        return themes
    
    def _merge_themes(self, clustered_themes: Dict, keyword_themes: Dict) -> Dict[str, Any]:
        """Merge clustering and keyword-based themes"""
        # Prioritize keyword-based themes as they're more interpretable
        final_themes = {}
        
        for theme_name, theme_data in keyword_themes.items():
            if theme_data['comments']:  # Only include non-empty themes
                display_name = self._get_theme_display_name(theme_name)
                final_themes[display_name] = self._analyze_theme_sentiment(theme_data)
        
        # Add any significant clustered themes not covered by keywords
        for theme_name, theme_data in clustered_themes.items():
            if len(theme_data['comments']) >= 2:  # Only significant clusters
                display_name = f"📊 {theme_name}"
                if display_name not in final_themes:
                    final_themes[display_name] = self._analyze_theme_sentiment(theme_data)
        
        return final_themes
    
    def _get_theme_display_name(self, theme_name: str) -> str:
        """Get display name with emoji for theme"""
        theme_info = THEME_CATEGORIES.get(theme_name, {'emoji': '💬'})
        emoji = theme_info['emoji']
        
        name_mapping = {
            'support': 'Positive Feedback',
            'criticism': 'Critical Comments', 
            'question': 'Questions & Inquiries',
            'technical': 'Technical Discussion',
            'personal': 'Personal Experiences',
            'humor': 'Humor & Jokes',
            'news': 'News & Updates',
            'debate': 'Debates & Arguments', 
            'suggestion': 'Suggestions & Ideas',
            'general': 'General Discussion'
        }
        display_name = name_mapping.get(theme_name, theme_name.title())
        return f"{emoji} {display_name}"
    
    def _analyze_theme_sentiment(self, theme_data: Dict) -> Dict[str, Any]:
        """Analyze sentiment distribution within a theme"""
        comments = theme_data['comments']
        if not comments:
            return theme_data
        
        # Calculate average score
        scores = [comment.get('score', 0) for comment in comments]
        theme_data['avg_score'] = sum(scores) / len(scores) if scores else 0
        
        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'love', 'amazing', 'perfect', 'excellent']
        negative_words = ['bad', 'hate', 'terrible', 'awful', 'worst', 'horrible']
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for comment in comments:
            text = (comment.get('body', '') or comment.get('text', '')).lower()
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                sentiment_counts['positive'] += 1
            elif neg_count > pos_count:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
        
        theme_data['sentiment_summary'] = sentiment_counts
        return theme_data
    
    def _create_single_theme(self, comments: List[Dict]) -> Dict[str, Any]:
        """Create a single theme for few comments"""
        return {
            "💬 General Discussion": {
                'comments': comments,
                'keywords': ['discussion', 'comments'],
                'sentiment_summary': {'positive': 0, 'negative': 0, 'neutral': len(comments)},
                'avg_score': 0
            }
        }

