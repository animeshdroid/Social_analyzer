from typing import Dict, Any, List
from datetime import datetime
from app.models.sentiment_model import SentimentAnalyzer
from app.models.emotion_detector import EmotionDetector
from app.models.theme_analyzer import ThemeAnalyzer
from app.utils.helpers import clean_text, format_number, get_time_ago

class DataProcessor:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.theme_analyzer = ThemeAnalyzer()
    
    def process_content(self, platform: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process content based on platform"""
        if platform == 'twitter':
            return self._process_twitter_data(data)
        elif platform == 'reddit':
            return self._process_reddit_data(data)
        else:
            return {}
    
    def _process_twitter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Twitter/X data"""
        tweet = data.get('data', {})
        includes = data.get('includes', {})
        
        # Extract basic info
        text = tweet.get('text', '')
        metrics = tweet.get('public_metrics', {})
        author_info = includes.get('users', [{}])[0] if includes.get('users') else {}
        
        # Clean text for analysis
        cleaned_text = clean_text(text)
        
        # Perform AI analysis
        sentiment_result = self.sentiment_analyzer.analyze(cleaned_text)
        emotion_result = self.emotion_detector.detect_emotions(cleaned_text)
        
        # Calculate engagement metrics
        total_engagement = sum([
            metrics.get('like_count', 0),
            metrics.get('retweet_count', 0),
            metrics.get('reply_count', 0),
            metrics.get('quote_count', 0)
        ])
        
        # Process location data
        location_info = self._process_location(author_info.get('location', ''))
        
        return {
            'platform': 'twitter',
            'content': {
                'text': text,
                'cleaned_text': cleaned_text,
                'created_at': tweet.get('created_at'),
                'lang': tweet.get('lang', 'en')
            },
            'author': {
                'username': author_info.get('username', 'Unknown'),
                'name': author_info.get('name', 'Unknown'),
                'verified': author_info.get('verified', False),
                'followers': author_info.get('public_metrics', {}).get('followers_count', 0),
                'location': author_info.get('location', '')
            },
            'metrics': {
                'likes': metrics.get('like_count', 0),
                'retweets': metrics.get('retweet_count', 0),
                'replies': metrics.get('reply_count', 0),
                'quotes': metrics.get('quote_count', 0),
                'total_engagement': total_engagement,
                'engagement_rate': self._calculate_engagement_rate(total_engagement, author_info.get('public_metrics', {}).get('followers_count', 0))
            },
            'analysis': {
                'sentiment': sentiment_result,
                'emotion': emotion_result,
                'readability': self._analyze_readability(cleaned_text),
                'entities': self._extract_entities(text)
            },
            'location': location_info,
            'processed_at': datetime.now().isoformat()
        }
    
    def _process_reddit_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Reddit data"""
        main_post = data.get('main_post', {})
        comments = data.get('comments', [])
        
        # Combine title and selftext for analysis
        title = main_post.get('title', '')
        selftext = main_post.get('selftext', '')
        full_text = f"{title}\n{selftext}".strip()
        cleaned_text = clean_text(full_text)
        
        # Perform AI analysis on main post
        sentiment_result = self.sentiment_analyzer.analyze(cleaned_text)
        emotion_result = self.emotion_detector.detect_emotions(cleaned_text)
        
        # Process comments
        processed_comments = []
        for comment in comments[:50]:  # Limit to top 50 comments for performance
            comment_text = clean_text(comment.get('body', ''))
            if len(comment_text.split()) >= 3:  # Only process substantial comments
                comment_sentiment = self.sentiment_analyzer.analyze(comment_text)
                comment_emotion = self.emotion_detector.detect_emotions(comment_text)
                
                processed_comments.append({
                    'id': comment.get('id'),
                    'text': comment.get('body', ''),
                    'cleaned_text': comment_text,
                    'author': comment.get('author', '[deleted]'),
                    'score': comment.get('score', 0),
                    'created_utc': comment.get('created_utc', 0),
                    'sentiment': comment_sentiment,
                    'emotion': comment_emotion,
                    'time_ago': get_time_ago(comment.get('created_utc', 0))
                })
        
        # Analyze comment themes
        theme_analysis = {}
        if processed_comments:
            theme_analysis = self.theme_analyzer.analyze_themes(processed_comments)
        
        return {
            'platform': 'reddit',
            'content': {
                'title': title,
                'text': selftext,
                'full_text': full_text,
                'cleaned_text': cleaned_text,
                'subreddit': main_post.get('subreddit', ''),
                'created_utc': main_post.get('created_utc', 0)
            },
            'author': {
                'username': main_post.get('author', '[deleted]'),
                'is_submitter': main_post.get('is_submitter', False)
            },
            'metrics': {
                'score': main_post.get('score', 0),
                'upvote_ratio': main_post.get('upvote_ratio', 0),
                'num_comments': main_post.get('num_comments', 0),
                'total_awards': main_post.get('total_awards_received', 0)
            },
            'analysis': {
                'sentiment': sentiment_result,
                'emotion': emotion_result,
                'readability': self._analyze_readability(cleaned_text),
                'entities': self._extract_entities(full_text)
            },
            'comments': {
                'processed_comments': processed_comments,
                'total_processed': len(processed_comments),
                'theme_analysis': theme_analysis,
                'sentiment_distribution': self._calculate_comment_sentiment_distribution(processed_comments)
            },
            'processed_at': datetime.now().isoformat()
        }
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        if not text:
            return {'word_count': 0, 'sentence_count': 0, 'avg_words_per_sentence': 0, 'reading_time': 0}
        
        words = text.split()
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        return {
            'word_count': len(words),
            'sentence_count': sentences,
            'avg_words_per_sentence': round(len(words) / sentences, 1),
            'character_count': len(text),
            'reading_time': max(1, len(words) // 200)  # Assuming 200 WPM
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        import re
        
        return {
            'hashtags': re.findall(r'#(\w+)', text),
            'mentions': re.findall(r'@(\w+)', text),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'phone_numbers': re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
        }
    
    def _calculate_engagement_rate(self, total_engagement: int, followers: int) -> float:
        """Calculate engagement rate"""
        if followers == 0:
            return 0.0
        return round((total_engagement / followers) * 100, 2)
    
    def _calculate_comment_sentiment_distribution(self, comments: List[Dict]) -> Dict[str, int]:
        """Calculate sentiment distribution across comments"""
        distribution = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        
        for comment in comments:
            sentiment = comment.get('sentiment', {}).get('sentiment', 'NEUTRAL')
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
        
        return distribution
    
    def _process_location(self, location_string: str) -> Dict[str, Any]:
        """Process location information"""
        if not location_string:
            return {}
        
        # Simple location processing - can be enhanced with geocoding
        return {
            'raw': location_string,
            'processed': location_string.strip(),
            'has_location': bool(location_string.strip())
        }

