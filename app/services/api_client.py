import requests
import time
from typing import Dict, Any, List, Tuple, Optional
from app.config import Config
from app.utils.helpers import extract_social_url_info

class SocialAPIClient:
    def __init__(self):
        self.x_bearer = Config.X_BEARER_TOKEN
        self.reddit_headers = {"User-Agent": "SocialAnalyzerPro/1.0"}
        self.rate_limit_delay = 1  # seconds between requests
    
    def fetch_content(self, url: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Main method to fetch content from any supported platform"""
        platform, post_id = extract_social_url_info(url)
        
        if not platform or not post_id:
            return None, None
        
        try:
            if platform == 'twitter':
                return platform, self._fetch_twitter_content(post_id)
            elif platform == 'reddit':
                return platform, self._fetch_reddit_content(post_id)
            else:
                return None, None
                
        except Exception as e:
            print(f"Error fetching {platform} content: {str(e)}")
            return platform, None
    
    def _fetch_twitter_content(self, tweet_id: str) -> Dict[str, Any]:
        """Fetch Twitter/X content"""
        if not self.x_bearer:
            raise ValueError("Twitter Bearer Token not configured")
        
        url = f"https://api.twitter.com/2/tweets/{tweet_id}"
        params = {
            'tweet.fields': 'created_at,author_id,public_metrics,geo,lang,context_annotations,conversation_id',
            'expansions': 'author_id,geo.place_id',
            'user.fields': 'username,name,location,verified,public_metrics,description',
            'place.fields': 'full_name,country,geo'
        }
        headers = {"Authorization": f"Bearer {self.x_bearer}"}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        time.sleep(self.rate_limit_delay)
        return response.json()
    
    def _fetch_reddit_content(self, post_id: str) -> Dict[str, Any]:
        """Fetch Reddit content"""
        url = f"https://www.reddit.com/comments/{post_id}.json"
        
        response = requests.get(url, headers=self.reddit_headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse main post
        main_post = data[0]['data']['children'][0]['data']
        
        # Parse comments
        comments = []
        if len(data) > 1:
            comments = self._parse_reddit_comments(data[1]['data']['children'])
        
        time.sleep(self.rate_limit_delay)
        
        return {
            'main_post': main_post,
            'comments': comments,
            'total_comments': len(comments)
        }
    
    def _parse_reddit_comments(self, comment_data: List[Dict]) -> List[Dict[str, Any]]:
        """Parse Reddit comments recursively"""
        comments = []
        
        for item in comment_data:
            if item['kind'] == 't1' and 'body' in item['data']:
                comment = item['data']
                
                parsed_comment = {
                    'id': comment['id'],
                    'body': comment['body'],
                    'score': comment.get('score', 0),
                    'author': comment.get('author', '[deleted]'),
                    'created_utc': comment.get('created_utc', 0),
                    'permalink': comment.get('permalink', ''),
                    'replies': []
                }
                
                # Parse replies recursively
                if 'replies' in comment and isinstance(comment['replies'], dict):
                    if 'data' in comment['replies'] and 'children' in comment['replies']['data']:
                        parsed_comment['replies'] = self._parse_reddit_comments(
                            comment['replies']['data']['children']
                        )
                
                comments.append(parsed_comment)
        
        return comments
    
    def get_platform_info(self, platform: str) -> Dict[str, str]:
        """Get platform metadata"""
        platform_info = {
            'twitter': {
                'name': 'Twitter/X',
                'emoji': '🐦',
                'color': '#1da1f2',
                'api_limit': '300 requests/15min'
            },
            'reddit': {
                'name': 'Reddit', 
                'emoji': '👽',
                'color': '#ff4500',
                'api_limit': '60 requests/min'
            }
        }
        
        return platform_info.get(platform, {
            'name': platform.title(),
            'emoji': '🌐', 
            'color': '#6366f1',
            'api_limit': 'Unknown'
        })

