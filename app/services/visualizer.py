import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from typing import Dict, Any, List
from app.utils.constants import COLORS, EMOTION_EMOJIS, SENTIMENT_EMOJIS, THEME_CATEGORIES

class Visualizer:
    def __init__(self):
        self.colors = {
            'primary': '#6366f1',
            'secondary': '#8b5cf6', 
            'success': '#22c55e',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'info': '#3b82f6',
            'accent': '#f59e0b',   # <-- Add this line
            'light': '#f8fafc',
            'dark': '#1e293b',
            'muted': '#64748b'
        }
        self.chart_config = {
            'displayModeBar': False,
            'staticPlot': False
        }
    
    def create_sentiment_gauge(self, sentiment_data: Dict[str, Any]) -> go.Figure:
        """Create sentiment confidence gauge"""
        confidence = sentiment_data.get('confidence', 0)
        sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
        
        # Map sentiment to color
        color_map = {
            'POSITIVE': self.colors['success'],
            'NEGATIVE': self.colors['danger'],
            'NEUTRAL': self.colors['info']
        }
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Sentiment Confidence<br>{SENTIMENT_EMOJIS.get(sentiment, '😐')} {sentiment}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color_map.get(sentiment, self.colors['primary'])},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(128,128,128,0.2)"},
                    {'range': [50, 80], 'color': "rgba(255,255,0,0.2)"},
                    {'range': [80, 100], 'color': "rgba(0,255,0,0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 14},
            height=300
        )
        
        return fig
    
    def create_emotion_radar(self, emotion_data: Dict[str, Any]) -> go.Figure:
        """Create emotion radar chart"""
        emotions = emotion_data.get('all_emotions', {})
        
        if not emotions:
            return self._create_empty_chart("No emotion data available")
        
        # Prepare data
        emotion_names = list(emotions.keys())
        emotion_scores = list(emotions.values())
        emotion_emojis = [EMOTION_EMOJIS.get(emotion, '😐') for emotion in emotion_names]
        
        # Create labels with emojis
        labels = [f"{emoji} {emotion.title()}" for emoji, emotion in zip(emotion_emojis, emotion_names)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=emotion_scores,
            theta=labels,
            fill='toself',
            name='Emotions',
            line_color=self.colors['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    color='white'
                ),
                angularaxis=dict(
                    color='white'
                )
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 12},
            height=400,
            title="🎭 Emotion Analysis"
        )
        
        return fig
    
    def create_engagement_metrics(self, metrics: Dict[str, Any], platform: str) -> go.Figure:
        """Create engagement metrics visualization"""
        if platform == 'twitter':
            categories = ['Likes', 'Retweets', 'Replies', 'Quotes']
            values = [
                metrics.get('likes', 0),
                metrics.get('retweets', 0), 
                metrics.get('replies', 0),
                metrics.get('quotes', 0)
            ]
            emojis = ['❤️', '🔄', '💬', '📝']
        else:  # reddit
            categories = ['Score', 'Comments']
            values = [metrics.get('score', 0), metrics.get('num_comments', 0)]
            emojis = ['⬆️', '💬']
        
        # Create labels with emojis
        labels = [f"{emoji} {cat}" for emoji, cat in zip(emojis, categories)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color='#f59e0b',  
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="📈 Engagement Metrics",
            xaxis_title="Count",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 12},
            height=300
        )
        
        return fig
    
    def create_theme_distribution(self, theme_data: Dict[str, Any]) -> go.Figure:
        """Create theme distribution chart"""
        if not theme_data:
            return self._create_empty_chart("No theme data available")
        
        # Extract theme names and comment counts
        theme_names = []
        comment_counts = []
        colors = []
        
        for theme_name, theme_info in theme_data.items():
            theme_names.append(theme_name)
            comment_counts.append(len(theme_info.get('comments', [])))
            
            # Extract theme type for color
            theme_type = theme_name.lower().split()[-1] if ' ' in theme_name else 'general'
            theme_color = THEME_CATEGORIES.get(theme_type, {'color': self.colors['primary']})['color']
            colors.append(theme_color)
        
        fig = go.Figure(data=[
            go.Bar(
                x=comment_counts,
                y=theme_names,
                orientation='h',
                marker_color=colors,
                text=comment_counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="🏷️ Comment Themes Distribution",
            xaxis_title="Number of Comments",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 12},
            height=400
        )
        
        return fig
    
    def create_sentiment_timeline(self, comments: List[Dict[str, Any]]) -> go.Figure:
        """Create sentiment timeline for comments"""
        if not comments:
            return self._create_empty_chart("No comment data available")
        
        # Prepare data
        timestamps = []
        sentiments = []
        scores = []
        
        for comment in sorted(comments, key=lambda x: x.get('created_utc', 0)):
            timestamps.append(comment.get('time_ago', 'Unknown'))
            sentiment = comment.get('sentiment', {}).get('sentiment', 'NEUTRAL')
            sentiments.append(sentiment)
            
            # Convert sentiment to score
            sentiment_score = {
                'POSITIVE': 1,
                'NEUTRAL': 0,
                'NEGATIVE': -1
            }.get(sentiment, 0)
            scores.append(sentiment_score)
        
        # Create color mapping
        colors = [
            self.colors['success'] if s == 'POSITIVE' 
            else self.colors['danger'] if s == 'NEGATIVE'
            else self.colors['info']
            for s in sentiments
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(timestamps))),
            y=scores,
            mode='markers+lines',
            marker=dict(
                color=colors,
                size=8,
                line=dict(width=2, color='white')
            ),
            line=dict(color=self.colors['primary'], width=2),
            name='Sentiment Timeline',
            hovertemplate='<b>Time:</b> %{text}<br><b>Sentiment:</b> %{y}<extra></extra>',
            text=timestamps
        ))
        
        fig.update_layout(
            title="📊 Comment Sentiment Timeline",
            xaxis_title="Comment Index",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1.2, 1.2]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 12},
            height=400
        )
        
        return fig
    
    def create_metrics_dashboard(self, processed_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive metrics dashboard"""
        metrics = processed_data.get('metrics', {})
        analysis = processed_data.get('analysis', {})
        platform = processed_data.get('platform', 'unknown')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '📊 Engagement Overview',
                '🎭 Sentiment Analysis', 
                '📖 Readability Metrics',
                '🏷️ Entity Analysis'
            ],
            specs=[
                [{'type': 'bar'}, {'type': 'indicator'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )
        
        # Engagement metrics
        if platform == 'twitter':
            engagement_labels = ['Likes', 'Retweets', 'Replies']
            engagement_values = [
                metrics.get('likes', 0),
                metrics.get('retweets', 0),
                metrics.get('replies', 0)
            ]
        else:
            engagement_labels = ['Score', 'Comments']
            engagement_values = [
                metrics.get('score', 0),
                metrics.get('num_comments', 0)
            ]
        
        fig.add_trace(
            go.Bar(x=engagement_labels, y=engagement_values, name='Engagement'),
            row=1, col=1
        )
        
        # Sentiment indicator
        sentiment_data = analysis.get('sentiment', {})
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sentiment_data.get('confidence', 0) * 100,
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=self.colors['primary']),
                    bgcolor="rgba(255,255,255,0.1)"
                ),
                title={'text': 'Confidence %'}
            ),
            row=1, col=2
        )
        
        # Readability metrics
        readability = analysis.get('readability', {})
        fig.add_trace(
            go.Bar(
                x=['Words', 'Sentences', 'Avg Words/Sentence'],
                y=[
                    readability.get('word_count', 0),
                    readability.get('sentence_count', 0),
                    readability.get('avg_words_per_sentence', 0)
                ],
                name='Readability'
            ),
            row=2, col=1
        )
        
        # Entity counts
        entities = analysis.get('entities', {})
        entity_counts = [len(entities.get(key, [])) for key in ['hashtags', 'mentions', 'urls']]
        fig.add_trace(
            go.Bar(x=['Hashtags', 'Mentions', 'URLs'], y=entity_counts, name='Entities'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 10}
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font={'size': 16, 'color': 'white'}
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=300
        )
        
        return fig

