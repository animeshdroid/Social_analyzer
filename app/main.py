import streamlit as st
import time
from datetime import datetime
import pandas as pd
from typing import Dict, Any

from app.config import Config
from app.services.api_client import SocialAPIClient
from app.services.data_processor import DataProcessor
from app.services.visualizer import Visualizer
from app.utils.constants import PLATFORM_EMOJIS, SENTIMENT_EMOJIS, EMOTION_EMOJIS
from app.utils.helpers import format_number, get_time_ago

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalist design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .hero-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.85);
        font-weight: 400;
        margin-bottom: 0;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        color: white;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.8rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.9;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .metric-emoji {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .content-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
    }
    
    .platform-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.15);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .sentiment-positive { color: #4ade80; }
    .sentiment-negative { color: #f87171; }
    .sentiment-neutral { color: #60a5fa; }
    
    .theme-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 0.4rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255,255,255,0.7);
        border-radius: 12px;
        font-weight: 500;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        font-weight: 600;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 1rem 1.5rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.1) !important;
    }
    
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 3rem;
        color: white;
    }
    
    .loading-spinner {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    api_client = SocialAPIClient()
    data_processor = DataProcessor()
    visualizer = Visualizer()
    return api_client, data_processor, visualizer

# Load components
api_client, data_processor, visualizer = initialize_components()

# Main app header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌟 Social Analyzer Pro</div>
    <div class="hero-subtitle">AI-Powered Social Media Intelligence • Real-time Analysis • Advanced Insights</div>
</div>
""", unsafe_allow_html=True)

# URL Input Section
st.markdown("## 🔗 Analyze Social Content")
url_input = st.text_input(
    "Enter Twitter/X or Reddit URL:",
    placeholder="https://twitter.com/username/status/... or https://reddit.com/r/subreddit/comments/...",
    help="Paste any public social media URL for comprehensive AI analysis"
)

if url_input:
    # Extract platform and post ID
    platform, post_data = api_client.fetch_content(url_input)
    
    if platform and post_data:
        # Show loading state
        with st.container():
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner">🤖</div>
                <h3>Analyzing with AI models...</h3>
                <p>Processing content with advanced NLP and emotion detection</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process data with AI models
            with st.spinner("Running HuggingFace transformers..."):
                processed_data = data_processor.process_content(platform, post_data)
            
        # Clear loading state
        st.empty()
        
        if processed_data:
            # Platform header
            platform_info = api_client.get_platform_info(platform)
            st.markdown(f"""
            <div class="content-card">
                <div class="platform-badge">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{platform_info['emoji']}</span>
                    <span>Analyzing {platform_info['name']} Content</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create main tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", 
                "🎭 AI Analysis", 
                "💬 Comments", 
                "📈 Insights", 
                "📄 Export"
            ])
            
            with tab1:
                # Overview metrics
                metrics = processed_data.get('metrics', {})
                analysis = processed_data.get('analysis', {})
                content = processed_data.get('content', {})
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment = analysis.get('sentiment', {})
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-emoji">{sentiment.get('emoji', '😐')}</div>
                        <div class="metric-value sentiment-{sentiment.get('sentiment', 'neutral').lower()}">{sentiment.get('sentiment', 'NEUTRAL')}</div>
                        <div class="metric-label">AI Sentiment</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    emotion = analysis.get('emotion', {})
                    dominant_emotion = emotion.get('dominant_emotion', 'neutral')
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-emoji">{emotion.get('emoji', '😐')}</div>
                        <div class="metric-value">{dominant_emotion.title()}</div>
                        <div class="metric-label">Dominant Emotion</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if platform == 'twitter':
                        engagement = metrics.get('total_engagement', 0)
                        label = "Total Engagement"
                    else:
                        engagement = metrics.get('score', 0)
                        label = "Reddit Score"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-emoji">📈</div>
                        <div class="metric-value">{format_number(engagement)}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    readability = analysis.get('readability', {})
                    word_count = readability.get('word_count', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-emoji">📝</div>
                        <div class="metric-value">{word_count}</div>
                        <div class="metric-label">Words</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Content display
                st.markdown("### 📱 Content")
                if platform == 'twitter':
                    text = content.get('text', '')
                    author = processed_data.get('author', {})
                    st.markdown(f"""
                    <div class="content-card">
                        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1rem;">"{text}"</p>
                        <p style="opacity: 0.8;"><strong>@{author.get('username', 'unknown')}</strong> • {author.get('followers', 0)} followers</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:  # Reddit
                    title = content.get('title', '')
                    text = content.get('text', '')
                    subreddit = content.get('subreddit', '')
                    st.markdown(f"""
                    <div class="content-card">
                        <h4 style="margin-bottom: 1rem;">{title}</h4>
                        <p style="line-height: 1.6; margin-bottom: 1rem;">{text[:500]}{'...' if len(text) > 500 else ''}</p>
                        <p style="opacity: 0.8;"><strong>r/{subreddit}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Engagement visualization
                st.markdown("### 📊 Engagement Breakdown")
                fig_engagement = visualizer.create_engagement_metrics(metrics, platform)
                st.plotly_chart(fig_engagement, use_container_width=True, config={'displayModeBar': False})
            
            with tab2:
                # AI Analysis results
                st.markdown("### 🤖 HuggingFace Transformer Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment gauge
                    st.markdown("#### 🎭 Sentiment Analysis")
                    fig_sentiment = visualizer.create_sentiment_gauge(analysis.get('sentiment', {}))
                    st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
                    
                    # Sentiment details
                    sentiment_details = analysis.get('sentiment', {})
                    confidence = sentiment_details.get('confidence', 0)
                    polarity = sentiment_details.get('polarity', 0)
                    
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>📋 Sentiment Details</h4>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>Polarity:</strong> {polarity:.3f}</p>
                        <p><strong>Model:</strong> {Config.SENTIMENT_MODEL.split('/')[-1]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Emotion radar
                    st.markdown("#### 🎭 Emotion Detection")
                    fig_emotion = visualizer.create_emotion_radar(analysis.get('emotion', {}))
                    st.plotly_chart(fig_emotion, use_container_width=True, config={'displayModeBar': False})
                    
                    # Top emotions
                    top_emotions = analysis.get('emotion', {}).get('top_3_emotions', [])
                    if top_emotions:
                        st.markdown("**🏆 Top 3 Emotions:**")
                        for i, emotion_data in enumerate(top_emotions, 1):
                            emotion_name = emotion_data.get('emotion', '').title()
                            score = emotion_data.get('score', 0)
                            emoji = emotion_data.get('emoji', '😐')
                            st.markdown(f"{i}. {emoji} **{emotion_name}** - {score:.1%}")
                
                # Entity analysis
                st.markdown("### 🏷️ Entity Extraction")
                entities = analysis.get('entities', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hashtags = entities.get('hashtags', [])
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>#️⃣ Hashtags ({len(hashtags)})</h4>
                        <p>{', '.join(['#' + tag for tag in hashtags[:5]]) if hashtags else 'None found'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    mentions = entities.get('mentions', [])
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>@️ Mentions ({len(mentions)})</h4>
                        <p>{', '.join(['@' + mention for mention in mentions[:5]]) if mentions else 'None found'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    urls = entities.get('urls', [])
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>🔗 URLs ({len(urls)})</h4>
                        <p>{'Found external links' if urls else 'No external links'}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                # Comments analysis (Reddit only)
                if platform == 'reddit':
                    comments_data = processed_data.get('comments', {})
                    processed_comments = comments_data.get('processed_comments', [])
                    theme_analysis = comments_data.get('theme_analysis', {})
                    
                    if processed_comments:
                        st.markdown(f"### 💬 Comment Analysis ({len(processed_comments)} comments)")
                        
                        # Comment sentiment distribution
                        sentiment_dist = comments_data.get('sentiment_distribution', {})
                        if sentiment_dist:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                positive_count = sentiment_dist.get('POSITIVE', 0)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-emoji">🟢</div>
                                    <div class="metric-value sentiment-positive">{positive_count}</div>
                                    <div class="metric-label">Positive Comments</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                neutral_count = sentiment_dist.get('NEUTRAL', 0)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-emoji">🟡</div>
                                    <div class="metric-value sentiment-neutral">{neutral_count}</div>
                                    <div class="metric-label">Neutral Comments</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                negative_count = sentiment_dist.get('NEGATIVE', 0)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-emoji">🔴</div>
                                    <div class="metric-value sentiment-negative">{negative_count}</div>
                                    <div class="metric-label">Negative Comments</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Theme analysis
                        if theme_analysis:
                            st.markdown("### 🏷️ AI Theme Grouping")
                            fig_themes = visualizer.create_theme_distribution(theme_analysis)
                            st.plotly_chart(fig_themes, use_container_width=True, config={'displayModeBar': False})
                            
                            # Show theme details
                            for theme_name, theme_data in theme_analysis.items():
                                with st.expander(f"{theme_name} ({len(theme_data['comments'])} comments)"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        avg_score = theme_data.get('avg_score', 0)
                                        st.write(f"**Average Score:** {avg_score:.1f}")
                                        
                                        keywords = theme_data.get('keywords', [])
                                        st.write(f"**Keywords:** {', '.join(keywords[:5])}")
                                    
                                    with col2:
                                        sentiment_summary = theme_data.get('sentiment_summary', {})
                                        st.write("**Sentiment Breakdown:**")
                                        st.write(f"• Positive: {sentiment_summary.get('positive', 0)}")
                                        st.write(f"• Negative: {sentiment_summary.get('negative', 0)}")
                                        st.write(f"• Neutral: {sentiment_summary.get('neutral', 0)}")
                                    
                                    # Show sample comments
                                    st.write("**Sample Comments:**")
                                    for comment in theme_data['comments'][:3]:
                                        comment_text = comment.get('text', comment.get('body', ''))
                                        author = comment.get('author', 'unknown')
                                        score = comment.get('score', 0)
                                        st.write(f"• **u/{author}** ({score} pts): {comment_text[:150]}...")
                        
                        # Comment timeline
                        if len(processed_comments) > 5:
                            st.markdown("### 📊 Comment Sentiment Timeline")
                            fig_timeline = visualizer.create_sentiment_timeline(processed_comments)
                            st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
                    
                    else:
                        st.info("💡 No comments found or comments are not accessible.")
                
                else:
                    st.info("💡 Comment analysis is available for Reddit posts only.")
            
            with tab4:
                # Advanced insights
                st.markdown("### 🧠 AI-Generated Insights")
                
                # Comprehensive metrics dashboard
                fig_dashboard = visualizer.create_metrics_dashboard(processed_data)
                st.plotly_chart(fig_dashboard, use_container_width=True, config={'displayModeBar': False})
                
                # Generate insights
                insights = []
                
                # Sentiment insights
                sentiment_data = analysis.get('sentiment', {})
                sentiment_label = sentiment_data.get('sentiment', 'NEUTRAL')
                confidence = sentiment_data.get('confidence', 0)
                
                if confidence > 0.8:
                    insights.append(f"🎯 **High Confidence Analysis**: The AI model is {confidence:.1%} confident in detecting {sentiment_label.lower()} sentiment.")
                elif confidence < 0.5:
                    insights.append("⚠️ **Ambiguous Content**: The sentiment is unclear and could be interpreted differently by different people.")
                
                # Engagement insights
                if platform == 'twitter':
                    total_engagement = metrics.get('total_engagement', 0)
                    likes = metrics.get('likes', 0)
                    retweets = metrics.get('retweets', 0)
                    
                    if total_engagement > 10000:
                        insights.append("🔥 **Viral Content**: This tweet has exceptional engagement and viral potential!")
                    elif retweets > likes * 0.3:
                        insights.append("🔄 **High Shareability**: This content is being shared more than typical, indicating strong resonance.")
                    elif likes > retweets * 5:
                        insights.append("❤️ **Appreciation Content**: High like-to-retweet ratio suggests people appreciate but don't feel compelled to share.")
                
                else:  # Reddit
                    score = metrics.get('score', 0)
                    upvote_ratio = metrics.get('upvote_ratio', 0)
                    
                    if upvote_ratio > 0.9:
                        insights.append("👍 **Highly Appreciated**: This post has an exceptional upvote ratio, indicating strong community approval.")
                    elif upvote_ratio < 0.6:
                        insights.append("⚖️ **Controversial Content**: Mixed voting patterns suggest this post is polarizing the community.")
                
                # Emotion insights
                emotion_data = analysis.get('emotion', {})
                dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
                emotion_confidence = emotion_data.get('confidence', 0)
                
                if emotion_confidence > 0.7:
                    insights.append(f"🎭 **Strong Emotional Signal**: The content strongly expresses {dominant_emotion} with {emotion_confidence:.1%} confidence.")
                
                # Content insights
                readability = analysis.get('readability', {})
                word_count = readability.get('word_count', 0)
                
                if word_count > 100:
                    insights.append("📖 **Detailed Content**: Longer-form content may indicate thoughtful expression or detailed information sharing.")
                elif word_count < 20:
                    insights.append("⚡ **Concise Communication**: Short, punchy content optimized for quick consumption and sharing.")
                
                # Display insights
                if insights:
                    for insight in insights:
                        st.markdown(f"""
                        <div class="glass-card">
                            <p>{insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="glass-card">
                        <p>🤖 **Analysis Complete**: Content analyzed successfully. All metrics are within normal ranges.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab5:
                # Export functionality
                st.markdown("### 📄 Export Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Quick Stats")
                    st.write(f"**Platform:** {platform_info['name']}")
                    st.write(f"**Analyzed At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Sentiment:** {sentiment_data.get('sentiment', 'NEUTRAL')}")
                    st.write(f"**Confidence:** {sentiment_data.get('confidence', 0):.1%}")
                    st.write(f"**Word Count:** {readability.get('word_count', 0)}")
                    
                    if platform == 'reddit' and processed_data.get('comments', {}).get('processed_comments'):
                        comment_count = len(processed_data['comments']['processed_comments'])
                        st.write(f"**Comments Analyzed:** {comment_count}")
                
                with col2:
                    st.markdown("#### 🎯 Model Information")
                    st.write(f"**Sentiment Model:** {Config.SENTIMENT_MODEL.split('/')[-1]}")
                    st.write(f"**Emotion Model:** {Config.EMOTION_MODEL.split('/')[-1]}")
                    st.write("**Analysis Type:** HuggingFace Transformers")
                    st.write("**Processing:** Real-time AI analysis")
                
                # Create export data
                export_data = {
                    'platform': platform,
                    'url': url_input,
                    'content': content.get('text', content.get('full_text', ''))[:500],
                    'sentiment': sentiment_data.get('sentiment', 'NEUTRAL'),
                    'sentiment_confidence': sentiment_data.get('confidence', 0),
                    'dominant_emotion': emotion_data.get('dominant_emotion', 'neutral'),
                    'emotion_confidence': emotion_data.get('confidence', 0),
                    'word_count': readability.get('word_count', 0),
                    'total_engagement': metrics.get('total_engagement', metrics.get('score', 0)),
                    'analyzed_at': datetime.now().isoformat(),
                    'model_sentiment': Config.SENTIMENT_MODEL,
                    'model_emotion': Config.EMOTION_MODEL
                }
                
                # Add platform-specific data
                if platform == 'twitter':
                    export_data.update({
                        'likes': metrics.get('likes', 0),
                        'retweets': metrics.get('retweets', 0),
                        'replies': metrics.get('replies', 0)
                    })
                else:  # Reddit
                    export_data.update({
                        'score': metrics.get('score', 0),
                        'upvote_ratio': metrics.get('upvote_ratio', 0),
                        'num_comments': metrics.get('num_comments', 0)
                    })
                    
                    if processed_data.get('comments', {}).get('processed_comments'):
                        export_data['comments_analyzed'] = len(processed_data['comments']['processed_comments'])
                
                df = pd.DataFrame([export_data])
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=csv,
                    file_name=f"{platform}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success("✅ Analysis completed successfully! Your data is ready for export.")
        
        else:
            st.error("❌ Failed to process the content. Please check the URL and try again.")
    
    else:
        st.warning("⚠️ Could not fetch content from the provided URL. Please check:")
        st.write("• The URL is correct and accessible")
        st.write("• The content is public (not private or protected)")

else:
    # Welcome screen
    st.markdown("## 🚀 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🐦 Twitter/X Analysis</h3>
            <ul style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                <li>🤖 HuggingFace transformer sentiment analysis</li>
                <li>🎭 Advanced emotion detection</li>
                <li>📊 Real-time engagement metrics</li>
                <li>🏷️ Entity extraction and categorization</li>
                <li>🧠 AI-powered insights generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>👽 Reddit Analysis</h3>
            <ul style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                <li>💬 Intelligent comment theme grouping</li>
                <li>📈 Sentiment timeline visualization</li>
                <li>🏆 Community engagement analysis</li>
                <li>🎯 Controversy detection</li>
                <li>🔍 Deep discussion insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model information
    #st.markdown("""
    #<div class="glass-card">
    #    <h3>🤖 AI Models Powered by HuggingFace</h3>
    #    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;">
    #       <div>
    #            <h4>Sentiment Analysis</h4>
    #           <p style="color: rgba(255,255,255,0.8);">cardiffnlp/twitter-roberta-base-sentiment-latest</p>
    #            <p style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">State-of-the-art RoBERTa model fine-tuned for social media sentiment analysis</p>
    #        </div>
    #       <div>
    #           <h4>Emotion Detection</h4>
    #           <p style="color: rgba(255,255,255,0.8);">j-hartmann/emotion-english-distilroberta-base</p>
    #           <p style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">Multi-class emotion classification with 7 emotion categories</p>
    #       </div>
    #   </div>
    #</div>
    #""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.8);">
    <h3 style="color: white;">🌟 Social Analyzer Pro</h3>
    <p>Powered by HuggingFace Transformers • Real-time AI Analysis </p>
</div>
""", unsafe_allow_html=True)

