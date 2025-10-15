# Social Analyzer

Social Analyzer is a complete social media analysis tool that uses **Artificial Intelligence (AI) and Transformer models** to automatically extract feelings, emotions, and patterns from social media content at scale.

## Core Features

### 1. Data Processing Pipeline
* **Platform Identification:** Determines if the input URL is from **Twitter** or **Reddit**.
* **Content Extraction:** Extracts the main text and relevant metadata (likes, retweets, replies, score, comments, etc.).
* **Text Cleaning:** Prepares raw text for AI models by removing URLs, handling repeated characters, and extracting entities like hashtags and mentions.

### 2. AI-Powered Analysis
* **Sentiment Analysis:** Categorizes the text's overall tone as **POSITIVE**, **NEGATIVE**, or **NEUTRAL** using the `cardiffnlp/twitter-roberta-base-sentiment-latest` model (a **ROBERTa Transformer Model**).
* **Emotion Analysis:** Detects the specific feeling from a list of seven emotions: **Joy, Sadness, Anger, Fear, Surprise, Disgust,** and **Neutral**, using the `j-hartmann/emotion-english-distilroberta-base` model.
* **Comment Analysis:** Processes all comments associated with a post, grouping them by theme using **K-Means clustering** and calculating a **Controversy Score** based on the sentiment distribution.

### 3. Visualization and Reporting
* **Visual Data:** Uses **Plotly.js** to create charts for instant insights.
* **Key Visualizations:** Includes a **Sentiment Confidence Gauge**, an **Emotion Radar Chart**, and a **Comment Sentiment Timeline**.
* **Emoji Mapping:** Assigns visual sentiment representation using emojis (e.g., :smiling_face:, :disappointed_face:).

## Technical Architecture

The application is structured for clarity and maintainability:

* **`models/`**: AI model wrappers (`SentimentAnalyzer`, `EmotionDetector`).
* **`services/`**: Business logic (`DataProcessor` orchestrating the pipeline).
* **`visualizer.py`**: Handles all chart generation.
* **`main.py`**: The application's entry point and web interface (likely using Streamlit, judging by the `st.plotly_chart` usage).

### Deployment 
* **Dockerized:** Uses a `python:3.9-slim` base image for a consistent environment. The build process includes setting up a model cache directory and creating a non-root user for security.

