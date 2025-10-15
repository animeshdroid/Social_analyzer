# 🧠 Social Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![HuggingFace](https://img.shields.io/badge/NLP-HuggingFace-yellow)
![Docker](https://img.shields.io/badge/Containerized-Docker-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Social Analyzer Pro** is an AI-powered sentiment and emotion analysis system that extracts and interprets posts from **X (Twitter)** and **Reddit**.  
It transforms raw social media data into meaningful emotional insights — complete with interactive charts, emoji-based analysis, and detailed visualizations.

---

## 🚀 Features

✅ **Multi-Platform Analysis**
- Analyze tweets and Reddit posts automatically.  
- Extract metadata like likes, retweets, upvotes, and comments.

🧠 **Sentiment + Emotion Detection**
- Uses state-of-the-art transformer models:
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - `j-hartmann/emotion-english-distilroberta-base`
- Detects emotions: Joy 😊, Sadness 😢, Anger 😠, Fear 😨, Surprise 😲, Disgust 🤢, Neutral 😐

📊 **Interactive Visualization**
- Sentiment confidence gauges  
- Emotion radar charts  
- Comment sentiment timelines  
- Engagement metric bars  

💬 **Advanced Comment Analysis**
- Clusters comments into themes  
- Detects controversial discussions  
- Tracks sentiment shifts over time  

⚙️ **Fully Containerized**
- Docker-ready for consistent local or cloud deployment.  
- Includes health checks and model caching for faster inference.

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Language** | Python 3.9+ |
| **Framework** | Streamlit |
| **AI Models** | RoBERTa, DistilRoBERTa |
| **Libraries** | Transformers, NLTK, Plotly, Pandas, NumPy |
| **APIs** | X (Twitter) API v2, Reddit API |
| **Deployment** | Docker |
| **Testing** | unittest |

---

## 🏗️ Project Structure
```python
Social_Analyzer/
│
├── app/
│   ├── main.py               # Streamlit entry point
│   ├── models/               # Sentiment + emotion model wrappers
│   ├── services/             # Processing logic & API clients
│   ├── visualization/        # Plotly chart builders
│   └── utils/                # Helpers (cleaning, formatting, config)
│
├── requirements.txt          # Dependencies
├── Dockerfile                # Container configuration
├── .env.example              # Example API credentials
└── README.md                 # This file
```

---

## 🔍 Workflow
### 1️⃣ Paste a Social Media URL
```python
https://twitter.com/elonmusk/status/12345
```
### 2️⃣ Processing Pipeline
```python
USER INPUT
   ↓
URL RECOGNITION
   ↓
CONTENT EXTRACTION
   ↓
TEXT CLEANING
   ↓
AI ANALYSIS (Sentiment + Emotion)
   ↓
VISUALIZATION
   ↓
FINAL INSIGHTS
```

---

## 🐳 Docker Setup
### 1️⃣ Build the image
```python
docker build -t social_analyzer
```
### 2️⃣ Run the container
```python
docker run -p 8501:8501 social_analyzer
```
