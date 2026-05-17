# Social Analyzer 

A full-stack AI pipeline that analyzes sentiment and emotion from Twitter/Reddit content using transformer models — built with production engineering principles.

-----

## What It Does

Paste a Twitter or Reddit URL. The system fetches the content, runs it through two transformer models, and returns:

- Sentiment classification (positive / negative / neutral) with confidence score
- Emotion detection across 7 categories (joy, anger, sadness, fear, surprise, disgust, neutral)
- Entity extraction (hashtags, mentions, URLs)
- Engagement metrics visualization
- Comment-level analysis with sentiment timeline and controversy scoring
- Theme clustering across comment threads

-----

# Architecture

```
app/
├── models/
│   ├── sentiment_model.py      # RoBERTa sentiment wrapper
│   ├── emotion_detector.py     # DistilRoBERTa emotion wrapper
│   └── theme_analyzer.py       # KMeans comment clustering
├── services/
│   ├── api_client.py           # Twitter / Reddit data fetching
│   ├── data_processor.py       # Orchestrates full analysis pipeline
│   └── visualizer.py           # Plotly chart generation
├── utils/
│   ├── constants.py            # Emoji maps, color palette, config
│   ├── helpers.py              # Text cleaning, entity extraction
└── main.py                     # Streamlit UI
```

**Why this separation?** Each layer has one job. Models know nothing about the UI. The DataProcessor orchestrates without caring which model it calls. This means models are swappable — switching from RoBERTa to a fine-tuned model requires changing one file.

-----

## Key Technical Decisions

### Why RoBERTa over VADER or TextBlob?

VADER and TextBlob use lexicon-based scoring — they look up words in a dictionary. They fail on:

- Sarcasm: *“Oh great, another Monday”*
- Negation: *“not bad at all”*
- Social slang: *“that’s fire”*

`cardiffnlp/twitter-roberta-base-sentiment-latest` was trained specifically on tweets. It understands context through self-attention, so *“not bad”* scores as mildly positive rather than negative. The tradeoff is a ~400MB model load — handled with `@st.cache_resource` so it loads once per session, not per request.

### Why a separate emotion model?

Sentiment (positive/negative) and emotion (anger/joy/fear) answer different questions. A tweet can be *negative* with *sadness* or *negative* with *anger* — these have completely different implications for brand monitoring or research. Using `j-hartmann/emotion-english-distilroberta-base` alongside the sentiment model gives that resolution without conflating the two signals.

### Why Docker?

Local environments introduce inconsistencies in model cache paths, Python versions, and dependency resolution. Docker ensures the app runs identically in development and production. The Dockerfile:

- Uses `python:3.9-slim` to keep image size down
- Mounts a persistent model cache so models don’t re-download on each container restart
- Runs as a non-root user
- Includes a health check on the Streamlit endpoint

### Graceful Degradation

The pipeline doesn’t crash if a model fails. Every analysis call has a fallback:

```python
def analyze_with_fallback(self, text):
    try:
        return self.sentiment_analyzer.analyze(text)
    except Exception as e:
        return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'emoji': '😐', 'error': str(e)}
```

Users get partial results. Errors surface in logs without breaking the session.

### Comment Batching

Analyzing 300+ comments one-by-one causes memory spikes. Comments are processed in batches of 50, with GPU cache cleared between batches when available:

```python
def process_comments_in_batches(comments, batch_size=50):
    results = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        results.extend(self.analyze_comment_batch(batch))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results
```

-----

## Pipeline

```
URL Input
   │
   ▼
Platform Detection (Twitter / Reddit)
   │
   ▼
Content Extraction (text, metadata, engagement metrics)
   │
   ▼
Text Cleaning (URLs removed, repeated chars normalized, entities extracted)
   │
   ▼
Parallel AI Analysis
   ├── Sentiment (RoBERTa)  →  POSITIVE / NEGATIVE / NEUTRAL + confidence
   └── Emotion (DistilRoBERTa)  →  dominant emotion + full distribution
   │
   ▼
Comment Analysis (Reddit)
   ├── Per-comment sentiment + emotion
   ├── Sentiment timeline
   ├── Controversy score
   └── Theme clustering (TF-IDF + KMeans)
   │
   ▼
Visualization (Plotly)
   ├── Sentiment confidence gauge
   ├── Emotion radar chart
   ├── Engagement metrics bar chart
   └── Comment sentiment timeline
   │
   ▼
Final Report
```

-----

## Models Used

|Model                                             |Task               |Why                                               |
|--------------------------------------------------|-------------------|--------------------------------------------------|
|`cardiffnlp/twitter-roberta-base-sentiment-latest`|Sentiment          |Trained on tweets, handles slang and negation     |
|`j-hartmann/emotion-english-distilroberta-base`   |Emotion (7 classes)|Distilled for speed, strong on English social text|

-----

## Running Locally

```bash
git clone https://github.com/yourusername/social-analyzer-pro
cd social-analyzer
pip install -r requirements.txt
streamlit run app/main.py
```

**With Docker:**

```bash
docker build -t social-analyzer .
docker run -p 8501:8501 social-analyzer
```

App runs at `http://localhost:8501`

-----

## Stack

- **Backend:** Python, HuggingFace Transformers, scikit-learn
- **Frontend:** Streamlit, Plotly
- **Models:** RoBERTa, DistilRoBERTa
- **Infrastructure:** Docker
- **APIs:** Twitter API v2, Reddit PRAW

-----

## What I’d Do Differently

- Replace Streamlit with a React frontend + FastAPI backend for better separation and scalability
- Add async processing so the UI doesn’t block during model inference
- Store results in a lightweight DB (SQLite → Postgres) instead of session state, enabling analysis history
- Add multi-language support using `cardiffnlp/twitter-xlm-roberta-base-sentiment`
