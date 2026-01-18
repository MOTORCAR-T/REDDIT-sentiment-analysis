import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER if needed
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

def process_posts(posts):
    df = pd.DataFrame(posts)
    df["created"] = df["created_utc"].apply(lambda x: datetime.utcfromtimestamp(x))
    df["text"] = df["title"] + " " + df["selftext"]

    # Sentiment scoring
    df["compound"] = df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])

    df["label"] = df["compound"].apply(
        lambda c: "POSITIVE" if c >= 0.05 else ("NEGATIVE" if c <= -0.05 else "NEUTRAL")
    )

    df["date"] = df["created"].dt.date

    return df
