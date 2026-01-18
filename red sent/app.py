# app.py
import os
import glob
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# visualization.py must provide the replacement functions
from visualization import plot_sentiment_distribution, plot_upvote_distribution
from text_processing import extract_keywords

# optional lottie
try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

# ----------------------------------------------
# Config / paths
# ----------------------------------------------
# update this path if your CSV is located elsewhere
DATA_PATH = r"C:\Users\TUSHAR\Desktop\reddit sentiment analysis main\kaggle_RC_2019-05.csv"
CHUNK_SIZE = 200_000

# ----------------------------------------------
# Download VADER
# ----------------------------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ----------------------------------------------
# UI base styling
# ----------------------------------------------
css_path = Path("styles/custom.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background:#0d0d0d; color:white; }
    h2, h3 { text-shadow: 0px 0px 8px rgba(255,69,0,0.6); }
    div[data-testid="metric-container"] {
        background:#1a1a1a;
        border-radius:16px;
        padding:18px;
        border:1px solid #ff4500;
        box-shadow:0 0 12px rgba(255,69,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Reddit Insight 🔥", page_icon="🔥", layout="wide")
st.title("🔥 Reddit Sentiment & Distribution Analysis")


# small toast guard
try:
    st.toast("Enter subreddit & topic — insights ready!", icon="🔥")
except Exception:
    pass

# ----------------------------------------------
# Helper: Load Lottie Animation
# ----------------------------------------------
def load_lottie(url):
    if st_lottie is None:
        return None
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ----------------------------------------------
# Cached helper: top subreddits
# ----------------------------------------------
@st.cache_data(show_spinner=False)
def get_top_subreddits(data_path: str = DATA_PATH, top_n: int = 40, chunksize: int = CHUNK_SIZE):
    """
    Return top_n subreddit names (lowercased) from the CSV or folder.
    Reads only the subreddit column in chunks for performance.
    """
    ctr = Counter()
    filenames = []
    try:
        if os.path.isdir(data_path):
            filenames = sorted(glob.glob(os.path.join(data_path, "*.csv")))
            if not filenames:
                return []
        else:
            if not os.path.exists(data_path):
                return []
            filenames = [data_path]
    except Exception:
        return []

    for f in filenames:
        try:
            for chunk in pd.read_csv(f, usecols=["subreddit"], chunksize=chunksize, low_memory=False):
                if "subreddit" in chunk.columns:
                    ctr.update(chunk["subreddit"].astype(str).str.lower().tolist())
        except Exception:
            # fallback: read whole chunk and try to find subreddit column name
            try:
                for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                    possible = next((c for c in chunk.columns if c.lower() in ("subreddit", "subreddit_name")), None)
                    if possible:
                        ctr.update(chunk[possible].astype(str).str.lower().tolist())
            except Exception:
                continue

    return [s for s, _ in ctr.most_common(top_n)]

# ----------------------------------------------
# Sidebar: dropdown of top 40 subreddits + controls
# ----------------------------------------------
top_subs = get_top_subreddits()

if top_subs:
    # default select 'gameofthrones' if present else first
    default_idx = 0
    if "gameofthrones" in top_subs:
        default_idx = top_subs.index("gameofthrones")
    subreddit = st.sidebar.selectbox("Choose a subreddit (top 40)", options=top_subs, index=default_idx)
else:
    # fallback to free text input if nothing found
    subreddit = st.sidebar.text_input("Subreddit", "technology")

query = st.sidebar.text_input("Keyword / Topic", "AI")
limit = st.sidebar.slider("Number of Posts", 10, 200, 50)
start = st.sidebar.button("Start Analysis")

# ----------------------------------------------
# Offline CSV loader (robust)
# ----------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_reddit_posts(subreddit: str, query: str, limit: int = 50, data_path: str = DATA_PATH):
    """
    Read CSV(s) in chunks and return up to 'limit' post-like dicts:
    {title, selftext, ups, created_utc, url}
    Works with files missing id/created fields.
    """
    q_lower = (query or "").strip().lower()
    subreddit_lower = (subreddit or "").strip().lower()

    def csv_chunk_iterator(path):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.csv")))
            if not files:
                raise FileNotFoundError(f"No CSV files found in folder: {path}")
            for f in files:
                for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
                    yield chunk
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")
            for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
                yield chunk

    results = []
    collected = 0

    try:
        iterator = csv_chunk_iterator(data_path)
    except Exception as e:
        st.error(f"❌ Error opening dataset: {e}")
        return []

    for chunk in iterator:
        # detect columns
        body_col = next((c for c in chunk.columns if c.lower() in ("body", "text", "selftext", "comment_body")), None)
        sub_col = next((c for c in chunk.columns if c.lower() in ("subreddit", "subreddit_name")), None)
        score_col = next((c for c in chunk.columns if c.lower() in ("score", "ups", "upvotes")), None)
        id_col = next((c for c in chunk.columns if c.lower() in ("id", "comment_id", "post_id", "link_id")), None)
        created_col = next((c for c in chunk.columns if c.lower() in ("created_utc", "created", "timestamp", "time", "created_utc_ms")), None)

        if body_col is None:
            st.error("Dataset missing a text column (expected 'body' or similar). Found columns: " + ", ".join(chunk.columns))
            return []

        # normalize columns
        chunk[body_col] = chunk[body_col].astype(str)
        if sub_col:
            chunk[sub_col] = chunk[sub_col].astype(str)
        if score_col:
            chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0).astype(int)
        else:
            chunk["_score_tmp"] = 0
            score_col = "_score_tmp"

        # exact subreddit filter first
        if subreddit_lower and sub_col:
            chunk = chunk[chunk[sub_col].str.lower() == subreddit_lower]

        # query filter
        if q_lower:
            chunk = chunk[chunk[body_col].str.contains(q_lower, case=False, na=False)]

        if chunk.empty:
            continue

        # order: by created if present, else by score
        if created_col and created_col in chunk.columns:
            def _norm_created(v):
                try:
                    v_int = int(v)
                    if v_int > 10**12:
                        return v_int // 1000
                    return v_int
                except Exception:
                    try:
                        ts = pd.to_datetime(v, utc=True)
                        return int(ts.timestamp())
                    except Exception:
                        return 0
            chunk["_created_secs"] = chunk[created_col].apply(_norm_created)
            chunk = chunk.sort_values("_created_secs", ascending=False)
        else:
            chunk = chunk.sort_values(score_col, ascending=False)

        for _, row in chunk.iterrows():
            raw_id = None
            if id_col and id_col in row.index and pd.notna(row[id_col]):
                raw_id = str(row[id_col])
            else:
                h = hashlib.md5()
                h.update((str(row.get(sub_col, "")) + "|" + row[body_col][:120] + "|" + str(row.get(score_col, 0))).encode("utf-8"))
                raw_id = h.hexdigest()

            created_val = int(row.get("_created_secs", 0)) if "_created_secs" in row.index else 0

            results.append({
                "title": f"Comment on r/{row.get(sub_col,'')}" if sub_col else "Comment",
                "selftext": row[body_col] or "",
                "ups": int(row.get(score_col, 0) or 0),
                "created_utc": created_val,
                "url": f"https://reddit.com/comment/{raw_id}"
            })
            collected += 1
            if collected >= limit:
                return results

    # relaxed substring match on subreddit if nothing found and user provided subreddit
    if not results and subreddit_lower and top_subs:
        try:
            iterator = csv_chunk_iterator(data_path)
        except Exception as e:
            st.error(f"❌ Error reopening dataset: {e}")
            return []
        collected = 0
        for chunk in iterator:
            body_col = next((c for c in chunk.columns if c.lower() in ("body", "text", "selftext", "comment_body")), None)
            sub_col = next((c for c in chunk.columns if c.lower() in ("subreddit", "subreddit_name")), None)
            score_col = next((c for c in chunk.columns if c.lower() in ("score", "ups", "upvotes")), None)
            id_col = next((c for c in chunk.columns if c.lower() in ("id", "comment_id", "post_id", "link_id")), None)
            if body_col is None:
                continue

            chunk[body_col] = chunk[body_col].astype(str)
            if sub_col:
                chunk[sub_col] = chunk[sub_col].astype(str)
            if score_col:
                chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0).astype(int)
            else:
                chunk["_score_tmp"] = 0
                score_col = "_score_tmp"

            mask_sub = chunk[sub_col].str.lower().str.contains(subreddit_lower, na=False) if sub_col else pd.Series([False] * len(chunk))
            chunk = chunk[mask_sub]
            if q_lower:
                chunk = chunk[chunk[body_col].str.contains(q_lower, case=False, na=False)]
            if chunk.empty:
                continue

            chunk = chunk.sort_values(score_col, ascending=False)
            for _, row in chunk.iterrows():
                h = hashlib.md5()
                h.update((str(row.get(sub_col, "")) + "|" + row[body_col][:120] + "|" + str(row.get(score_col, 0))).encode("utf-8"))
                raw_id = h.hexdigest()
                results.append({
                    "title": f"Comment on r/{row.get(sub_col,'')}" if sub_col else "Comment",
                    "selftext": row[body_col] or "",
                    "ups": int(row.get(score_col, 0) or 0),
                    "created_utc": 0,
                    "url": f"https://reddit.com/comment/{raw_id}"
                })
                collected += 1
                if collected >= limit:
                    return results

    # final query-only fallback
    if not results and q_lower:
        try:
            iterator = csv_chunk_iterator(data_path)
        except Exception as e:
            st.error(f"❌ Error reopening dataset: {e}")
            return []
        collected = 0
        for chunk in iterator:
            body_col = next((c for c in chunk.columns if c.lower() in ("body", "text", "selftext", "comment_body")), None)
            score_col = next((c for c in chunk.columns if c.lower() in ("score", "ups", "upvotes")), None)
            id_col = next((c for c in chunk.columns if c.lower() in ("id", "comment_id", "post_id", "link_id")), None)
            if body_col is None:
                continue
            chunk[body_col] = chunk[body_col].astype(str)
            if score_col:
                chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0).astype(int)
            else:
                chunk["_score_tmp"] = 0
                score_col = "_score_tmp"
            chunk = chunk[chunk[body_col].str.contains(q_lower, case=False, na=False)]
            if chunk.empty:
                continue
            chunk = chunk.sort_values(score_col, ascending=False)
            for _, row in chunk.iterrows():
                h = hashlib.md5()
                h.update((str(row.get(None, "")) + "|" + row[body_col][:120] + "|" + str(row.get(score_col, 0))).encode("utf-8"))
                raw_id = h.hexdigest()
                results.append({
                    "title": f"Comment on r/{row.get(sub_col,'')}" if sub_col else "Comment",
                    "selftext": row[body_col] or "",
                    "ups": int(row.get(score_col, 0) or 0),
                    "created_utc": 0,
                    "url": f"https://reddit.com/comment/{raw_id}"
                })
                collected += 1
                if collected >= limit:
                    return results

    if not results:
        st.warning("⚠ No matching comments found in the local dataset. Try a more common query or a different subreddit.")
    return results

# ----------------------------------------------
# Keyword Plotting (Dark Themed)  (unchanged)
# ----------------------------------------------
def plot_keyword_bar_dark(keywords):
    if not keywords:
        st.info("No keywords to display.")
        return
    words, counts = zip(*keywords)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title("Top Keywords")
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white")
    st.pyplot(fig)

# ----------------------------------------------
# RUN ANALYSIS PIPELINE
# ----------------------------------------------
if start:
    with st.container():
        animation = load_lottie("https://assets5.lottiefiles.com/packages/lf20_touohxv0.json")
        if animation and st_lottie is not None:
            st_lottie(animation, height=150)

    posts = fetch_reddit_posts(subreddit, query, limit)
    if not posts:
        st.stop()

    df = pd.DataFrame(posts)
    # created_utc may be 0 if dataset has no timestamps; guard conversion
    if "created_utc" in df.columns and df["created_utc"].any():
        df["created"] = df["created_utc"].apply(lambda x: datetime.utcfromtimestamp(x) if x else pd.NaT)
    else:
        df["created"] = pd.NaT

    # text, sentiment & labels
    df["text"] = df["title"] + " " + df["selftext"]
    df["compound"] = df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["label"] = df["compound"].apply(lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"))

    # create date only when created exists
    if df["created"].notna().any():
        df["date"] = pd.to_datetime(df["created"]).dt.date
    else:
        df["date"] = pd.NaT

    # ensure ups exists
    if "ups" not in df.columns and "score" in df.columns:
        df["ups"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    elif "ups" in df.columns:
        df["ups"] = pd.to_numeric(df["ups"], errors="coerce").fillna(0).astype(int)
    else:
        df["ups"] = 0

    # ------------------------
    # Create tabs
    # ------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Sentiment Distribution",
        "👍 Upvote Distribution",
        "🔤 Keywords",
        "🗂 Posts & Scores"
    ])

    # ------------------------
    # Overview tab
    # ------------------------
    with tab1:
        st.subheader("🧠 Overview of Analysis")
        with st.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📥 Total Posts", len(df))
            m2.metric("😊 Positive", int((df["label"] == "Positive").sum()))
            m3.metric("😡 Negative", int((df["label"] == "Negative").sum()))
            m4.metric("😐 Neutral", int((df["label"] == "Neutral").sum()))

        st.markdown("---")
        st.markdown("### ⭐ Engagement Insights")
        st.write(f"Average Upvotes: **{round(df['ups'].mean(),2)}⬆**")

        if not df.empty:
            top_post = df.loc[df["ups"].idxmax()]
            st.write(f"Most Upvoted: **{top_post['title']}** ({top_post['ups']}⬆)")
        else:
            top_post = None

        if not df.empty:
            tone = "Mostly **Positive 😊**" if (df["label"] == "Positive").sum() > (df["label"] == "Negative").sum() else "Mostly **Negative 😡**"
            st.write(f"**Sentiment Tone:** {tone}")

        st.markdown("---")
        st.markdown("### 📌 Quick Summary")
        with st.container():
            st.write(f"""
            🔥 **Topic:** {query}  
            💬 **Subreddit:** r/{subreddit}  
            📊 **Dominant Sentiment:** {df['label'].mode()[0] if not df.empty else 'N/A'}  
            🏆 **Most Upvoted Post:** {top_post['title'] if top_post is not None else 'N/A'}  
            """)

        st.markdown("---")
        st.markdown("### 😊 Sentiment Breakdown")
        if not df.empty:
            sentiment_counts = df["label"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
            fig.patch.set_facecolor("#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            plt.setp(ax.texts, color="white")
            st.pyplot(fig)
        else:
            st.info("No data to display.")

    # ------------------------
    # Sentiment Distribution tab
    # ------------------------
    with tab2:
        st.subheader("📈 Sentiment Distribution")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        plot_sentiment_distribution(df)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------
    # Upvote Distribution tab
    # ------------------------
    with tab3:
        st.subheader("👍 Upvote Distribution")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        plot_upvote_distribution(df)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------
    # Keywords tab
    # ------------------------
    with tab4:
        st.subheader("🔤 Most Frequent Keywords")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        keywords = extract_keywords(df["text"].tolist())
        plot_keyword_bar_dark(keywords)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------
    # Posts & Scores tab
    # ------------------------
    with tab5:
        st.subheader("🗂 Posts & Scores")
        if not df.empty:
            st.dataframe(df[["created", "title", "label", "compound", "ups", "url"]], height=450, use_container_width=True)
            st.markdown("---")
            st.markdown("### 🏆 Most Upvoted Post")
            st.write(f"**{top_post['title']}** — {top_post['ups']}⬆")
            st.write(top_post['url'])
        else:
            st.info("No posts to display.")
else:
    st.info("👈 Select a subreddit & topic, then hit **Start Analysis** 🔥")
