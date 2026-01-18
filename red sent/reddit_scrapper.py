import pandas as pd
import streamlit as st

DATA_PATH = "/mnt/data/kaggle_RC_2019-05.csv.zip"
CHUNK_SIZE = 200_000

@st.cache_data(show_spinner=False)
def fetch_reddit_posts(subreddit, query, limit=50):
    """
    Offline replacement for Pushshift/Reddit API using Kaggle Reddit comments dataset.
    Returns list of dicts shaped like submissions for backward compatibility.
    """
    try:
        df_iter = pd.read_csv(
            DATA_PATH,
            compression="zip",
            chunksize=CHUNK_SIZE,
            usecols=["id", "body", "subreddit", "created_utc", "score"],
            low_memory=False
        )
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return []

    results = []
    q_lower = (query or "").strip().lower()
    subreddit_lower = (subreddit or "").strip().lower()

    for chunk in df_iter:
        # Clean and normalize
        chunk["subreddit"] = chunk["subreddit"].astype(str)
        chunk["body"] = chunk["body"].astype(str)

        # Filter subreddit
        if subreddit_lower:
            chunk = chunk[chunk["subreddit"].str.lower() == subreddit_lower]
            if chunk.empty:
                continue

        # Filter by keyword
        if q_lower:
            chunk = chunk[chunk["body"].str.contains(q_lower, case=False, na=False)]
            if chunk.empty:
                continue

        # Sort by newest
        chunk = chunk.sort_values("created_utc", ascending=False)

        # Build post-like objects
        for _, row in chunk.iterrows():
            results.append({
                "title": f"Comment on r/{row['subreddit']}",
                "selftext": row["body"],
                "ups": int(row.get("score", 0)),
                "created_utc": int(row.get("created_utc", 0)),
                "url": f"https://reddit.com/comment/{row['id']}"
            })

            if len(results) >= limit:
                return results

    if not results:
        st.warning("⚠ No matching comments found in the local dataset.")
    return results
