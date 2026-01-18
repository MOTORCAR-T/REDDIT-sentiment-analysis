# data_loader.py  (paste this or replace fetch_reddit_posts in your modules)
import os
import glob
import pandas as pd
import streamlit as st
import hashlib

# Set this to your extracted CSV path (single file) or folder of CSVs
DATA_PATH = r"C:\Users\TUSHAR\Desktop\reddit sentiment analysis main\kaggle_RC_2019-05.csv"
CHUNK_SIZE = 200_000

@st.cache_data(show_spinner=False)
def fetch_reddit_posts(subreddit: str, query: str, limit: int = 50, data_path: str = DATA_PATH):
    """
    Loader adapted to CSV with columns: subreddit, body, controversiality, score
    - Returns list of dicts: title, selftext, ups, created_utc, url
    - created_utc will be 0 (falsy) since file doesn't include timestamps.
    """
    q_lower = (query or "").strip().lower()
    subreddit_lower = (subreddit or "").strip().lower()

    # simple iterator supporting single CSV or a folder of CSVs
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
        st.error(f"Error opening dataset: {e}")
        return []

    # 1) preferred pass: exact subreddit match (if provided) + query (if provided)
    for chunk in iterator:
        # normalize possible columns
        cols = [c.lower() for c in chunk.columns]
        # find text and subreddit column names (fallback to common names)
        body_col = next((c for c in chunk.columns if c.lower() in ("body","text","selftext","comment_body")), None)
        sub_col = next((c for c in chunk.columns if c.lower() in ("subreddit","subreddit_name")), None)
        score_col = next((c for c in chunk.columns if c.lower() in ("score","ups","upvotes")), None)
        id_col = next((c for c in chunk.columns if c.lower() in ("id","comment_id","post_id","link_id")), None)

        # ensure body exists
        if body_col is None:
            st.error("Dataset missing a text column (expected 'body' or similar). Found columns: " + ", ".join(chunk.columns))
            return []

        # coerce types
        chunk[body_col] = chunk[body_col].astype(str)
        if sub_col:
            chunk[sub_col] = chunk[sub_col].astype(str)
        if score_col:
            chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0).astype(int)
        else:
            chunk["_score"] = 0
            score_col = "_score"

        # exact subreddit filter (if provided)
        if subreddit_lower and sub_col:
            chunk = chunk[chunk[sub_col].str.lower() == subreddit_lower]

        # query filter (if provided)
        if q_lower:
            chunk = chunk[chunk[body_col].str.contains(q_lower, case=False, na=False)]

        if chunk.empty:
            continue

        # sort by score as fallback for ordering (desc)
        chunk = chunk.sort_values(score_col, ascending=False)

        for idx, row in chunk.iterrows():
            # build a deterministic synthetic id if id_col missing
            raw_id = None
            if id_col and id_col in row.index and pd.notna(row[id_col]):
                raw_id = str(row[id_col])
            else:
                # create hash from subreddit+body+score to make URL/id stable
                h = hashlib.md5()
                h.update((str(row.get(sub_col,"")) + "|" + row[body_col][:120] + "|" + str(row.get(score_col,0))).encode("utf-8"))
                raw_id = h.hexdigest()

            results.append({
                "title": f"Comment on r/{row.get(sub_col,'')}" if sub_col else "Comment",
                "selftext": row[body_col] or "",
                "ups": int(row.get(score_col, 0) or 0),
                "created_utc": 0,   # no timestamp in this CSV; downstream code treats falsy as NaT
                "url": f"https://reddit.com/comment/{raw_id}"
            })
            collected += 1
            if collected >= limit:
                return results

    # 2) If nothing found and subreddit was provided, try substring match (relaxed)
    if not results and subreddit_lower:
        try:
            iterator = csv_chunk_iterator(data_path)
        except Exception as e:
            st.error(f"Error reopening dataset: {e}")
            return []

        for chunk in iterator:
            body_col = next((c for c in chunk.columns if c.lower() in ("body","text","selftext","comment_body")), None)
            sub_col = next((c for c in chunk.columns if c.lower() in ("subreddit","subreddit_name")), None)
            score_col = next((c for c in chunk.columns if c.lower() in ("score","ups","upvotes")), None)
            id_col = next((c for c in chunk.columns if c.lower() in ("id","comment_id","post_id","link_id")), None)

            if body_col is None:
                continue

            chunk[body_col] = chunk[body_col].astype(str)
            if sub_col:
                chunk[sub_col] = chunk[sub_col].astype(str)
            if score_col:
                chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0).astype(int)
            else:
                chunk["_score"] = 0
                score_col = "_score"

            # substring match on subreddit
            if sub_col:
                chunk = chunk[chunk[sub_col].str.lower().str.contains(subreddit_lower, na=False)]

            # query filter
            if q_lower:
                chunk = chunk[chunk[body_col].str.contains(q_lower, case=False, na=False)]

            if chunk.empty:
                continue

            chunk = chunk.sort_values(score_col, ascending=False)
            for idx, row in chunk.iterrows():
                h = hashlib.md5()
                h.update((str(row.get(sub_col,"")) + "|" + row[body_col][:120] + "|" + str(row.get(score_col,0))).encode("utf-8"))
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

    # 3) Final fallback: query-only across all subreddits
    if not results and q_lower:
        try:
            iterator = csv_chunk_iterator(data_path)
        except Exception as e:
            st.error(f"Error reopening dataset: {e}")
            return []
        for chunk in iterator:
            body_col = next((c for c in chunk.columns if c.lower() in ("body","text","selftext","comment_body")), None)
            sub_col = next((c for c in chunk.columns if c.lower() in ("subreddit","subreddit_name")), None)
            score_col = next((c for c in chunk.columns if c.lower() in ("score","ups","upvotes")), None)
            id_col = next((c for c in chunk.columns if c.lower() in ("id","comment_id","post_id","link_id")), None)

            if body_col is None:
                continue
            chunk[body_col] = chunk[body_col].astype(str)
            if score_col:
                chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0).astype(int)
            else:
                chunk["_score"] = 0
                score_col = "_score"

            chunk = chunk[chunk[body_col].str.contains(q_lower, case=False, na=False)]
            if chunk.empty:
                continue

            chunk = chunk.sort_values(score_col, ascending=False)
            for idx, row in chunk.iterrows():
                h = hashlib.md5()
                h.update((str(row.get(sub_col,"")) + "|" + row[body_col][:120] + "|" + str(row.get(score_col,0))).encode("utf-8"))
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
        st.warning("⚠ No matching comments found in the dataset. Try a more common query or a different subreddit.")
    return results

