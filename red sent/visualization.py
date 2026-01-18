# visualization.py
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


# ================================================================
# SENTIMENT DISTRIBUTION TAB
# ================================================================
def plot_sentiment_distribution(df):
    """
    Shows sentiment breakdown + sentiment vs upvote buckets.
    Does NOT require timestamps.
    """

    if df is None or df.empty:
        st.info("No data available for sentiment analysis.")
        return

    df = df.copy()

    # Ensure 'ups'
    if "ups" not in df.columns and "score" in df.columns:
        df["ups"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    elif "ups" in df.columns:
        df["ups"] = pd.to_numeric(df["ups"], errors="coerce").fillna(0).astype(int)
    else:
        df["ups"] = 0

    # Ensure sentiment label
    if "label" not in df.columns and "compound" in df.columns:
        df["label"] = df["compound"].apply(
            lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")
        )

    if "label" not in df.columns:
        st.info("No 'label' or 'compound' column found.")
        return

    # -------------------------------
    # Pie chart + count bar chart
    # -------------------------------
    st.subheader("Overall Sentiment Breakdown")

    counts = df["label"].value_counts()
    col1, col2 = st.columns([1, 1])

    with col1:
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(counts.index, counts.values)
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    st.markdown("---")

    # -------------------------------
    # Sentiment by upvote buckets
    # -------------------------------
    st.subheader("Sentiment by Upvote Bucket")

    bins = [-1, 0, 1, 5, 10, 50, 200, np.inf]
    labels = ["0", "1", "2–4", "5–9", "10–49", "50–199", "200+"]

    df["up_bucket"] = pd.cut(df["ups"], bins=bins, labels=labels)

    pivot = pd.crosstab(df["up_bucket"], df["label"])

    if pivot.empty:
        st.info("Not enough upvote data to compute bucket analysis.")
        return

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax3)
    ax3.set_xlabel("Upvote Bucket")
    ax3.set_ylabel("Count")
    ax3.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)


# ================================================================
# UPVOTE DISTRIBUTION TAB
# ================================================================
def plot_upvote_distribution(df):
    """
    Shows histogram, boxplot, and top-N comments by upvotes.
    Works without timestamps.
    """

    if df is None or df.empty:
        st.info("No data available for upvote distribution.")
        return

    df = df.copy()

    # Ensure ups exists
    if "ups" not in df.columns and "score" in df.columns:
        df["ups"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    elif "ups" in df.columns:
        df["ups"] = pd.to_numeric(df["ups"], errors="coerce").fillna(0).astype(int)
    else:
        df["ups"] = 0

    non_zero = df["ups"].replace(0, np.nan).dropna()
    max_clip = (
        np.percentile(non_zero, 99) if len(non_zero) > 0 else df["ups"].max()
    )

    # -------------------------------
    # Upvote Histogram
    # -------------------------------
    st.subheader("Upvote Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["ups"].clip(upper=max_clip), bins=40)
        ax.set_xlabel("Upvotes")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # -------------------------------
    # Boxplot
    # -------------------------------
    with col2:
        figb, axb = plt.subplots(figsize=(4, 3))
        if len(non_zero) > 0:
            axb.boxplot(non_zero)
            axb.set_title("Upvote Boxplot (non-zero)")
        else:
            axb.text(0.5, 0.5, "No non-zero upvotes", ha='center')
        st.pyplot(figb)

    st.markdown("---")

    # -------------------------------
    # Top N comments by upvotes
    # -------------------------------
    st.subheader("Top Comments by Upvotes")

    top_n = st.slider("How many comments to show?", 3, 30, 10)

    top = df.sort_values("ups", ascending=False).head(top_n)

    for i, row in top.iterrows():
        st.markdown(f"### 🔥 {row.get('title', 'Comment')} — {int(row['ups'])}⬆")
        text = (
            row.get("selftext")
            or row.get("body")
            or "(no text available)"
        )
        st.write(text[:500] + ("..." if len(text) > 500 else ""))
        st.divider()

    # Summary stats
    st.markdown("### 📊 Upvote Summary")
    st.write(f"**Max:** {int(df['ups'].max())}  |  **Mean:** {df['ups'].mean():.2f}  |  **Median:** {df['ups'].median():.2f}")
