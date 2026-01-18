import re
from collections import Counter

def extract_keywords(texts):
    stopwords = set(["the","and","is","to","in","of","for","a","an","on","with","this","that","it","as"])

    words = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z ]+", "", t.lower())
        for w in t.split():
            if w not in stopwords and len(w) > 3:
                words.append(w)

    return Counter(words).most_common(15)
