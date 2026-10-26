import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("20newsgroups.csv")
texts = df["text"].dropna()

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)
counts = np.asarray(X.sum(axis=0)).ravel()

mle = counts / counts.sum()

for alpha in [0.1, 1, 10]:
    map_est = (counts + alpha - 1) / (
        counts.sum() + len(counts) * (alpha - 1)
    ) if alpha >= 1 else (counts + alpha) / (
        counts.sum() + len(counts) * alpha
    )

    print(f"\nDirichlet Prior alpha = {alpha}")
    print("First 10 probabilities:", map_est[:10])

print("\nMLE probabilities:", mle[:10])
