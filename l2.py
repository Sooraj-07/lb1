import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load local CSV
df = pd.read_csv('20newsgroups.csv')
df = df.dropna(subset=['text', 'target'])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42
)

cv = CountVectorizer(stop_words='english', max_features=5000)
X_train = cv.fit_transform(X_train_text)
X_test = cv.transform(X_test_text)

X_train_bin = (X_train > 0).astype(int)
X_test_bin = (X_test > 0).astype(int)

models = [
    ('Multinomial NB', MultinomialNB(), X_train, X_test),
    ('Bernoulli NB', BernoulliNB(), X_train_bin, X_test_bin)
]

for name, model, Xtr, Xte in models:
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print('\n', name)
    print('Accuracy:', accuracy_score(y_test, pred))
    print('F1-score:', f1_score(y_test, pred, average='macro'))
