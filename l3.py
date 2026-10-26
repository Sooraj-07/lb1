import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('fashion_train.csv')
test = pd.read_csv('fashion_test.csv')

X_train = train.drop('label', axis=1) / 255.0
y_train = train['label']

X_test = test.drop('label', axis=1) / 255.0
y_test = test['label']

for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print("K =", k, "Accuracy =", accuracy)
