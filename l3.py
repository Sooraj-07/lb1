import pandas as pd
from sklearn.datasets import fetch_openml

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


'''
train = fetch_openml('Fashion-MNIST', version=1, as_frame=True,parser='liac-arff')

X = train.data / 255.0
y = train.target.astype(str).astype(int)

# Split training and testing data
X_train = X.iloc[:5000]
y_train = y.iloc[:5000]

X_test = X.iloc[6000:6500]
y_test = y.iloc[6000:6500]
'''