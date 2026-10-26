import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

train = train.sample(10000, random_state=42)
test = test.sample(2000, random_state=42)

X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values

for kernel in ['linear', 'poly', 'rbf']:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(kernel, "Accuracy:", accuracy_score(y_test, pred))
