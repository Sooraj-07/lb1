import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

train = train.sample(5000, random_state=42)
test = test.sample(1000, random_state=42)

X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values

for act in ['logistic', 'relu', 'tanh']:
    model = MLPClassifier(hidden_layer_sizes=(100,),
                          activation=act, max_iter=20,
                          random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    print(act, "Accuracy:", accuracy_score(y_test, pred))
    print("Training time:", round(elapsed, 2), "seconds")
    print("Iterations:", model.n_iter_)
