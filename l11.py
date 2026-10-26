import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for layers in [(10,), (50,), (50, 30), (100, 50)]:
    model = MLPClassifier(hidden_layer_sizes=layers,
                          max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(layers, "Accuracy:", accuracy_score(y_test, pred))
