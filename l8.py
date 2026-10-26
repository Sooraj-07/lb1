import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv("breast_cancer.csv")
#df = df.dropna()
#y = df["target"].map({"M": 1, "B": 0})
#X = df.drop(columns=["target"])
#X = X.select_dtypes(include="number").drop(
 #   columns=["id", "Unnamed: 32"], errors="ignore"
#)
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "MLE": make_pipeline(StandardScaler(),
                         LogisticRegression(penalty=None, max_iter=5000)),
    "MAP-L1": make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l1", solver="liblinear",
                                               C=1, max_iter=5000)),
    "MAP-L2": make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l2", C=1,
                                               max_iter=5000))
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    lr = model.named_steps["logisticregression"]

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Coefficients:", lr.coef_)
