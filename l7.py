import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("diabetes.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "Ridge": GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": [0.1, 1, 10]}, cv=5
    ),
    "Lasso": GridSearchCV(
        make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
        {"lasso__alpha": [0.01, 0.1, 1]}, cv=5
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n{name} Regression")
    if name != "Linear":
        print("Best Parameters:", model.best_params_)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))