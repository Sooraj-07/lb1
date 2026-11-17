import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("auto-mpg.csv")
df = df.dropna(subset=["mpg", "displacement"])

X = df[["displacement"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Visualization
X_plot = np.linspace(X["displacement"].min(),
                     X["displacement"].max(), 200).reshape(-1, 1)

plt.scatter(X["displacement"], y, s=10, label="Actual data")
plt.plot(X_plot, model.predict(X_plot), label="Linear Regression")

# Polynomial Regression
for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)

    print(f"\nPolynomial Degree {degree}:")
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

    y_plot = model.predict(poly.transform(X_plot))
    plt.plot(X_plot, y_plot, label=f"Degree {degree}")

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid()
plt.show()
