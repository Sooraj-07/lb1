import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('california_housing.csv').dropna()
df['AveRooms'] = df['total_rooms'] / df['households']
X = df[['AveRooms']].values
y = df['median_house_value'].values / 100000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

m, s = X_train.mean(), X_train.std()
a, t = (X_train[:, 0]-m)/s, (X_test[:, 0]-m)/s
w, b = 0, y_train.mean()

for _ in range(1000):
    p = w*a + b
    w -= 0.01*np.mean((p-y_train)*a)
    b -= 0.01*np.mean(p-y_train)

gd = w*t + b

A = np.c_[np.ones(len(X_train)), X_train]
theta = np.linalg.pinv(A) @ y_train
ne = np.c_[np.ones(len(X_test)), X_test] @ theta

for name, pred in [('Gradient Descent', gd), ('Normal Equation', ne)]:
    print(name, 'MSE:', mean_squared_error(y_test, pred),
          'R2:', r2_score(y_test, pred))

i = np.argsort(X_test[:, 0])
plt.scatter(X_test[:, 0], y_test, s=5)
plt.plot(X_test[i, 0], ne[i], 'r')
plt.xlabel('Average Rooms')
plt.ylabel('House Value (hundred thousand dollars)')
plt.show()
