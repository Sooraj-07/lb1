import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data[:, :2]
y = (iris.target == 0).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = SVC(kernel='linear', C=1)
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
ax = plt.gca()
xlim, ylim = ax.get_xlim(), ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
grid = np.c_[XX.ravel(), YY.ravel()]
Z = model.decision_function(grid).reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[-1, 0, 1],
           colors='black', linestyles=['--', '-', '--'])
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('Linear SVM Decision Boundary and Margins')
plt.show()