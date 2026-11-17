import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#df = pd.read_csv('digits.csv')

#X = df.drop(columns=['target'])
data = load_digits()
X = data.data

X = StandardScaler().fit_transform(X)

K = [5, 8, 10, 12]
#K=range(2,21)
inertia = []

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Evaluation
    inertia.append(model.inertia_)
    score = silhouette_score(X, labels)

    print("\nK =", k)
    print("Inertia:", model.inertia_)
    print("Silhouette Score:", score)

# 5. Elbow Method
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
