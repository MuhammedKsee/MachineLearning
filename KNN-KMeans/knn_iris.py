import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
data = pd.DataFrame(data.data, columns=data.feature_names)
data = data.iloc[:, 1:-1].values
kmodel = KMeans(n_clusters=5, init="k-means++", random_state=0)
kmodel.fit(data)
tahmin = kmodel.predict(data)

merkezler = kmodel.cluster_centers_
print(merkezler)

plt.scatter(data[tahmin == 0, 0], data[tahmin == 0, 1], s=50, color="red")
plt.scatter(data[tahmin == 1, 0], data[tahmin == 1, 1], s=50, color="blue")
plt.scatter(data[tahmin == 2, 0], data[tahmin == 2, 1], s=50, color="green")
plt.scatter(data[tahmin == 3, 0], data[tahmin == 3, 1], s=50, color="purple")
plt.scatter(data[tahmin == 4, 0], data[tahmin == 4, 1], s=50, color="black")
plt.title("K-Means Iris Sınıflandırması")
plt.savefig("grafik.png")
