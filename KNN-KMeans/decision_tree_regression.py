import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision_tree.csv", sep=";",header = None)
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
# decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() #randomstate = 0
tree_reg.fit(x,y)
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict(x_)
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()