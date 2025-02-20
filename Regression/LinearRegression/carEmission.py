import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('carEmissionDataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

linear_regression = LinearRegression()

linear_regression.fit(X, y) 

predictionsOfYears = [[18],[19],[20],[21]]

predictionValues = linear_regression.predict(predictionsOfYears)
print(str(predictionsOfYears) + " years: " + str(predictionValues))


plt.scatter(X, y, color = 'blue')
plt.scatter(predictionsOfYears, predictionValues, color = 'red')
plt.plot(X, linear_regression.predict(X), color = 'green')
plt.title('Car Emission Prediction')
plt.xlabel('Year')
plt.ylabel('Emission')
plt.legend(['Prediction', 'Actual Data', 'Prediction Data'])
plt.show()