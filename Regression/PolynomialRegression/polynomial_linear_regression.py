import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polinomial_regression.csv",sep=";")

y = df.araba_max_hiz.values.reshape(-1,1)

x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_hiz")
plt.xlabel("araba_fiyat")

# linear regression => y = b0+b1*x
# multiple linear regression y = b0 + b1*x + b2*y


# linear regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x,y)
# predict
y_head = lr.predict(x)

plt.plot(x,y_head,color ="red",label = "linear")
# polynomial regression y = b0 + b1*x + b2*y + b3*x^3 + .... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)
# fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color="black",label = "Polynomial")
plt.legend()
plt.show()






