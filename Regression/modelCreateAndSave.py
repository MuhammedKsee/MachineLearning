import pickle
from sklearn.linear_model import LinearRegression

x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)

# Model Saving
pickle.dump(linear_regression_model, open("model.pkl", "wb"))

# Model Loading
model = pickle.load(open("model.pkl", "rb"))

 