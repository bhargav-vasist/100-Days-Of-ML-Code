# %%
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../datasets/linear-regression/data.csv')
X = dataset.iloc[:, 2].values
Y = dataset.iloc[:, 3].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/4, random_state=0)

X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
# %%
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# %%
plt.scatter(X_train, Y_train, color='red')
plt.title = "Predict Brain Mass for Brain Head Size"
plt.xlabel = "Brain head size in Cm^3"
plt.ylabel = "Brain weight in grams"
plt.plot(X_train, regressor.predict(X_train), color='blue')


# %%
plt.scatter(X_test, Y_test, color='red')
plt.title = "Predict Brain Mass for Brain Head Size"
plt.xlabel = "Brain head size in Cm^3"
plt.ylabel = "Brain weight in grams"
plt.plot(X_test, regressor.predict(X_test), color='blue')

print("R2 score =", round(sm.r2_score(Y_test, Y_pred), 2))
# %%
