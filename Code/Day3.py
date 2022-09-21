# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/linear-regression/data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,  3].values

onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 0:]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')

# %%
