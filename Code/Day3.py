# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/linear-regression/data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,  3].values

# %%
ct = ColumnTransformer(
    [("Gender", OneHotEncoder(), [0])], remainder="passthrough")
X = ct.fit_transform(X)

X = X[:, 1:]

# %%
labelEnc = LabelEncoder()
X[:, 1] = labelEnc.fit_transform(X[:, 1])


# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# %%
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# %%
y_pred = regressor.predict(X_test)

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train, c='blue', marker='o')

# set your labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
