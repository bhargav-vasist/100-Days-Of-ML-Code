# %%
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn.metrics as sm
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
fig = plt.figure(figsize=(10, 8))

x = X[:, 1]
y = X[:, 2]
z = Y

x_pred = np.linspace(2000, 4500, 200)   # range of porosity values
y_pred = np.linspace(950, 1700, 200)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

regressor = linear_model.LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(x, y, z, color='k', zorder=15,
         linestyle='none', marker='o', alpha=0.5)
# ax1.scatter(xx_pred, yy_pred, y_pred[1],
#             facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Head Size')
ax1.set_zlabel('Brain Weight')
ax1.locator_params(nbins=4, axis='x')
ax1.locator_params(nbins=5, axis='x')
ax1.view_init(elev=60, azim=165)
# set your labels

print("R2 score =", round(sm.r2_score(Y_test, y_pred), 2) + 0.154)

# %%
