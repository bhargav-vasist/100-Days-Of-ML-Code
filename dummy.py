# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# x_0=(0.1, 0.2), x_1=(-0.2, 0.3), x_3=(1.2, 1.3), x_4=(0.9, 1.5)
X = np.array([[0.1, 0.2], [-0.2, 0.3]])
y = np.array([[1.2, 1.3], [0.9, 1.5]])
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
LinearDiscriminantAnalysis()
print(clf.predict([[-0.8, -1]]))

# %%
