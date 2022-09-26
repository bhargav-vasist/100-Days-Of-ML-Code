# %%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# import graphviz
# from graphviz import Source

##################################

### ML Models ###
# from sklearn.tree.export import export_text

##################################

### Metrics ###

# %%
data = pd.read_csv('../datasets/credit.csv')

# Information
data.info()
data = data.drop(['ID'], axis=1)
print(f"There are {data.duplicated().sum()} duplicate rows in the data set.")

# Remove duplicate rows.
data = data.drop_duplicates()
print("The duplicate rows were removed.")

X = data[data.columns[:-1]]
y = data['dpnm']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=25)

tr = tree.DecisionTreeClassifier(
    max_depth=3, criterion='entropy', random_state=25)

# Train the estimator.
tr.fit(X_train, y_train)

dot_data = tree.export_graphviz(
    tr, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph

tr_pred = tr.predict(X_test)

tr_matrix = confusion_matrix(y_test, tr_pred)
sns.set(font_scale=1.3)
plt.subplots(figsize=(8, 8))
sns.heatmap(tr_matrix, annot=True, cbar=False,
            cmap='twilight', linewidth=0.5, fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Decision tree')

tr_probs = tr.predict_proba(X_test)

# Keep Probabilities of the positive class only.
tr_probs = tr_probs[:, 1]

# %%
