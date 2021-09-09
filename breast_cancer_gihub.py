import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

cancer = load_breast_cancer()
print(cancer)

cancer.keys()


print(cancer['target'])
print(cancer['target_names'])
cancer['data'].shape

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()

sns.countplot(df_cancer['target'])

plt.figure(figsize=(20, 10))
sns.heatmap(df_cancer.corr(), annot = True)

#################################################

X = df_cancer.drop(['target'], axis = 1)
print(X)

y = df_cancer['target']
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

cmodel = svm.SVC(kernel='linear')

cmodel.fit(X_train, y_train)

y_pred = cmodel.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))


###### Optimized Code ######
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train) / range_train


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test) / range_test

cmodel.fit(X_train_scaled, y_train)
y_pred = cmodel.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))