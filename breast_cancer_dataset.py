#importar tot el que necessitarem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

# importar el dataset del cancer de mama

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()

# Normalitzacio: per cada feature restar mitjana i dividir entre la std(variancia), (DataFrame .mean(1/0) i same amb std)

features = data['data']
labels = data['target']
features_norm = (features - features.mean(0))/features.std(0)

# PCA (amb colors verd per benigne i vermell per maligne a partir de la label 1/0).
# En el grafic els numeros no volen dir res, simplement que com mes junts estiguin dos punts, mes semblants son les seves features

pca = PCA(n_components = 2).fit(features_norm)
features_pca = pca.transform(features_norm)
M = labels == 0
B = labels == 1
plt.figure()
plt.scatter(features_pca[B,0], features_pca[B,1], color = 'g', label = 'Benigne')
plt.scatter(features_pca[M,0], features_pca[M,1], color = 'r', label = 'Maligne')
plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Train/test del 25% test

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size = 0.25, random_state = 23)

# classificació: logistic regression

model = LogisticRegression().fit(train_x, train_y)
train_yhat = model.predict(train_x)
test_yhat = model.predict(test_x)
print(model.score(test_x, test_y))

# classificació: KNN

knn = KNeighborsClassifier(n_neighbors = 10).fit(train_x, train_y)
prediction_knn = knn.predict(test_x)
print(knn.score(test_x, test_y))

# confusion matrix

model = LinearRegression().fit(train_x, train_y)
train_yhat = model.predict(train_x)
test_yhat = model.predict(test_x)
def confusion_matrix(predicted, real, gamma):
  test_decisions = np.where(predicted < gamma, 0, 1)
  tp = np.logical_and(test_decisions == 1, real == 1).sum()
  fp = np.logical_and(test_decisions == 1, real == 0).sum()
  tn = np.logical_and(test_decisions == 0, real == 0).sum()
  fn = np.logical_and(test_decisions == 0, real == 1).sum()
  return tp, fp, tn, fn
print(confusion_matrix(test_yhat, test_y, 0.5))

# ROC curve

tpr_train = []
fpr_train = []
tpr_test = []
fpr_test = []

for gamma in np.arange(-0.5, 1.5, 0.05):
  tp_train, fp_train, tn_train, fn_train = confusion_matrix(train_yhat, train_y, gamma)
  tpr_train.append(tp_train/(tp_train + fn_train))
  fpr_train.append(fp_train/(tn_train + fp_train))
  tp_test, fp_test, tn_test, fn_test = confusion_matrix(test_yhat, test_y, gamma)
  tpr_test.append(tp_test/(tp_test + fn_test))
  fpr_test.append(fp_test/(tn_test + fp_test))

plt.figure()
plt.scatter(fpr_train, tpr_train, c = 'b', label = 'train')
plt.scatter(fpr_test, tpr_test, c = 'y', label = 'test')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# AUC per veure com de bona es la corba ROC

print(roc_auc_score(test_y, test_yhat))
print(roc_auc_score(train_y, train_yhat))

# Histogrames de les PCA 1 i 2

plt.figure()
plt.hist(features_pca[B,0], color = 'g', label = 'Benigne', bins = 100)
plt.hist(features_pca[M,0], color = 'r', label = 'Maligne', bins = 100)
plt.legend()
plt.ylabel('Repeticions')
plt.xlabel('PCA 1')
plt.show()

plt.figure()
plt.hist(features_pca[B,1], color = 'g', label = 'Benigne', bins = 100)
plt.hist(features_pca[M,1], color = 'r', label = 'Maligne', bins = 100)
plt.legend()
plt.ylabel('Repeticions')
plt.xlabel('PCA 2')
plt.show()
