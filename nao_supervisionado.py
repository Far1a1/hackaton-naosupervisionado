import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
# pode incluir outras bibliotecas conforme necessário

pd.set_option('display.max_columns', None)

df = pd.read_csv('data/wholesale_customers.csv')
print(df.head())
print(df.describe())
df['Channel'].value_counts(normalize=True)
print(df.drop(columns=['Channel']).corr())

X = df.drop(columns=['Channel'])
y = df['Channel']
X_train, X_test, y_train, y_test =train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
X_scaled = StandardScaler().fit_transform(X)

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_90 = np.argmax(cumulative >= 0.90) + 1
print (n_90) #encontra o primeiro componente em que a variância acumulada chega a pelo menos 90%.

pca_4 = PCA(n_components=4)
X_4d = pca_4.fit_transform(X_scaled)
X_train_4d = pca_4.fit_transform(X_train)
X_test_4d = pca_4.transform(X_test)

modelo_original = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)


modelo_original.fit(X_train, y_train)

y_pred_original = modelo_original.predict(X_test)

acc_original = accuracy_score(y_test, y_pred_original)

modelo_4d = LogisticRegression(max_iter=5000)

modelo_4d.fit(X_train_4d, y_train)

y_pred_4d = modelo_4d.predict(X_test_4d)

acc_4d = accuracy_score(y_test, y_pred_4d)
print(acc_original)
print(acc_4d)
print(classification_report(y_test, y_pred_4d))
print(f"Acuracia perdida: {acc_original - acc_4d}")