import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("crop.csv")

data = df.copy()
c = data['label'].astype('category')
targets = dict(enumerate(c.cat.categories))
data['target'] = c.cat.codes
x= data[['N','P','K','temperature','humidity','ph','rainfall']]

cor = x.corr()
# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(5,5) )


mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.title("Correlation of Features", y = 1, size = 15)

sns.heatmap(cor, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.show()

sns.jointplot(x= 'rainfall',y = 'ph', data = df, hue='label')
sns.jointplot(x= 'rainfall',y = 'humidity', data = df[df['rainfall'] >= 150], hue='label')
sns.jointplot(x= 'rainfall',y = 'ph', data = df[df['rainfall'] >= 150], hue='label')
sns.jointplot(x= 'rainfall',y = 'humidity', data = df[df['temperature'] >= 30], hue='label')
sns.jointplot(x= 'rainfall',y = 'ph', data = df[df['temperature'] >= 30], hue='label')


features = {'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'}

def crop_boxplot(X):
    ax = sns.set_style('whitegrid')
    plt.subplots(figsize=(15,8))
    sns.boxplot(x=X,y='label',data=df)
    
    plt.title("Crops Relation with " + str(X),fontsize=24)
    plt.xlabel("values of " + str(X),fontsize=18)
    plt.ylabel("Crops Name", fontsize=18)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x=df[df.columns[:-1]]
y=df
# data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_standardized = x.copy()
df_standardized

for col in df_standardized.columns:
    df_standardized[col] = scaler.fit_transform(df_standardized[col].values.reshape(-1, 1))

df_standardized.head(10)    

pca = PCA(n_components=3)
df_pcs = pca.fit_transform(df_standardized.values)
df_pca = pd.DataFrame(data = df_pcs, columns = ["pca_1", "pca_2", "pca_3"])

df_pca = pd.concat([df_pca, y], axis = 1)
df_pca

import plotly.express as px
df_avg_pca_proximity = df_pca.groupby('label')[['pc_1', 'pc_2', 'pc_3']].mean()

fig = px.scatter_3d(df_avg_pca_proximity, x='pc_1', y='pc_2', z='pc_3',
                    color = df_avg_pca_proximity.index,
                    text = df_avg_pca_proximity.index, 
                    title = "Which Crops Need Similar Conditions?",
                    template = 'none')
fig.update_layout(showlegend=False)
fig.show()


X=df[['N','P','K','temperature','humidity','ph','rainfall']]
c = df['label'].astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes
y = df['target']

# feature scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state=48)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
knn_accuracies = []

for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    y_pred = knn.fit(X_train, y_train).predict(X_test)
    knn_accuracies.append(metrics.accuracy_score(y_test, y_pred))
    
k_best = knn_accuracies.index(max(knn_accuracies)) + 1

knn = KNeighborsClassifier(n_neighbors = k_best)
y_pred = knn.fit(X_train, y_train).predict(X_test)

knn_accuracy = metrics.accuracy_score(y_test, y_pred)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

naive_bayes_accuracy = metrics.accuracy_score(y_test,y_pred)

log_reg = LogisticRegression()
y_pred = log_reg.fit(X_train, y_train).predict(X_test)

logistic_regression_accuracy = metrics.accuracy_score(y_test, y_pred)

svc = SVC(kernel='linear') 
y_pred = svc.fit(X_train, y_train).predict(X_test)

svc_accuracy = metrics.accuracy_score(y_test, y_pred)

dt = DecisionTreeClassifier()
y_pred = dt.fit(X_train, y_train).predict(X_test)

tree_accuracy = metrics.accuracy_score(y_test, y_pred)

rfc = RandomForestClassifier(n_estimators = 100)
y_pred = rfc.fit(X_train, y_train).predict(X_test)

rfc_accuracy = metrics.accuracy_score(y_test, y_pred)

xgb = XGBClassifier()
y_pred = xgb.fit(X_train, y_train).predict(X_test)
xgb_accuracy = metrics.accuracy_score(y_test, y_pred)

classification_performance = {'Classification Procedure': ['Naive Bayes', 'Logistic Regression', 'KNN', 'SVC', 'Decision Tree', 'Random Forest','XGB'],
                              'Accuracy': [naive_bayes_accuracy, logistic_regression_accuracy, knn_accuracy, svc_accuracy, tree_accuracy, rfc_accuracy,xgb_accuracy]}

classification_performance = pd.DataFrame.from_dict(classification_performance)
classification_performance
