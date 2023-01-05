from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression #Logistic(Regression)Classifier
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.svm import SVC #Support Vector Machine
from sklearn.naive_bayes import GaussianNB #Naive Bayesian
from sklearn.neighbors import KNeighborsClassifier #K Nearest Neighbor
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosing
from sklearn.neural_network import MLPClassifier #Neural Network
from sklearn.metrics import accuracy_score
from sklearn import model_selection

iris = load_iris()



#feature_names와 target을 레코드로 갖는 데이터프레임 생성
df = pd.DataFrame(data = iris.data, columns=iris.feature_names)
df['target']= iris.target
df['target'] = df['target'].map({0:"setosa",1:"versicolor",2:"virginica"})
print(df)

#슬라이싱을 통해 feature와 label 분리
x_data = df.iloc[:, :-1]
y_data = df.iloc[:, [-1]]

sns.pairplot(df, hue="target", height=3)
#plt.show()

models = []
models.append(("LR", LogisticRegression()))
models.append(("DT", DecisionTreeClassifier()))
models.append(("SVM", SVC()))
models.append(("NB", GaussianNB()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("GB", GradientBoostingClassifier()))
models.append(("ANN", MLPClassifier()))

#모델 학습 및 정확도 분석
for name, model in models:
    model.fit(x_data, y_data.values.ravel())
    y_pred = model.predict(x_data)
    print(name, "'s Accuracy is", accuracy_score(y_data, y_pred))
    
#교차검증
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
    cv_results = model_selection.cross_val_score(model, x_data, y_data.values.ravel(), cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    
fig = plt.figure()

fig.suptitle('Classifier Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
