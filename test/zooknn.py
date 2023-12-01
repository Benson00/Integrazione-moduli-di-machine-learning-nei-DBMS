import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from DecisionTree import DecisionTree, DecisionTree2
from KNN import Knn, KnnClassifier
from randomForest import RandomForest, RandomForest2
from logistic_regression import Logistic_Regression, Logistic_Regression2
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



data = 'data/zoo.csv'

df = pd.read_csv(data, header=None)
print(df.info())
df = df.drop([0], axis=1)
X = df.drop([len(df.columns)], axis=1)

y = df[len(df.columns)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train = X_train.to_numpy().astype(float)
X_test=X_test.to_numpy().astype(float)
y_train=y_train.to_numpy().astype(float)
y_test=y_test.to_numpy().astype(float)
for i in range(y_test.shape[0]):
    y_test[i] = y_test[i]-1
engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")
c = DecisionTree2()
c.clear(engine)
c.fit(X_train, y_train,engine)
a=c.predict(X_test,engine)
print(a)
accuracy = accuracy_score(y_test, a) 
print('accuracy: ',accuracy)