import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqml.decisionTree import *
from sqml.randomForest import *
from sqml.knn import *
from sqml.logistic_regression import *
from sklearn.metrics import accuracy_score


data = 'data/zoo.csv'

df = pd.read_csv(data, header=None)
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

##################################################

# Primo metodo (COO)

database_path = 'sqlite:///data/data.sqlite'        
#database_path = '"postgresql://postgres:0698@localhost:5432/classification"'
engine = create_engine(database_path)
model = RandomForestCOO(4)
model.clear(engine)
model.fit(X_train,y_train,engine)
results = model.predict(X_test,engine)
accuracy = accuracy_score(y_test, results) 
print('DecisionTree COO accuracy: ',accuracy)

##################################################

# Secondo metodo (VEC)

database_path = 'sqlite:///data/data.sqlite'        
#database_path = '"postgresql://postgres:0698@localhost:5432/classification"'
engine = create_engine(database_path)
model = RandomForestVEC(4)
model.clear(engine)
model.fit(X_train,y_train,engine)
results = model.predict(X_test,engine)
accuracy = accuracy_score(y_test, results) 
print('DecisionTree VEC accuracy: ',accuracy)

##################################################

# Terzo metodo (scikit-learn)
from sklearn.ensemble import RandomForestClassifier  
model = RandomForestClassifier(4)
model.fit(X_train,y_train)
results = model.predict(X_test)
for i in range(y_test.shape[0]):
    y_test[i] = y_test[i]+1
accuracy = accuracy_score(y_test, results) 
print('scikit-learn accuracy: ',accuracy)