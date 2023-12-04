import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqml.decisionTree import *
from sqml.randomForest import *
from sqml.knn import *
from sqml.logistic_regression import *
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data  
y = iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   

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
accuracy = accuracy_score(y_test, results) 
print('scikit-learn accuracy: ',accuracy)