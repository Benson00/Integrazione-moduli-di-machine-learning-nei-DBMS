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
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris.data  
y = iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   


database_path = 'sqlite:///data/data.sqlite'        
#database_path = "postgresql://postgres:0698@localhost:5432/classification"
engine = create_engine(database_path)
model = DecisionTreeBuilder.build(True)
model.clear(engine)
model.insert_test(X,"test", engine, names)
print("fitted")
