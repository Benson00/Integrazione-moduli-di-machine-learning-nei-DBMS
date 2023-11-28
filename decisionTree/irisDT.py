from sqlalchemy import create_engine
from DecisionTree import DecisionTree
from DecisionTree import DecisionTree2
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data  
Y = iris.target 

engine = create_engine("postgresql://postgres:0698@localhost:5432/irisdt")

model = DecisionTree() 
model.clear(engine)
model.fit(X,Y,engine)
results = model.predict(X,engine)
accuracy = accuracy_score(Y, results) 
print(accuracy)


