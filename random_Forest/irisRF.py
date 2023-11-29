from sqlalchemy import create_engine
from randomForest import RandomForest, RandomForest2
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data  
Y = iris.target

# Primo metodo (COO)
engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")
model = RandomForest(2)
model.clear(engine)
model.fit(X,Y,engine)
results = model.predict(X,engine)
accuracy = accuracy_score(Y, results) 
print('DecisionTree COO accuracy: ',accuracy)

##################################################

