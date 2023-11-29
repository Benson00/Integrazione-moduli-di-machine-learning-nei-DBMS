from sqlalchemy import create_engine
from DecisionTree import DecisionTree
from DecisionTree import DecisionTree2
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data  
Y = iris.target 

engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")

model = DecisionTree() 
model.clear(engine)
model.fit(X,Y,engine)
results = model.predict(X,engine)
accuracy = accuracy_score(Y, results) 
print('DecisionTree COO accuracy: ',accuracy)

##################################################

iris = load_iris()
X = iris.data  
Y = iris.target 

engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")

model = DecisionTree2() 
model.clear(engine)
model.fit(X,Y,engine)
results = model.predict(X,engine)
accuracy = accuracy_score(Y, results) 
print('DecisionTree vect accuracy: ',accuracy)

##################################################

from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data  
Y = iris.target 
model = DecisionTreeClassifier()
model.fit(X,Y)
results = model.predict(X)
accuracy = accuracy_score(Y, results) 
print('DecisionTree sklearn accuracy: ',accuracy)
