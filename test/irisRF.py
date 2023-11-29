from sqlalchemy import create_engine
from sqlearn.randomForest import RandomForest, RandomForest2
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data  
Y = iris.target

# Primo metodo (COO)
engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")
model = RandomForest(4)
model.clear(engine)
model.fit(X,Y,engine)
results = model.predict(X,engine)
accuracy = accuracy_score(Y, results) 
print('DecisionTree COO accuracy: ',accuracy)

##################################################

# Secondo metodo (VEC)
engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")
model = RandomForest2(4)
model.clear(engine)
model.fit(X,Y,engine)
results = model.predict(X,engine)
#print(results)
accuracy = accuracy_score(Y, results) 
print('DecisionTree VEC accuracy: ',accuracy)

##################################################

# Terzo metodo (scikit-learn)
from sklearn.ensemble import RandomForestClassifier  
model = RandomForestClassifier(4)
model.fit(X,Y)
results = model.predict(X)
#print(results)
accuracy = accuracy_score(Y, results) 
print('scikit-learn accuracy: ',accuracy)