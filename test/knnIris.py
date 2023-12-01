from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from KNN import KnnClassifier


iris = load_iris()
X = iris.data  
y = iris.target 
print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   
engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")
c = KnnClassifier(3)
c.clear(engine)
c.fit(X_train, y_train,engine)
a=c.predict(X_test, engine)
accuracy = accuracy_score(y_test, a) 
print('VEC accuracy: ',accuracy)