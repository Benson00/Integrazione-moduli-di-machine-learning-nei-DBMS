#preparazione dati
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from knn import Knn


iris = load_iris()
X = iris.data  
y = iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   


postgres = 'postgresql://postgres:0698@localhost:5432/classification' 

engine = create_engine(postgres)
model = Knn.build(dense=False, n=2)
#model.clear(engine)
names = ["f1","f2","f3","f4"]   #inserire i nomi degli attributi
#model.fit(X_train, y_train, engine, names)
#model.insert_test(X_train, "primo", engine, names)
#model.insert_test(X_test, "secondo", engine, names)
results = model.predict("primo",engine)
print(results)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, results) 
print(f"Precisione del classificatore: {accuracy*100}%") 
results = model.predict("secondo", engine, names)
accuracy = accuracy_score(y_test, results) 
print(f"Precisione del classificatore: {accuracy*100}%") 
