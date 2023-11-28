from sklearn.datasets import load_iris
from sqlalchemy import create_engine
from logistic_regression import Logistic_Regression
from logistic_regression import Logistic_Regression2

X, y = load_iris(return_X_y=True)

# VETTORIALE
engine = create_engine("postgresql://postgres:0698@localhost:5432/irislr")

model = Logistic_Regression()
model.clear(engine)
model.fit(X,y,engine)
results=model.predict(X,engine)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, results) 
print(accuracy) 


# COO

engine = create_engine("postgresql://postgres:0698@localhost:5432/irislr2")

model2 = Logistic_Regression2()
model2.clear(engine)
model2.fit(X, y, engine)

res = model2.predict(X,engine)
accuracy2 = accuracy_score(y, res) 

# scikit
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X, y)
clf.fit(X,y)
r = clf.predict(X)
accuracy3 = accuracy_score(y, r) 

# RISULTATI
print(f"Vettoriale: {accuracy}, COO: {accuracy2}, scikit; {accuracy3}")                                                           
