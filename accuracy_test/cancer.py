import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqml.decisionTree import *
from sqml.randomForest import *
from sqml.knn import *
from sqml.logistic_regression import *
from sklearn.metrics import accuracy_score


#metodo per prepare i dati
def prep():
    data = 'data/cancer.txt'

    df = pd.read_csv(data, header=None)

    col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
                'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

    df.columns = col_names

    # converto
    df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')

    df = df.drop(['Id'], axis=1)
    X = df.drop(['Class'], axis=1)

    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    #Cambio in NULL con valori presi dalla mediana della colonna
    for df1 in [X_train, X_test]:
        for col in X_train.columns:
            col_median=X_train[col].median()
            df1[col].fillna(col_median, inplace=True)

    X_train = X_train.to_numpy()
    X_test=X_test.to_numpy()
    y_train=y_train.to_numpy()
    y_test=y_test.to_numpy()

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prep()       

###################################################

# Primo metodo (COO)

#database_path = 'sqlite:///data/data.sqlite'        
database_path = "postgresql://postgres:0698@localhost:5432/classification"
engine = create_engine(database_path)
model = LogisticRegressionVEC()
model.clear(engine)
model.fit(X_train,y_train,engine)
results = model.predict(X_test,engine)
accuracy = accuracy_score(y_test, results) 
print('DecisionTree COO accuracy: ',accuracy)

##################################################

# Secondo metodo (VEC)
""" 
database_path = 'sqlite:///data/data.sqlite'        
#database_path = '"postgresql://postgres:0698@localhost:5432/classification"'
engine = create_engine(database_path)
model = RandomForestVEC(4)
model.clear(engine)
model.fit(X_train,y_train,engine)
results = model.predict(X_test,engine)
results = prep_results(results)
accuracy = accuracy_score(y_test, results) 
print('DecisionTree VEC accuracy: ',accuracy)

##################################################

# Terzo metodo (scikit-learn)
from sklearn.ensemble import RandomForestClassifier  
model = RandomForestClassifier(4)
model.fit(X_train,y_train)
results = model.predict(X_test)
accuracy = accuracy_score(y_test, results) 
print('scikit-learn accuracy: ',accuracy) """