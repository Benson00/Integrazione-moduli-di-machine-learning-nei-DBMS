import pandas as pd
from sqml.knn import KnnClassifierVEC, KnnClassifierCOO
from sqml.decisionTree import DecisionTreeCOO, DecisionTreeVEC
from sqml.randomForest import RandomForestVEC, RandomForestCOO
from sqml.logistic_regression import LogisticRegressionVEC, LogisticRegressionCOO
from sklearn.model_selection import train_test_split
import re


class Interpreter:

    def __init__(self,engine):

        self.engine = engine
        
        # dict contains:
        # - model's name as key
        # - function that return classification as value
        self.models = {
            "vecknn":Interpreter.__vec_knn,
            "cooknn":Interpreter.__coo_knn,
            "coodt":Interpreter.__coo_dt,
            "vecdt":Interpreter.__vec_dt,
            "vecrf":Interpreter.__vec_rf,
            "coorf":Interpreter.__coo_rf,
            "veclr":Interpreter.__vec_lr,
            "coolr":Interpreter.__coo_lr,
            "vecmb":Interpreter.__vec_mb,
            "coomb":Interpreter.__coo_mb
            # other models ...
        }


    def input(self,program:str):
        # modello data
        # parse
        # controllo che il modello sia presente in self.models
        # se presente eseguo la funzione associata 
        # se non presente sollevo eccezione
        
        pattern = re.compile(r'\S+')
        elementi = pattern.findall(program)
        self.data = elementi[1]
        X_train, X_test, y_train, y_test = Interpreter.__prepare_data(self.data)
        self.classifier = elementi[0]
        if self.classifier in self.models:
            return self.models[self.classifier](self.engine,X_train, X_test, y_train, y_test)
        else:
            raise KeyError

    # KNN

    @staticmethod
    def __vec_knn(engine,X_train, X_test, y_train, y_test):

        try:
            n = int(input("Enter neighbors: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

        model = KnnClassifierVEC(n)
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
        

    @staticmethod
    def __coo_knn(engine,X_train, X_test, y_train, y_test):

        try:
            n = int(input("Enter neighbors: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

        model = KnnClassifierCOO(n)        
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    
    # DECISION TREE
    
    @staticmethod
    def __vec_dt(engine,X_train, X_test, y_train, y_test):
        model = DecisionTreeVEC()
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    
    @staticmethod
    def __coo_dt(engine,X_train, X_test, y_train, y_test):
        model = DecisionTreeCOO()
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    

    # RANDOM FOREST
    
    @staticmethod
    def __coo_rf(engine,X_train, X_test, y_train, y_test):
        try:
            n = int(input("Enter number of estimators: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        model = RandomForestCOO(n)
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    
    @staticmethod
    def __vec_rf(engine,X_train, X_test, y_train, y_test):
        try:
            n = int(input("Enter number of estimators: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        model = RandomForestVEC(n)
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    
    # LOGISTIC REGRESSION

    @staticmethod
    def __coo_lr(engine,X_train, X_test, y_train, y_test):
        model = LogisticRegressionVEC()
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    
    @staticmethod
    def __vec_lr(engine,X_train, X_test, y_train, y_test):
        model = LogisticRegressionCOO()
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results,y_test
    

    # MULTINOMIAL BAYES

    @staticmethod
    def __coo_mb(engine,X_train, X_test, y_train, y_test):
        pass

    @staticmethod
    def __vec_mb(engine,X_train, X_test, y_train, y_test):
        pass

    
    # PREPARAZIONE DATI

    @staticmethod
    def __prepare_data(data:str):
        df = pd.read_csv(data, header=None)
        X = df.drop(df.columns[-1], axis=1)     # remove label
        y = df.iloc[:, -1]                      # save in y labels
        
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        X_train = X_train.to_numpy()
        X_test=X_test.to_numpy()
        y_train=y_train.to_numpy()
        y_test=y_test.to_numpy()
        return X_train,X_test,y_train,y_test
    