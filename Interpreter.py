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
            print('no model: ',elementi[0])
            raise KeyError(elementi[0])

    # KNN

    @staticmethod
    def __vec_knn(engine,X_train, X_test, y_train, y_test):

        try:
            n = int(input("Enter neighbors: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

        model = KnnClassifierVEC(n)
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
        

    @staticmethod
    def __coo_knn(engine,X_train, X_test, y_train, y_test):

        try:
            n = int(input("Enter neighbors: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

        model = KnnClassifierCOO(n)        
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
    
    # DECISION TREE
    
    @staticmethod
    def __vec_dt(engine,X_train, X_test, y_train, y_test):
        model = DecisionTreeVEC()
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
    
    @staticmethod
    def __coo_dt(engine,X_train, X_test, y_train, y_test):
        model = DecisionTreeCOO()
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
    

    # RANDOM FOREST
    
    @staticmethod
    def __coo_rf(engine,X_train, X_test, y_train, y_test):
        try:
            n = int(input("Enter number of estimators: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        model = RandomForestCOO(n)
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
    
    @staticmethod
    def __vec_rf(engine,X_train, X_test, y_train, y_test):
        try:
            n = int(input("Enter number of estimators: "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        model = RandomForestVEC(n)
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
    
    # LOGISTIC REGRESSION

    @staticmethod
    def __coo_lr(engine,X_train, X_test, y_train, y_test):
        model = LogisticRegressionVEC()
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
        return results,y_test
    
    @staticmethod
    def __vec_lr(engine,X_train, X_test, y_train, y_test):
        model = LogisticRegressionCOO()
        results = Interpreter.do(engine, X_train, X_test, y_train, model)
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
        X = df.drop(df.columns[-1], axis=1)     
        y = df.iloc[:, -1]                      
        
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        X_train = X_train.to_numpy().astype(float)
        X_test=X_test.to_numpy().astype(float)
        y_train=y_train.to_numpy().astype(float)
        y_test=y_test.to_numpy()
        return X_train,X_test,y_train,y_test
    

    @staticmethod
    def do(engine, X_train, X_test, y_train, model):
        model.clear(engine)
        model.fit(X_train,y_train,engine)
        results=model.predict(X_test,engine)
        return results