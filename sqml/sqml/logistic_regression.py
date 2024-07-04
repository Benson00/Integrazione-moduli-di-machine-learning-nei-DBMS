from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sqlalchemy.orm import Session


class LogisticRegressionBuilder:
    @staticmethod
    def build(dense: bool):
        return LogisticRegressionRE() if dense else LogisticRegressionCOO()


from sqlalchemy import Column, Float, Integer, MetaData, Numeric, Table, text

class LogisticRegressionRE(LogisticRegression):

    def __init__(self):
        super().__init__()
    
    def fit(self,X,Y,engine, names):
        clf = super().fit(X,Y)
        coeffs = clf.coef_
        bias = clf.intercept_
        self.mapping = {value: index for index, value in enumerate(sorted(set(Y)))}
        if coeffs.shape[0] == 1:
            coeffs = [coeffs[0], [-value for value in coeffs[0]]]
            bias = np.array([bias[0], -bias[0]])
            self.mapping = {value: index for index, value in enumerate(sorted(set(Y), reverse=True))}
        self.n = X.shape[1]
        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('id', Integer, primary_key=True),
            *[Column(f'{i}',Float) for i in names],
            Column('bias', Float)
        )
        metadata.create_all(engine)
        session = Session(engine)
        # INSERIMENTO DATI
        for x_values, y_value in zip(coeffs, bias):
            row_data = dict(zip([f'{i}' for i in names] + ['bias'], list(x_values) + [float(y_value)]))
            session.execute(training.insert().values(row_data))
        # Commit the changes
        session.commit()       


    
    def clear(self, engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)

    def insert_test(self,X_test,name:str,engine,names):
        metadata = MetaData()
        training = Table(
            name, 
            metadata,
            Column('id', Integer, primary_key=True),
            *[Column(f'{i}',Float) for i in names],
        )
        metadata.create_all(engine)
        session = Session(engine)
        X_test = np.array(X_test).astype(float)
        # INSERIMENTO DATI
        for row_values in X_test:
            row_data = dict(zip([f'{i}' for i in names], row_values))
            session.execute(training.insert().values(row_data))
        # Commit the changes
        session.commit()
    
    def predict(self, table, engine, names):
        

        connection = engine.connect()
           
        microquery = "(l.bias + "
        
        for word in names:
            microquery += f"(l.{word}*{table}.{word})+"
        microquery = microquery[:-1]+")"

        query = f'''

        WITH step AS (
            SELECT l.id as label,{table}.id as id, 1/(1+POW({math.e},-{microquery})) as prob
            FROM training as l, {table}
        ) SELECT label-1, id FROM step WHERE prob >= (SELECT MAX(prob) from step as s WHERE s.id = step.id) ORDER BY id

        '''
        query = text(query)
        results = connection.execute(query)
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[0])
            lista.append(secondo_elemento)
        a = np.array(lista)
        a = []
        for v in lista:
            e = next(key for key, value in self.mapping.items() if value == v)
            a.append(e)
        return np.array(a)

########################################################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sqlalchemy.orm import Session
from sqlalchemy import Column, Float, Integer, MetaData, Numeric, Table, text

class LogisticRegressionCOO(LogisticRegression):

    def __init__(self):
        super().__init__()
    
    def fit(self,X,Y,engine,names=None):
        clf = super().fit(X,Y)
        self.mapping = {value: index for index, value in enumerate(sorted(set(Y)))}
        coeffs = clf.coef_
        bias = clf.intercept_

        if coeffs.shape[0] == 1:
            coeffs = [coeffs[0], [-value for value in coeffs[0]]]
            coeffs = np.array(coeffs)
            bias = np.array([bias[0], -bias[0]])
            self.mapping = {value: index for index, value in enumerate(sorted(set(Y), reverse=True))}

        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('row', Integer, primary_key=True),
            Column('column', Integer, primary_key=True),
            Column('value', Float, primary_key=True),
            Column('bias',Float)
        )
        metadata.create_all(engine)
        session = Session(engine)
        righe, colonne = coeffs.shape
        for i in range(righe):
            for j in range(colonne):
                insert_query = text(
                    '''
                    INSERT INTO training ("row", "column", "value", "bias")
                    VALUES (:row, :column, :value, :bias)
                    '''
                )
                session.execute(insert_query, {"row": i, "column": j, "value": coeffs[i][j], "bias":int(bias[i])})

        # Commit the changes
        session.commit()
        
    def clear(self, engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)

    def insert_test(self,X_test, name:str, engine, names=None):
        #INSERISCO I DATI DA CLASSIFICARE IN UNA TABELLA
        metadata = MetaData()
        test = Table(
            name, 
            metadata,
            Column('row', Integer, primary_key=True),
            Column('column', Integer, primary_key=True),
            Column('value', Numeric, primary_key=True)
        )
        metadata.create_all(engine)
        session = Session(engine)
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                val = X_test[i][j]
                if isinstance(val, np.integer):
                    val = int(X_test[i][j])
                if val != 0:
                    insert_query = text(
                        f'''
                        INSERT INTO {name} ("row", "column", "value")
                        VALUES (:row, :column, :value)
                        '''
                    )
                    session.execute(insert_query, {"row": i, "column": j, "value": val})  
        session.commit()
        
    def predict(self, table, engine, names=None):
        
        connection = engine.connect()

        query = f'''
        WITH step AS (
            SELECT l.row as label,{table}.row as id, 1/(1+POW({math.e},-(l.bias+SUM(l.value*{table}.value)))) as prob
            FROM training as l, {table}
            WHERE l.column = {table}.column
            GROUP BY l.row, {table}.row, l.bias
        ) SELECT label, id FROM step WHERE prob >= (SELECT MAX(prob) from step as s WHERE s.id = step.id) ORDER BY id
        '''
        

        query = text(query)
        results = connection.execute(query)
        
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[0])
            lista.append(secondo_elemento)
        a = []
        for v in lista:
            e = next(key for key, value in self.mapping.items() if value == v)
            a.append(e)
        return np.array(a)