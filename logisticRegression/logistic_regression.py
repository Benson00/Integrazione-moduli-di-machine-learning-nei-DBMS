from sklearn.linear_model import LogisticRegression
import sqlite3
import numpy as np
import math
from sqlalchemy.orm import Session


from sqlalchemy import Column, Float, Integer, MetaData, Table, text

class Logistic_Regression(LogisticRegression):

    def __init__(self):
        super().__init__()
    
    def fit(self,X,Y,engine):
        clf = super().fit(X,Y)
        coeffs = clf.coef_
        bias = clf.intercept_
        print(bias)
        if coeffs.shape[0] == 1:
            coeffs = [coeffs[0], [-value for value in coeffs[0]]]
            bias = np.array([bias[0], -bias[0]])
        self.n = X.shape[1]
        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('id', Integer, primary_key=True),
            *[Column(f'f{i}',Float) for i in range(0, self.n)],
            Column('bias', Float)
        )
        metadata.create_all(engine)
        session = Session(engine)
        # INSERIMENTO DATI
        for x_values, y_value in zip(coeffs, bias):
            row_data = dict(zip([f'f{i}' for i in range(self.n)] + ['bias'], list(x_values) + [float(y_value)]))
            session.execute(training.insert().values(row_data))
        # Commit the changes
        session.commit()
    
        
    
    def clear(self, engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)
    
    def predict(self, X, engine):
        metadata = MetaData()
        training = Table(
            "test", 
            metadata,
            Column('id', Integer, primary_key=True),
            *[Column(f'f{i}',Float) for i in range(0, self.n)],
        )
        metadata.create_all(engine)
        session = Session(engine)
        # INSERIMENTO DATI
        for row_values in X:
            row_data = dict(zip([f'f{i}' for i in range(self.n)], row_values))
            session.execute(training.insert().values(row_data))
        session.commit()

        connection = engine.connect()
           
        microquery = "(l.bias + "
        
        for word in range (self.n):
            microquery += f"(l.f{word}*test.f{word})+"
        microquery = microquery[:-1]+")"
        print(microquery) 
        query = f'''

        WITH step AS (
            SELECT l.id as label,test.id as id, 1/(1+POW({math.e},-{microquery})) as prob
            FROM training as l, test
        ) SELECT label-1, id FROM step WHERE prob >= (SELECT MAX(prob) from step as s WHERE s.id = step.id) ORDER BY id

        '''
        query = text(query)
        results = connection.execute(query)
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[0])
            lista.append(secondo_elemento)
        a = np.array(lista)
        return a


class Logistic_Regression2(LogisticRegression):

    def __init__(self):
        super().__init__()
    
    def fit(self,X,Y,engine):
        clf = super().fit(X,Y)
        
        coeffs = clf.coef_
        bias = clf.intercept_

        if coeffs.shape[0] == 1:
            coeffs = [coeffs[0], [-value for value in coeffs[0]]]

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
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                insert_query = text(
                    '''
                    INSERT INTO training ("row", "column", "value", "bias")
                    VALUES (:row, :column, :value, :bias)
                    '''
                )
                session.execute(insert_query, {"row": i, "column": j, "value": coeffs[i][j], "bias": int(bias[i])})

        # Commit the changes
        session.commit()
        
    def clear(self, engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)
        
    def predict(self, X_test, engine):
        #INSERISCO I DATI DA CLASSIFICARE IN UNA TABELLA
        metadata = MetaData()
        test = Table(
            "test", 
            metadata,
            Column('row', Integer, primary_key=True),
            Column('column', Integer, primary_key=True),
            Column('value', Float, primary_key=True)
        )
        metadata.create_all(engine)
        session = Session(engine)
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                insert_query = text(
                    '''
                    INSERT INTO test ("row", "column", "value")
                    VALUES (:row, :column, :value)
                    '''
                )
                session.execute(insert_query, {"row": i, "column": j, "value": X_test[i][j]})  
        session.commit()
        query = f'''
        WITH step AS (
            SELECT l.row as label,test.row as id, 1/(1+POW({math.e},-(l.bias+SUM(l.value*test.value)))) as prob
            FROM training as l, test
            WHERE l.column = test.column
            GROUP BY l.row, test.row, l.bias
        ) SELECT label, id FROM step WHERE prob >= (SELECT MAX(prob) from step as s WHERE s.id = step.id) ORDER BY id

        '''
        connection = engine.connect()

        query = text(query)
        results = connection.execute(query)
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[0])
            lista.append(secondo_elemento)
        a = np.array(lista)
        return a
                