import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import Column, Float, Integer, MetaData, Table, text
from sqlalchemy.orm import Session

class KnnBuilder:
    @staticmethod
    def build(dense: bool, n: int):
        return KnnClassifierRE(n) if dense else KnnClassifierCOO(n)



class KnnClassifierRE():


    def __init__(self, n_neighbors:int):
        self.n_neighbors = n_neighbors

    def fit(self, X:np.ndarray, y:np.ndarray, engine, names):   
        # FIT DEL MODELLO     
        model = KNeighborsClassifier(self.n_neighbors)
        model.fit(X,y)       
        # CREAZIONE TABELLA
        self.n = X.shape[1]
        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('id', Integer, primary_key=True),
            *[Column(word,Float) for word in names],
            Column('label',Integer)
        )
        metadata.create_all(engine)
        session = Session(engine)
        X = np.array(X).astype(float)
        y = np.array(y).astype(float)

        # INSERIMENTO DATI
        for x_values, y_value in zip(X, y):
            row_data = dict(zip([f'{names[i]}' for i in range(self.n)] + ['label'], list(x_values) + [int(y_value)]))
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

    def predict(self, table:str, engine, names):


        connection = engine.connect()
        
        #CREAZIONE QUERY PARAMETRIZZATA
        parte1 = f'''  
        WITH neighbors AS (     
            SELECT training.id AS point1_id, {table}.id AS point2_id,
            SQRT(
            
        '''
        parte2 = ""
        for word in names:
            microquery = f"(training.{word}-{table}.{word})*(training.{word}-{table}.{word})+"
            parte2 += microquery
        parte2 = parte2[:-1]
        parte2 += f") AS euclidean_distance, training.label AS label "
        parte3 = (
                f'''
                    FROM training, {table} 
                    WHERE training.id <> {table}.id
                    ORDER BY {table}.id ASC, euclidean_distance ASC
                    ), neighbors_with_rownum AS (   
                    SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY point2_id ORDER BY euclidean_distance ASC) AS row_num
                    FROM neighbors
                    ),votes as (     
                    SELECT n.point2_id as punto, count(n.label) as count_label, label
                    from neighbors_with_rownum as n
                    WHERE row_num <= {self.n_neighbors}
                    GROUP by n.point2_id, n.label
                    order by punto ASC
                    ),rankedRes AS (
                SELECT punto, label, v.count_label, ROW_NUMBER() OVER (PARTITION BY punto ORDER BY v.count_label ASC) as Rn
                FROM votes as v
                WHERE v.count_label >= (SELECT max(v2.count_label) FROM votes as v2 WHERE v2.punto = v.punto) 
                ) SELECT punto, label FROM rankedRes WHERE Rn = 1
                ''')
        
        
        query = parte1 + parte2 + parte3
        #print(query)
        query = text(query)
        a = connection.execute(query)
        predictions=[]
        for row in a:
            secondo_elemento = int(row[1])
            predictions.append(secondo_elemento)
        predictions = np.array(predictions)
        return predictions
    


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import Column, Float,REAL, Integer, MetaData, Table, text
from sqlalchemy.orm import Session
from sqlalchemy.types import Numeric

#####################################################################################################################

class KnnClassifierCOO():

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self,X:np.ndarray, y:np.ndarray,engine,names=None):        
        model = KNeighborsClassifier(self.n_neighbors)
        model.fit(X,y)       
        
        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('row', Integer, primary_key=True),
            Column('column', Integer, primary_key=True),
            Column('value', Numeric, primary_key=True),
            Column('label',Integer)
        )
        metadata.create_all(engine)
        session = Session(engine)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = X[i][j]
                if isinstance(val, np.integer):
                    val = int(X[i][j])                    
                insert_query = text(
                    '''
                    INSERT INTO training ("row", "column", "value", "label")
                    VALUES (:row, :column, :value, :label)
                    '''
                )
                session.execute(insert_query, {"row": i, "column": j, "value": val, "label": int(y[i])})

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


    def predict(self, table:str, engine, names=None):
        
        #LA QUERY
        query = f'''
           WITH neighbors AS (
                SELECT row1.row as r1, row2.row as r2, 
                SQRT(SUM((row1.value - row2.value) * (row1.value - row2.value))) AS euclidean_distance, row1.label
                FROM training AS row1, {table} AS row2
                WHERE row1.column = row2.column
                GROUP BY row1.row, row2.row, row1.label
                ORDER BY row2.row ASC, euclidean_distance ASC
            ),
        
        
        
        neighbors_with_rownum AS (   
                SELECT *,
                ROW_NUMBER() OVER (PARTITION BY r2 ORDER BY euclidean_distance ASC) AS row_num
                FROM neighbors
            ), votes AS (
                SELECT n.r2 as punto, count(n.label) as count_label, label
                FROM neighbors_with_rownum as n
                WHERE row_num <= {self.n_neighbors}
                GROUP by n.r2, n.label
                ORDER BY punto ASC
            ), rankedRes AS (
                SELECT punto, label, v.count_label, ROW_NUMBER() OVER (PARTITION BY punto ORDER BY v.count_label ASC) as Rn
                FROM votes as v
                WHERE v.count_label >= (SELECT max(v2.count_label) FROM votes as v2 WHERE v2.punto = v.punto) 
                ) SELECT punto, label FROM rankedRes WHERE Rn = 1
        '''
        query = text(query)
        connection = engine.connect()
        a = connection.execute(query)

        #RESTITUISCO I RISULTATI DELLA QUERY
        predictions=[]
        for row in a:
            secondo_elemento = int(row[1])
            predictions.append(secondo_elemento)
        predictions = np.array(predictions)
        return predictions