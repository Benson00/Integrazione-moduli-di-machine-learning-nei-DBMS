import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import Column, Float, Integer, MetaData, Table, text
from sqlalchemy.orm import Session



class KnnClassifierVEC():


    def __init__(self, n_neighbors:int):
        self.n_neighbors = n_neighbors

    def fit(self,X:np.ndarray, y:np.ndarray, engine):   
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
            *[Column(f'col_{i}',Float) for i in range(0, self.n)],
            Column('label',Integer)
        )
        metadata.create_all(engine)
        session = Session(engine)
        # INSERIMENTO DATI
        for x_values, y_value in zip(X, y):
            row_data = dict(zip([f'col_{i}' for i in range(self.n)] + ['label'], list(x_values) + [int(y_value)]))
            session.execute(training.insert().values(row_data))
        # Commit the changes
        session.commit()


    def clear(self, engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)

    def predict(self, X_test, engine):

        metadata = MetaData()
        training = Table(
            "test", 
            metadata,
            Column('id', Integer, primary_key=True),
            *[Column(f'col_{i}',Float) for i in range(0, self.n)],
        )
        metadata.create_all(engine)
        session = Session(engine)
        # INSERIMENTO DATI
        for row_values in X_test:
            row_data = dict(zip([f'col_{i}' for i in range(self.n)], row_values))
            session.execute(training.insert().values(row_data))
        # Commit the changes
        session.commit()

        connection = engine.connect()
        
        #CREAZIONE QUERY PARAMETRIZZATA
        parte1 = f'''  
        WITH neighbors AS (     
            SELECT training.id AS point1_id, test.id AS point2_id,
            SQRT(
            
        '''
        parte2 = ""
        for word in range(self.n):
            microquery = f"(training.col_{word}-test.col_{word})*(training.col_{word}-test.col_{word})+"
            parte2 += microquery
        parte2 = parte2[:-1]
        parte2 += f") AS euclidean_distance, training.label AS label "
        parte3 = (
                f'''
                    FROM training, test 
                    WHERE training.id <> test.id
                    ORDER BY test.id ASC, euclidean_distance ASC
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
        query = text(query)
        a = connection.execute(query)
        predictions=[]
        for row in a:
            secondo_elemento = int(row[1])
            predictions.append(secondo_elemento)
        predictions = np.array(predictions)
        return predictions


class KnnClassifierCOO():

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self,X:np.ndarray, y:np.ndarray,engine):        
        model = KNeighborsClassifier(self.n_neighbors)
        model.fit(X,y)       
        
        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('row', Integer, primary_key=True),
            Column('column', Integer, primary_key=True),
            Column('value', Float, primary_key=True),
            Column('label',Integer)
        )
        metadata.create_all(engine)
        session = Session(engine)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                insert_query = text(
                    '''
                    INSERT INTO training ("row", "column", "value", "label")
                    VALUES (:row, :column, :value, :label)
                    '''
                )
                session.execute(insert_query, {"row": i, "column": j, "value": X[i][j], "label": int(y[i])})

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
        #LA QUERY
        query = f'''
           WITH neighbors AS (
                SELECT row1.row as r1, row2.row as r2, 
                SQRT(SUM((row1.value - row2.value) * (row1.value - row2.value))) AS euclidean_distance, row1.label
                FROM training AS row1, test AS row2
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
