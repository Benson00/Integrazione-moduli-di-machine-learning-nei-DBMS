from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, text
from sqlalchemy.orm import Session

class DecisionTree(DecisionTreeClassifier):

    def __init__(self):
        super().__init__()

    def fit(self,X,Y,engine):
        super().fit(X,Y)
            
            
        tree_rules = self.tree_


        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('nodeid', Integer, primary_key=True),
            Column('featurename', Integer),
            Column('threshold', Float),
            Column('leftchild', Integer),
            Column('rightchild', Integer),
            Column('label',String)
        )
        
        metadata.create_all(engine)
        session = Session(engine)
            
        def insert_rules(node, parent_id):
            if tree_rules.children_left[node] == tree_rules.children_right[node]:  # Nodo foglia # type: ignore
                insert_query = text(
                    '''
                    INSERT INTO training ("nodeid", "featurename", "threshold", "leftchild", "rightchild", "label")
                    VALUES (:nodeid, :featurename, :threshold, :leftchild, :rightchild, :label)
                    '''
                )
                session.execute(insert_query, {"nodeid":int(node), "featurename":None, "threshold":None, "leftchild":None, "rightchild":None, "label":str(tree_rules.value[node].argmax())})
                session.commit()
            else:
                feature = tree_rules.feature[node] # type: ignore
                threshold = tree_rules.threshold[node] # type: ignore
                left_node = tree_rules.children_left[node] # type: ignore
                right_node = tree_rules.children_right[node] # type: ignore

                insert_query = text(
                    '''
                    INSERT INTO training ("nodeid", "featurename", "threshold", "leftchild", "rightchild", "label")
                    VALUES (:nodeid, :featurename, :threshold, :leftchild, :rightchild, :label)
                    '''
                )

                session.execute(insert_query, {"nodeid":int(node), "featurename":str(feature), "threshold":float(threshold), "leftchild":int(left_node), "rightchild":int(right_node), "label":None})
                session.commit()

                insert_rules(left_node, node)
                insert_rules(right_node, node)
                    
        insert_rules(0, -1)



    def clear(self, engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)        

                           
               
    def predict(self, X_test, engine):
        
        metadata = MetaData()
        test = Table(
            "test", 
            metadata,
            Column('row', Integer, primary_key=True),
            Column('col', Integer, primary_key=True),
            Column('value', Float, primary_key=True)
        )
        metadata.create_all(engine)
        session = Session(engine)
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                insert_query = text(
                    '''
                    INSERT INTO test ("row", "col", "value")
                    VALUES (:row, :col, :value)
                    '''
                )
                session.execute(insert_query, {"row": i, "col": j, "value": X_test[i][j]})  
        session.commit()
        
        query ='''
            WITH RECURSIVE DecisionPath AS (
                SELECT
                    training.NodeID AS CurrentNode,
                    FeatureName,
                    Threshold,
                    LeftChild,
                    RightChild,
                    label,
                    T.row as tr
                FROM
                    training, test as T
                WHERE
                    NodeID = 0  -- Inizia dalla radice dell'albero

                UNION ALL

                SELECT
                    d.NodeId,
                    d.FeatureName,
                    d.Threshold,
                    d.LeftChild,
                    d.RightChild,
                    d.label,
                    test.row
                FROM
                    training AS d, test, DecisionPath as dp
                WHERE
                    (
                        dp.LeftChild = d.NodeID AND dp.FeatureName = test.col AND test.value <= dp.Threshold AND tr = test.row
                    ) OR
                    (
                        dp.RightChild = d.NodeID AND dp.FeatureName = test.col AND test.value > dp.Threshold AND tr = test.row
                    )
            )
            SELECT label, tr
            FROM DecisionPath
            WHERE label IS NOT NULL
            group by tr, label
            ORDER BY tr
        '''
        query = text(query)
        connection = engine.connect()
        results = connection.execute(query)
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[0])
            lista.append(secondo_elemento)
        a = np.array(lista)
        return a
    

class DecisionTree2(DecisionTreeClassifier):
    def __init__(self):
        super().__init__()

    def fit(self,X,Y,conn):
        super().fit(X,Y)
        tree_rules = self.tree_
        self.n_features = X.shape[1]

        cursor = conn.cursor()
            
        # Creazione tabella contenente le informazioni dell'albero
        cursor.execute('''CREATE TABLE IF NOT EXISTS decision_tree (
                            NodeID INT PRIMARY KEY,
                            FeatureName VARCHAR(255),
                            Threshold FLOAT,
                            LeftChild INT,  -- NULL se il nodo è una foglia
                            RightChild INT,  -- NULL se il nodo è una foglia
                            ClassLabel VARCHAR(255)  -- NULL se il nodo non è una foglia 
                        );''')
            
        def insert_rules(node, parent_id):
            if tree_rules.children_left[node] == tree_rules.children_right[node]:  # Nodo foglia # type: ignore
                cursor.execute('''
                        INSERT INTO decision_tree (NodeID, FeatureName, Threshold, LeftChild, RightChild, ClassLabel)
                        VALUES (?, NULL, NULL, NULL, NULL, ?)
                    ''', (int(node), str(tree_rules.value[node].argmax())))
                conn.commit()
            else:
                feature = tree_rules.feature[node] # type: ignore
                threshold = tree_rules.threshold[node] # type: ignore
                left_node = tree_rules.children_left[node] # type: ignore
                right_node = tree_rules.children_right[node] # type: ignore

                cursor.execute('''
                        INSERT INTO decision_tree (NodeID, FeatureName, Threshold, LeftChild, RightChild, ClassLabel)
                        VALUES (?, ?, ?, ?, ?, NULL)
                    ''', (int(node), "f"+str(feature), float(threshold), int(left_node), int(right_node)))
                conn.commit()
                insert_rules(left_node, node)
                insert_rules(right_node, node)
                    
        insert_rules(0, -1)


    def clear_data(self,conn):
        cursor=conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS test")
        cursor.execute("DROP TABLE IF EXISTS decision_tree")

    def prepare_test(self,X_test, conn):
        cursor = conn.cursor()
        columns = [f'f{i}' for i in range(len(X_test[0]))]
        column_types = ['FLOAT' if isinstance(X_test[0][i], float) else 'INTEGER' for i in range(len(X_test[0]))]

        table_name = 'test'
        create_table_query = f'CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, '

        for i in range(len(columns)):
            create_table_query += f'{columns[i]} {column_types[i]}, '       #colonna tipo_colonna
        create_table_query = create_table_query[:-2]+")"
        cursor.execute(create_table_query)          #CREA LA TABELLA
        # Inserimento dei dati nella tabella
        for i in range(len(X_test)):
            insert_query = f'INSERT INTO {table_name} ('
            insert_query += ', '.join(columns) +") VALUES ("
            insert_query += ', '.join(['?' for _ in range(len(X_test[0]))]) + ')'
            cursor.execute(insert_query, tuple(X_test[i]))
        conn.commit()
 
    def predict(self,conn):
        cursor = conn.cursor()
        
        query ='''
            WITH RECURSIVE DecisionPath AS (
                SELECT
                    NodeID AS CurrentNode,
                    FeatureName,
                    Threshold,
                    LeftChild,
                    RightChild,
                    ClassLabel,
                    test.id as tr
                FROM
                    decision_tree, test
                WHERE
                    NodeID = 0  -- Inizia dalla radice dell'albero

                UNION ALL

                SELECT
                    d.NodeId,
                    d.FeatureName,
                    d.Threshold,
                    d.LeftChild,
                    d.RightChild,
                    d.ClassLabel,
                    test.id
                FROM
                    decision_tree AS d, test
                JOIN
                    DecisionPath AS dp ON
                    (
                        dp.LeftChild = d.NodeID AND'''
        
        
        microquery = "("
        for i in range(self.n_features):
            microquery += f''' (dp.FeatureName='f{i}' AND test.f{i} <= dp.Threshold) OR''' 
        microquery = microquery[:-2] +")"

        query += microquery
        query += '''
        AND tr = test.id
                    ) OR
                    (
                        dp.RightChild = d.NodeID AND'''
        
        microquery = "("
        for i in range(self.n_features):
            microquery += f''' (dp.FeatureName='f{i}' AND test.f{i} > dp.Threshold) OR''' 
        microquery = microquery[:-2] +")"
        query += microquery
        query += '''
        
        AND tr = test.id
                    )
            )
            SELECT ClassLabel, tr
            FROM DecisionPath
            WHERE ClassLabel IS NOT NULL
            group by tr;
        '''
        results = cursor.execute(query)
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[0])
            lista.append(secondo_elemento)
        a = np.array(lista)
        return a