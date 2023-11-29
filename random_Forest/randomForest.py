import numpy as np
from sklearn.ensemble import RandomForestClassifier  
import numpy as np
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, text
from sqlalchemy.orm import Session


class RandomForest(RandomForestClassifier):

    def __init__(self,n_estimators):
        super().__init__(n_estimators)

    def fit(self,X,Y, engine):
        model = super().fit(X,Y)
        forest = model.estimators_


        metadata = MetaData()
        training = Table(
            "training", 
            metadata,
            Column('treeid', Integer, primary_key=True),
            Column('nodeid', Integer, primary_key=True),
            Column('featurename', Integer),
            Column('threshold', Float),
            Column('leftchild', Integer),
            Column('rightchild', Integer),
            Column('label',String)
        )
        
        metadata.create_all(engine)
        session = Session(engine)

        def insert_tree(tree,i):
            def insert_rules(node, parent_id):
                if tree.children_left[node] == tree.children_right[node]:  # Nodo foglia # type: ignore
                    insert_query = text(
                        '''
                        INSERT INTO training ("treeid", "nodeid", "featurename", "threshold", "leftchild", "rightchild", "label")
                        VALUES (:treeid, :nodeid, :featurename, :threshold, :leftchild, :rightchild, :label)
                        '''
                    )
                    session.execute(insert_query, {"treeid":int(i), "nodeid":int(node), "featurename":None, "threshold":None, "leftchild":None, "rightchild":None, "label":str(tree.value[node].argmax())})
                    session.commit()
                else:
                    feature = tree.feature[node] # type: ignore
                    threshold = tree.threshold[node] # type: ignore
                    left_node = tree.children_left[node] # type: ignore
                    right_node = tree.children_right[node] # type: ignore

                    insert_query = text(
                        '''
                        INSERT INTO training ("treeid","nodeid", "featurename", "threshold", "leftchild", "rightchild", "label")
                        VALUES (:treeid, :nodeid, :featurename, :threshold, :leftchild, :rightchild, :label)
                        '''
                    )

                    session.execute(insert_query, {"treeid":int(i), "nodeid":int(node), "featurename":str(feature), "threshold":float(threshold), "leftchild":int(left_node), "rightchild":int(right_node), "label":None})
                    session.commit()

                    insert_rules(left_node, node)
                    insert_rules(right_node, node)
                        
            insert_rules(0, -1)

        for i,tree in enumerate(forest):
            insert_tree(tree.tree_,i)
        
        


    def clear(self,engine):
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(bind=engine)

    
    def predict(self,X_test,engine):
        
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
        WITH RECURSIVE ForestDecisionPaths AS (
            SELECT
                treeid,
                NodeID AS CurrentNode,
                FeatureName,
                Threshold,
                LeftChild,
                RightChild,
                label,
                test.row AS tr
            FROM
                training, test
            WHERE
                NodeID = 0  -- Inizia dalla radice dell'albero

            UNION ALL

            SELECT
                d.treeid,
                d.NodeId,
                d.FeatureName,
                d.Threshold,
                d.LeftChild,
                d.RightChild,
                d.label,
                test.row
            FROM
                training as d, ForestDecisionPaths as dp, test
                WHERE 
                (
                    dp.TreeID = d.TreeID  -- Assicura che le condizioni siano applicate solo all'interno dello stesso albero
                    AND (
                        (dp.LeftChild = d.NodeID AND dp.FeatureName = test.col AND test.value <= dp.Threshold AND tr = test.row)
                        OR (dp.RightChild = d.NodeID AND dp.FeatureName = test.col AND test.value > dp.Threshold AND tr = test.row)
                    )
                )
        ), classifications AS (
            SELECT TreeID, label, tr
            FROM ForestDecisionPaths
            WHERE label IS NOT NULL
            group by TreeID, tr, label
        ), aggregation AS (
            SELECT COUNT(label) as cont, label, tr
            FROM classifications 
            GROUP BY tr, label
            ORDER BY tr ASC
        ), r AS (SELECT v.tr, v.label, ROW_NUMBER() OVER (PARTITION BY v.tr ORDER BY v.cont ASC) as Rn 
        FROM aggregation as v
        WHERE v.cont >= (SELECT max(v2.cont) FROM aggregation as v2 WHERE v2.tr = v.tr)) 
        SELECT tr, label FROM r WHERE Rn = 1'''
          

            
        
        query = text(query)
        connection = engine.connect()
        results = connection.execute(query)
        lista = []
        for tupla in results:
            secondo_elemento = int(tupla[1])
            lista.append(secondo_elemento)
        a = np.array(lista)
        return a
    
###################################################################################################################
        
class RandomForest2(RandomForestClassifier):

    def __init__(self,n_estimators):
        super().__init__(n_estimators)

    def fit(self,X,Y, conn):
        model = super().fit(X,Y)
        forest = model.estimators_
        self.n_features = X.shape[1]
        cursor = conn.cursor()


        # Creazione tabella contenente le informazioni dell'albero
        cursor.execute('''CREATE TABLE IF NOT EXISTS decision_trees (
                                TreeID INT,
                                NodeID INT,
                                FeatureName VARCHAR(255),
                                Threshold FLOAT,
                                LeftChild INT,  -- NULL se il nodo è una foglia
                                RightChild INT,  -- NULL se il nodo è una foglia
                                ClassLabel VARCHAR(255)  -- NULL se il nodo non è una foglia 
                            );''')
        
        def insert_tree(tree,i):
            def insert_rules(node, parent_id):
                    if tree.children_left[node] == tree.children_right[node]:  # Nodo foglia
                        cursor.execute('''
                            INSERT INTO decision_trees (TreeID, NodeID, FeatureName, Threshold, LeftChild, RightChild, ClassLabel)
                            VALUES (?, ?, NULL, NULL, NULL, NULL, ?)
                        ''', (int(i), int(node), str(tree.value[node].argmax())))
                        conn.commit()
                    else:
                        feature = tree.feature[node]
                        threshold = tree.threshold[node]
                        left_node = tree.children_left[node]
                        right_node = tree.children_right[node]

                        cursor.execute('''
                            INSERT INTO decision_trees (TreeID, NodeID, FeatureName, Threshold, LeftChild, RightChild, ClassLabel)
                            VALUES (?, ?, ?, ?, ?, ?, NULL)
                        ''', (int(i), int(node), str(feature), float(threshold), int(left_node), int(right_node)))
                        conn.commit()
                        insert_rules(left_node, node)
                        insert_rules(right_node, node)   
            insert_rules(0, -1)

        for i,tree in enumerate(forest):
            insert_tree(tree.tree_,i)


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

    def clear_data(self,conn):
        cursor=conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS test")
        cursor.execute("DROP TABLE IF EXISTS decision_tree")
    
    def predict(self,conn):
    
        cursor = conn.cursor()
        
        query ='''
            WITH RECURSIVE ForestDecisionPaths AS (
                SELECT
                    TreeID,
                    NodeID AS CurrentNode,
                    FeatureName,
                    Threshold,
                    LeftChild,
                    RightChild,
                    ClassLabel,
                    test.id as tr
                FROM
                    decision_trees, test
                WHERE
                    NodeID = 0  -- Inizia dalla radice dell'albero

                UNION ALL

                SELECT
                    d.TreeID,
                    d.NodeId,
                    d.FeatureName,
                    d.Threshold,
                    d.LeftChild,
                    d.RightChild,
                    d.ClassLabel,
                    test.id
                FROM
                    decision_trees AS d, test
                JOIN
                    ForestDecisionPaths AS dp ON
                    (
                        dp.TreeID = d.TreeID AND ((
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
                    )))
            ),classifications AS (
            SELECT TreeID, ClassLabel, tr
            FROM ForestDecisionPaths
            WHERE ClassLabel IS NOT NULL
            group by TreeID, tr
        ), aggregation AS (
            SELECT COUNT(ClassLabel) as cont, ClassLabel, tr
            FROM classifications 
            GROUP BY tr, ClassLabel
            ORDER BY tr ASC
        ), r AS (SELECT v.tr, v.ClassLabel, ROW_NUMBER() OVER (PARTITION BY v.tr ORDER BY v.cont ASC) as Rn 
        FROM aggregation as v
        WHERE v.cont >= (SELECT max(v2.cont) FROM aggregation as v2 WHERE v2.tr = v.tr)) 
        SELECT tr, ClassLabel FROM r WHERE Rn = 1'''
        
        results = cursor.execute(query).fetchall()
        return np.array(results)

    


'''classifications AS (
            SELECT TreeID, label, tr
            FROM ForestDecisionPaths
            WHERE label IS NOT NULL
            group by TreeID, tr
        ), aggregation AS (
            SELECT COUNT(label) as cont, label, tr
            FROM classifications 
            GROUP BY tr, label
            ORDER BY tr ASC
        ), r AS (SELECT v.tr, v.label, ROW_NUMBER() OVER (PARTITION BY v.tr ORDER BY v.cont ASC) as Rn 
        FROM aggregation as v
        WHERE v.cont >= (SELECT max(v2.cont) FROM aggregation as v2 WHERE v2.tr = v.tr)) 
        SELECT tr, label FROM r WHERE Rn = 1'''