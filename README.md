# Integrazione-moduli-di-machine-learning-nei-DBMS

 
 # Obiettivi del lavoro
 L'obiettivo principale del lavoro era quello di integrare modelli di machine learning,
 all'interno di un ambiente SQL, garantendo precisione nella predizione e buone prestazioni.
 Questo avrebbe consentito di eseguire analisi predittive direttamente sui dati memorizzati nel
 database, ottenendo i seguenti vantaggi: il calcolo vicino ai dati, che evita l'estrazione dei dati
 e risparmia sui costi associati, le capacità dei moderni sistemi DBMS nella gestione ed
 elaborazione di grandi volumi di dati, al fine di migliorare l'efficienza e la scalabilità dei
 calcoli, la portabilità dei modelli su diversi sistemi DBMS ottenuta che si intende perseguire
 utilizzando il linguaggio standard adottato da maggior parte dei sistemi dbms relazionali per
 la definizione della logica inferenziale degli algoritmi di machine learning.
 
 # Descrizione lavoro svolto
 Ho lavorato alla progettazione e implementazione di un pacchetto Python che fornisca metodi
 di Machine Learning integrati in SQL. In particolare sono stati integrati quattro algoritmi di
 Machine Learning, ad ognuno è stato dedicato un modulo specifico:
 1) K-nearest neighbors (KNN)
 2) Decision tree
 3) Random forests
 4) Logistic regression
<p>Per ogni algoritmo sono state fornite due classi, per due rappresentazioni dei dati differenti,
 relazionale per i dati densi e Coordinat Format List(COO) per i dati sparsi.
 Le classi sviluppate permettono di addestrare modelli di machine learning usando la libreria
 scikit-learn in Python e di eseguire la previsione direttamente attraverso query SQL sui dati
 presenti nel database.
 In sintesi, il pacchetto sviluppato consente di sfruttare le potenzialità del Machine Learning in
 un ambiente SQL, aprendo nuove possibilità per l'analisi e la previsione dei dati direttamente
 nel contesto dei database relazionali. </p>
 
 # Tecnologie coinvolte
 Durante lo stage, ho lavorato principalmente con le seguenti tecnologie:
 Linguaggio di programmazione Python: è stato il linguaggio principale utilizzato per lo
 sviluppo del pacchetto. La sua versatilità e la vasta gamma di librerie disponibili lo rendono
 una scelta ideale per implementare algoritmi di Machine Learning e interagire con database.
 Librerie scikit-learn e SQLAlchemy: La libreria scikit-learn è stata fondamentale per
 implementare gli algoritmi di Machine Learning, offrendo un'ampia gamma di algoritmi e
 funzionalità per l'addestramento dei modelli e la valutazione delle prestazioni. SQLAlchemy,
 d'altra parte, è stata utilizzata per interagire con il database SQL, consentendo di eseguire
 query e manipolare dati in modo efficiente e sicuro.
 SQL: Il linguaggio SQL è stato al centro del progetto. SQL è stato utilizzato principalmente
 per eseguire la predizione dei dati direttamente all'interno del database. Le query SQL sono
 state scritte in modo da calcolare le previsioni utilizzando i modelli di Machine Learning
 addestrati ed integrati nel database stesso. Questo approccio ha consentito di eseguire analisi
 predittive senza dover estrarre i dati dal database, garantendo una maggiore sicurezza e
 riducendo i tempi di elaborazione. La comprensione di SQL è stata quindi essenziale per
 scrivere query complesse che implementassero gli algoritmi di Machine Learning e
 generassero previsioni accurate.
 DBMS: Holavorato con due sistemi di gestione di database: PostgreSQL e SQLite.
 PostgreSQL è un DBMS open-source ampiamente utilizzato, noto per la sua affidabilità,
 scalabilità e capacità di gestire grandi volumi di dati. SQLite, d'altra parte, è un DBMS
 leggero e serverless che offre una soluzione locale per lo sviluppo e il testing, ed è spesso
 integrato direttamente nelle applicazioni.
 Complessivamente, l'utilizzo di queste tecnologie ha reso possibile lo sviluppo di un
 pacchetto che offre un'interfaccia semplice ed efficiente per eseguire analisi predittive
 direttamente sui dati memorizzati in un database SQL
