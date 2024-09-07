# Obiettivi del Lavoro

L'obiettivo principale del lavoro era quello di integrare modelli di machine learning all'interno di un ambiente SQL, garantendo precisione nella predizione e buone prestazioni. Questo avrebbe consentito di eseguire analisi predittive direttamente sui dati memorizzati nel database, ottenendo i seguenti vantaggi:

- **Calcolo vicino ai dati**: evitando l'estrazione dei dati e risparmiando sui costi associati.
- **Capacità dei moderni sistemi DBMS**: gestione ed elaborazione di grandi volumi di dati per migliorare efficienza e scalabilità.
- **Portabilità dei modelli**: l'utilizzo di linguaggi standard, adottati dalla maggior parte dei DBMS relazionali, garantisce la definizione della logica inferenziale degli algoritmi di machine learning su diversi sistemi.

# Descrizione del Lavoro Svolto

Ho progettato e implementato un pacchetto Python che fornisce metodi di Machine Learning integrati in SQL. In particolare, sono stati integrati quattro algoritmi di Machine Learning, ciascuno con un modulo specifico:

- **K-nearest neighbors (KNN)**
- **Decision tree**
- **Random forests**
- **Logistic regression**

Per ogni algoritmo, sono state sviluppate due classi per rappresentazioni dei dati differenti:

- **Relazionale**: per dati densi.
- **Coordinate Format List (COO)**: per dati sparsi.

Le classi sviluppate consentono di addestrare modelli di machine learning usando la libreria scikit-learn in Python e di eseguire previsioni direttamente tramite query SQL sui dati presenti nel database. In sintesi, il pacchetto consente di sfruttare le potenzialità del Machine Learning in un ambiente SQL, aprendo nuove possibilità per l'analisi e la previsione dei dati nel contesto dei database relazionali.

# Tecnologie Coinvolte

Durante lo stage, ho lavorato con le seguenti tecnologie:

- **Python**: linguaggio principale utilizzato per lo sviluppo del pacchetto, grazie alla sua versatilità e alle numerose librerie disponibili.
- **Librerie scikit-learn e SQLAlchemy**:
  - **scikit-learn**: utilizzata per implementare gli algoritmi di Machine Learning, fornendo strumenti per l'addestramento e la valutazione dei modelli.
  - **SQLAlchemy**: utilizzata per interagire con il database SQL, consentendo query efficienti e sicure.
- **SQL**: utilizzato per eseguire previsioni direttamente all'interno del database, garantendo sicurezza e riducendo i tempi di elaborazione. La scrittura di query SQL complesse è stata essenziale per implementare gli algoritmi di Machine Learning e generare previsioni accurate.
- **DBMS**: ho lavorato con due sistemi di gestione database:
  - **PostgreSQL**: un DBMS open-source, noto per la sua affidabilità, scalabilità e capacità di gestire grandi volumi di dati.
  - **SQLite**: un DBMS leggero e serverless, utile per lo sviluppo locale e il testing.

L'utilizzo combinato di queste tecnologie ha permesso lo sviluppo di un pacchetto efficiente per eseguire analisi predittive direttamente sui dati memorizzati in un database SQL.
