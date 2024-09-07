# Project Objectives

The main objective of the project was to integrate machine learning models within an SQL environment, ensuring prediction accuracy and high performance. This approach would enable predictive analysis directly on the data stored in the database, providing the following benefits:

- **Proximity to the data**: avoiding data extraction and reducing associated costs.
- **Capabilities of modern DBMS**: handling and processing large volumes of data to improve efficiency and scalability.
- **Model portability**: the use of standard languages, adopted by most relational DBMS, ensures the inferential logic of machine learning algorithms can be defined across different systems.

# Work Description

I designed and implemented a Python package that provides machine learning methods integrated into SQL. Specifically, four machine learning algorithms were integrated, each with its own dedicated module:

- **K-nearest neighbors (KNN)**
- **Decision tree**
- **Random forests**
- **Logistic regression**

For each algorithm, two classes were developed to support different data representations:

- **Relational**: for dense data.
- **Coordinate Format List (COO)**: for sparse data.

These classes allow the training of machine learning models using Python's scikit-learn library and enable predictions to be made directly through SQL queries on data within the database. In essence, the package unlocks the potential of machine learning in an SQL environment, opening up new possibilities for data analysis and prediction within relational databases.

# Technologies Used

During the project, I worked with the following technologies:

- **Python**: the primary language used to develop the package, chosen for its versatility and wide range of available libraries.
- **scikit-learn and SQLAlchemy libraries**:
  - **scikit-learn**: used to implement machine learning algorithms, offering tools for model training and performance evaluation.
  - **SQLAlchemy**: used to interact with the SQL database, enabling efficient and secure query execution.
- **SQL**: used to perform predictions directly within the database, ensuring data security and reducing processing times. Writing complex SQL queries was essential for implementing machine learning algorithms and generating accurate predictions.
- **DBMS**: I worked with two database management systems:
  - **PostgreSQL**: an open-source DBMS known for its reliability, scalability, and ability to manage large datasets.
  - **SQLite**: a lightweight, serverless DBMS, useful for local development and testing.

The combined use of these technologies enabled the development of an efficient package for performing predictive analytics directly on data stored in an SQL database.
