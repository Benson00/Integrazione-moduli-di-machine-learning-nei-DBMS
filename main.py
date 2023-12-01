from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
from Interpreter import Interpreter

def take_input():
    import sys

    # Verifica che sia stato fornito il testo come argomento da riga di comando
    if len(sys.argv) < 1:
        print("Usage: python script.py <Classifier> <data_path>")
        sys.exit(1)

    # Ottieni il testo dagli argomenti della riga di comando
    testo_input = sys.argv[1:]
    s = ""
    for word in testo_input:
        s += word +" "
    s = s[:len(s)-1]
    return s

engine = create_engine("postgresql://postgres:0698@localhost:5432/classification")

interpreter = Interpreter(engine)
results,y_test=interpreter.input(take_input())
print("Classification: ",results)
print("Test: ",y_test)
