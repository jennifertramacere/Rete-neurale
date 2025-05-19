# Rete-neurale

# Rete Neurale per la Classificazione del Diabete

Questo script implementa una rete neurale semplice usando TensorFlow/Keras per prevedere la probabilità che un paziente abbia il diabete, basandosi su dati clinici. I dati utilizzati provengono da Pima Indians Diabetes Dataset.

Ogni riga rappresenta un paziente, e ogni colonna è una caratteristica medica (es. pressione, BMI, insulina, ecc.), mentre la colonna Outcome rappresenta la diagnosi non diabetico o  diabetico tramite 0 e 1. 

Viene utilizzata una rete composta da due layer con 16 neuroni ciascuno e funzione di attivazione ReLu, e un layer di output con funzione di attivazione Sigmoid per la classificazione binaria. Il modello viene addestrato per 100 epoche, se la probabilità è >= 0.5, il paziente viene classificato come diabetico, non diabetico altrimenti. 

Dopo l'addestramento il modello ha raggiunto una Loss (funzione di perdita) = 0.37 e una accuracy dell'81,90% sull'intero dataset. 

## Contenuto

- `model_diabetes.py`: script principale per addestramento e valutazione
- `diabetes.csv`: dataset usato
- `loss_plot.png`: grafico di accuratezza e perdita


## Esecuzione

```bash
python model_diabetes.py
