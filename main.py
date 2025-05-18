import tensorflow as tf 
from tensorflow import keras as k 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys
print(sys.executable)

#okk

data = pd.read_csv('diabetes.csv')
#print(data.head())

x = data.drop('Outcome', axis=1)
y = data['Outcome']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)



print(x.head())
print(y.head())

import matplotlib.pyplot as plt

plt.scatter(x.iloc[:, 6], x.iloc[:, 7], c=y, cmap='bwr')
plt.xlabel(x.columns[6])
plt.ylabel(x.columns[7])
plt.title('Diabetes Data: Feature 1 vs Feature 2')
plt.show()



from tensorflow import keras as k 
model = k.Sequential([
                    k.layers.Dense(16, activation = k.activations.relu, input_shape = (8,)),
                    k.layers.Dense(16, activation = k.activations.relu),
                    k.layers.Dense(1, activation = k.activations.sigmoid)
])
model.summary()
k.utils.plot_model(model, show_shapes = True)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)




epochs = 100 # addestra il modello - ha passato 100 volte l'intero dataset X per imparare
model.fit(x,y,epochs=epochs) # x: dataset input, y: dataset output

plt.figure(figsize=(10,5))
plt.plot(range(epochs), model.history.history['loss']) # asse x epoche e asse y loss (funzione di perdita)
plt.plot(range(epochs), model.history.history['accuracy'], label='Accuracy')
plt.grid(True)
plt.savefig('loss_plot.png')  # salva nella cartella corrente con questo nome

plt.show()


y_pred = model.predict(x)
print(y_pred)

y_pred_class = np.where(y_pred[:,0] >= 0.5, 1,0)
print(y_pred_class)

k.metrics.binary_accuracy(y, y_pred_class) # questa riga calcola accuratezza, altre x visualizzazione
acc = k.metrics.binary_accuracy(y, y_pred_class)
print(f"Accuratezza: {acc.numpy() * 100:.2f}%")