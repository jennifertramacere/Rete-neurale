import tensorflow as tf 
from tensorflow import keras as k 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys
print(sys.executable)



data = pd.read_csv('diabetes.csv')
print(data.head())

x = data.drop('Outcome', axis=1)
y = data['Outcome']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)



print(pd.DataFrame(x).head())
print(y.head())

# Viene creato uno scatter plot tra due feature per osservare come i pazienti si distribuiscono rispetto all’Outcome

x_df = pd.DataFrame(x, columns=data.drop('Outcome', axis=1).columns)
plt.scatter(x_df['BMI'], x_df['Age'], c=y, cmap='bwr')
plt.xlabel(x_df.columns[6])
plt.ylabel(x_df.columns[7]) 
plt.title('Diabetes Data: Feature 1 vs Feature 2')
plt.show()


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


epochs = 100 # addestra il modello - ha passato 100 volte l'intero dataset x per imparare
history = model.fit(x,y,epochs=epochs) # x: dataset input, y: dataset output
plt.figure(figsize=(10,5))
plt.plot(range(epochs), model.history.history['loss']) # asse x epoche e asse y loss (funzione di perdita)
plt.plot(range(epochs), model.history.history['accuracy'], label='Accuracy')
plt.grid(True)
plt.savefig('loss_plot.png')  
plt.show()


y_pred = model.predict(x)

y_pred_class = np.where(y_pred[:,0] >= 0.5, 1,0)
print(y_pred_class)

k.metrics.binary_accuracy(y, y_pred_class) # calcola accuratezza
acc = k.metrics.binary_accuracy(y, y_pred_class)
print(f"Accuratezza: {acc.numpy() * 100:.2f}%")