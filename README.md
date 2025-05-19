# Rete Neurale per la Classificazione del Diabete

Questo script implementa una rete neurale semplice usando TensorFlow/Keras per prevedere la probabilità che un paziente abbia il diabete, basandosi su dati clinici. I dati utilizzati provengono da Pima Indians Diabetes Dataset.

Ogni riga rappresenta un paziente, e ogni colonna è una caratteristica medica (es. pressione, BMI, insulina, ecc.), mentre la colonna Outcome rappresenta la diagnosi non diabetico o  diabetico tramite 0 e 1. 

Viene utilizzata una rete composta da due layer con 16 neuroni ciascuno e funzione di attivazione ReLu, e un layer di output con funzione di attivazione Sigmoid per la classificazione binaria. Il modello viene addestrato per 100 epoche, se la probabilità è >= 0.5, il paziente viene classificato come diabetico, non diabetico altrimenti. 

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 16)                144       
                                                                 
 dense_1 (Dense)             (None, 16)                272       
                                                                 
 dense_2 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 433
Trainable params: 433
Non-trainable params: 0



Epoch 1/100
24/24 [==============================] - 1s 2ms/step - loss: 0.6540 - accuracy: 0.6732
Epoch 2/100
24/24 [==============================] - 0s 2ms/step - loss: 0.6084 - accuracy: 0.7161
Epoch 3/100
24/24 [==============================] - 0s 3ms/step - loss: 0.5716 - accuracy: 0.7318
Epoch 4/100
24/24 [==============================] - 0s 6ms/step - loss: 0.5437 - accuracy: 0.7357
Epoch 5/100
24/24 [==============================] - 0s 3ms/step - loss: 0.5215 - accuracy: 0.7487
Epoch 6/100
24/24 [==============================] - 0s 4ms/step - loss: 0.5052 - accuracy: 0.7552
Epoch 7/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4932 - accuracy: 0.7578
Epoch 8/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4839 - accuracy: 0.7682
Epoch 9/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4762 - accuracy: 0.7669
Epoch 10/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4709 - accuracy: 0.7721
Epoch 11/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4657 - accuracy: 0.7773
Epoch 12/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4614 - accuracy: 0.7695
Epoch 13/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4576 - accuracy: 0.7786
Epoch 14/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4540 - accuracy: 0.7760
Epoch 15/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4510 - accuracy: 0.7773
Epoch 16/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4488 - accuracy: 0.7786
Epoch 17/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4464 - accuracy: 0.7812
Epoch 18/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4443 - accuracy: 0.7812
Epoch 19/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4425 - accuracy: 0.7826
Epoch 20/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4413 - accuracy: 0.7799
Epoch 21/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4397 - accuracy: 0.7839
Epoch 22/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4379 - accuracy: 0.7839
Epoch 23/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4364 - accuracy: 0.7865
Epoch 24/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4351 - accuracy: 0.7891
Epoch 25/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4337 - accuracy: 0.7865
Epoch 26/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4328 - accuracy: 0.7878
Epoch 27/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4312 - accuracy: 0.7917
Epoch 28/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4301 - accuracy: 0.7904
Epoch 29/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4287 - accuracy: 0.7891
Epoch 30/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4274 - accuracy: 0.7917
Epoch 31/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4268 - accuracy: 0.7943
Epoch 32/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4258 - accuracy: 0.7904
Epoch 33/100
24/24 [==============================] - 0s 4ms/step - loss: 0.4243 - accuracy: 0.7904
Epoch 34/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4230 - accuracy: 0.7930
Epoch 35/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4217 - accuracy: 0.7930
Epoch 36/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4209 - accuracy: 0.7878
Epoch 37/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4196 - accuracy: 0.7917
Epoch 38/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4191 - accuracy: 0.7917
Epoch 39/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4182 - accuracy: 0.7904
Epoch 40/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4176 - accuracy: 0.7917
Epoch 41/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4169 - accuracy: 0.7969
Epoch 42/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4149 - accuracy: 0.7982
Epoch 43/100
24/24 [==============================] - 0s 3ms/step - loss: 0.4144 - accuracy: 0.7943
Epoch 44/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4146 - accuracy: 0.7930
Epoch 45/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4132 - accuracy: 0.7956
Epoch 46/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4125 - accuracy: 0.7956
Epoch 47/100
24/24 [==============================] - 0s 7ms/step - loss: 0.4112 - accuracy: 0.7995
Epoch 48/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4104 - accuracy: 0.7969
Epoch 49/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4110 - accuracy: 0.7995
Epoch 50/100
24/24 [==============================] - 0s 5ms/step - loss: 0.4092 - accuracy: 0.7982
Epoch 51/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4084 - accuracy: 0.8060
Epoch 52/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4072 - accuracy: 0.8047
Epoch 53/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4070 - accuracy: 0.8034
Epoch 54/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4060 - accuracy: 0.8060
Epoch 55/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4050 - accuracy: 0.8047
Epoch 56/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4051 - accuracy: 0.8034
Epoch 57/100
24/24 [==============================] - 0s 5ms/step - loss: 0.4036 - accuracy: 0.8086
Epoch 58/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4038 - accuracy: 0.8047
Epoch 59/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4028 - accuracy: 0.8047
Epoch 60/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4021 - accuracy: 0.8073
Epoch 61/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4016 - accuracy: 0.8099
Epoch 62/100
24/24 [==============================] - 0s 2ms/step - loss: 0.4011 - accuracy: 0.8073
Epoch 63/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3999 - accuracy: 0.8138
Epoch 64/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3993 - accuracy: 0.8099
Epoch 65/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3987 - accuracy: 0.8138
Epoch 66/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3987 - accuracy: 0.8151
Epoch 67/100
24/24 [==============================] - 0s 3ms/step - loss: 0.3978 - accuracy: 0.8073
Epoch 68/100
24/24 [==============================] - 0s 4ms/step - loss: 0.3961 - accuracy: 0.8151
Epoch 69/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3961 - accuracy: 0.8112
Epoch 70/100
24/24 [==============================] - 0s 4ms/step - loss: 0.3955 - accuracy: 0.8190
Epoch 71/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3938 - accuracy: 0.8125
Epoch 72/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3932 - accuracy: 0.8177
Epoch 73/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3933 - accuracy: 0.8138
Epoch 74/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3923 - accuracy: 0.8138
Epoch 75/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3920 - accuracy: 0.8138
Epoch 76/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3913 - accuracy: 0.8138
Epoch 77/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3910 - accuracy: 0.8164
Epoch 78/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3903 - accuracy: 0.8151
Epoch 79/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3893 - accuracy: 0.8164
Epoch 80/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3886 - accuracy: 0.8203
Epoch 81/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3879 - accuracy: 0.8177
Epoch 82/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3891 - accuracy: 0.8125
Epoch 83/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3866 - accuracy: 0.8164
Epoch 84/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3883 - accuracy: 0.8164
Epoch 85/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3854 - accuracy: 0.8203
Epoch 86/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3848 - accuracy: 0.8177
Epoch 87/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3842 - accuracy: 0.8203
Epoch 88/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3834 - accuracy: 0.8164
Epoch 89/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3829 - accuracy: 0.8203
Epoch 90/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3825 - accuracy: 0.8151
Epoch 91/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3825 - accuracy: 0.8177
Epoch 92/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3816 - accuracy: 0.8216
Epoch 93/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3807 - accuracy: 0.8216
Epoch 94/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3793 - accuracy: 0.8177
Epoch 95/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3787 - accuracy: 0.8190
Epoch 96/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3785 - accuracy: 0.8177
Epoch 97/100
24/24 [==============================] - 0s 3ms/step - loss: 0.3779 - accuracy: 0.8190
Epoch 98/100
24/24 [==============================] - 0s 5ms/step - loss: 0.3777 - accuracy: 0.8203
Epoch 99/100
24/24 [==============================] - 0s 3ms/step - loss: 0.3766 - accuracy: 0.8216
Epoch 100/100
24/24 [==============================] - 0s 2ms/step - loss: 0.3759 - accuracy: 0.8190


Visualizzazione delle y predette dal modello convertite in classi 0/1 tramite np.where():

[1 0 1 0 1 0 0 1 1 0 0 1 0 1 1 0 1 0 0 0 1 0 1 0 1 0 1 0 0 0 0 1 0 0 0 0 1
 1 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 1 0
 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1
 1 0 0 1 1 1 0 0 0 1 0 0 0 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
 1 0 1 0 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 0 1
 1 1 1 0 0 0 1 1 1 0 1 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 1 1 0 0 0 1 1 1
 0 0 0 0 0 1 0 0 1 1 0 1 0 1 1 1 1 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1
 1 1 1 0 1 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 0 1 1 0 1
 1 0 1 0 1 1 0 1 1 0 1 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 0 1 0 0 1 1 0 0 0 0 1
 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 0 1 0 0 0 0 0 1
 1 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 1 0
 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 0
 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 1
 0 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 1 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 1 0 1 1 0 0 0 0 0
 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 0 1 0 1 0
 1 0 1 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 0 0 0 1
 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 1 1 1 0 0 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0
 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 1 1 1 0 1 1 0 0 1 0 0 1
 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1
 1 0 0 1 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 1 0 1 0 0 0 0 1 0]


Dopo l'addestramento il modello ha raggiunto una Loss (funzione di perdita) = 0.37 e una accuracy dell'82.29% sull'intero dataset. 

## Contenuto

- `model_diabetes.py`: script principale per addestramento e valutazione
- `diabetes.csv`: dataset usato
- `loss_plot.png`: grafico di accuratezza e perdita

## Esecuzione

```bash
python model_diabetes.py
