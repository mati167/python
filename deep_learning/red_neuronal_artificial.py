# -*- coding: utf-8 -*-
#import paquetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#cargando datos
dataset = pd.read_csv("modelo_bajas_banco.csv");
x=dataset.iloc[:,3:13].values;
y=dataset.iloc[:,13];

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

#separando datos en set de entrenamiento y test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

#escalado de caracteristicas
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#importando  keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#inicializando secuancias
clasificador = Sequential()

#agregamos primera capa input
clasificador.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
clasificador.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#
clasificador.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#
clasificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
clasificador.fit(x_train,y_train,batch_size=10,epochs=100)

#
y_pred = clasificador.predict(x_test)
y_pred = (y_pred>0.5)

#
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Prediciendo Nuevo Cliente
"""
Geografía: Francia
Puntaje de crédito: 600
Género: Masculino
Edad: 40 años
Tenencia: 3 años
Saldo: $60000
Número de productos: 2
¿Tiene este cliente una tarjeta de crédito? Sí
¿Este cliente es un miembro activo? Si
Salario estimado: $50000
"""

nuevo_cliente = clasificador.predict(sc_x.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
nuevo_cliente = (nuevo_cliente>0.5)


