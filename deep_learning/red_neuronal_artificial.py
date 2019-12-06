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
y_predic = classifier.predict(x_test)

#
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

