# -*- coding: utf-8 -*-

#import paquetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#cargando datos
dataset = pd.read_csv("salario.csv");
x=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,1].values;

#separando datos en set de entrenamiento y test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state=0)

#adaptacion de una regresion
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train, y_train)

#predicio resultados del set de prueba
y_pred = regresor.predict(x_test)

#visualizando resultados set de entrenamiento
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regresor.predict(x_train),color='blue')
plt.title("Salario vs Años de experiencia(set entrenamiento")
plt.xlabel("años de experiencia")
plt.ylabel("Salario")
plt.show()

#visualizando resultados set de prueba
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regresor.predict(x_train),color='blue')
plt.title("Salario vs Años de experiencia(set entrenamiento")
plt.xlabel("años de experiencia")
plt.ylabel("Salario")
plt.show()