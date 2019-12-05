# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:54:39 2019

@author: Matias Gonzalez
"""

#import paquetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#cargando datos
dataset = pd.read_csv("Data.csv");
x=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,3];

#separando datos en set de entrenamiento y test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

#escalado de caracteristicas
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
'''