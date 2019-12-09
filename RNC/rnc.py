# -*- coding: utf-8 -*-

#PRE PROCESAMIENTO DE IMAGENES

#importando keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import flatten

#CREANDO EL MODELO

#inicializando rnc
clasificador = Sequential()

#paso 1 - Convolucion
clasificador.add(Conv2D(input_shape=(64,64,3),filters=32,kernel_size=3,strides=3,activation='relu'))

#paso 2 - Agrupacion
clasificador.add(MaxPooling2D(pool_size=(2,2)))

#paso 2.5 capa extra
clasificador.add(Conv2D(filters=32,kernel_size=3,strides=3,activation='relu'))
clasificador.add(MaxPooling2D(pool_size=(2,2)))

#paso 3 - aplanamiento
clasificador.add(flatten)

#paso 4 - conexion completa
clasificador.add(Dense(units=128,activation='relu'))
clasificador.add(Dense(units=1,activation='sigmoid')) #capa salida

#paso 5 - compilacion
clasificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#PARTE 2 - Encajando RNC en imagenes
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

clasificador.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)