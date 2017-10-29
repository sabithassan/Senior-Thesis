import numpy as np
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
 
batch_size = 2000 
nb_classes = 10 
nb_epoch = 20
 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
#Flatten the each image into 1D array
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
 
#Make the value floats in [0;1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
 
# convert class vectors to binary class matrices (ie one-hot vectors)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
 
#model achitecture
model = Sequential()

## level 1
model.add(Dense(784, input_shape=(784,)))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

## level 2
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

## level 3
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

## final level
model.add(Dense(10)) 
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

## optimizier
rms = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

#cross entropy
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
 
#train the model
model.fit(X_train, Y_train,
batch_size=batch_size, nb_epoch=nb_epoch,
verbose=2,
validation_data=(X_test, Y_test))
 
#Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
 
print('Test score:', score[0])
print('Test accuracy:', score[1])
