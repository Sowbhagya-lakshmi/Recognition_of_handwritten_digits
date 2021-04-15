from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
import numpy as np
import cv2

mnist = keras.datasets.mnist
(train_inputs,train_targets), (test_inputs,test_targets) = mnist.load_data()

normalised_train_inputs = train_inputs/255
normalised_test_inputs = test_inputs/255

model = keras.Sequential()

# Adding input layer
model.add(Flatten(input_shape = normalised_train_inputs.shape[1:])) 

# Hidden layer #1
model.add(Dense(28))
model.add(Activation('relu'))

# Hidden layer #2
model.add(Dense(28))
model.add(Activation('relu'))      

# Output layer
model.add(Dense(10))
model.add(Activation('softmax'))



model.compile(loss = 'sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.05), metrics = ['accuracy'])

model.fit(normalised_train_inputs, train_targets, batch_size = 1000, epochs = 10)

model.summary()

model.save('Model/mnist_model.hdf5')

# Achieved accuracy of 0.96


