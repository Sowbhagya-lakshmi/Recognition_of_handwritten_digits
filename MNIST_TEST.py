from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
import numpy as np
import cv2

mnist = keras.datasets.mnist
(train_inputs,train_targets), (test_inputs,test_targets) = mnist.load_data()

normalised_test_inputs = test_inputs/255

model = keras.models.load_model('Model/mnist_model.hdf5')

#model.evaluate(normalised_test_inputs, test_targets) -- > NOT WORKING

# Calculating accuracy
predictions = model.predict_classes(normalised_test_inputs)
correct = predictions == test_targets
correct = correct.astype(np.uint8)
sum = correct.sum()
accuracy = sum/len(correct)

print('Accuracy: ', accuracy)


# Printing out images and testing
for i in range(20):
    img = test_inputs[i]

    print('\npredicting')
    result = model.predict(img.reshape(1,28,28))

    x = max(result[0])
    
    for index, element in enumerate(result[0]):
        if element == x:
            print(index)

    cv2.imshow('img', img)
    cv2.waitKey(0)

# Achieved accuracy of 0.9493

