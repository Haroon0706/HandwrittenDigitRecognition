# Import 4 essential libraries

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import tensorflow as tf

# HYPERPARAMS
OPTIMIZER = "adam"
EPOCHS = 3
LOSS ="sparse_categorical_crossentropy"

""" First step is to load the dataset of the handwritten digits 
Using the MNIST dataset wohich consists of around 60,000 images of handwritten digits
"""

mnist = tf.keras.datasets.mnist

# Split into training and testing data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise the data, scaling it down between 0 and 1

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Next step is to define the model, an input layer, 2 hidden layers and an output layer

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Model is built with a flattened input layer, 2 dense "hidden" layers and an output layer with the softmax function

# Compiling the model

model.compile(optimizer=OPTIMIZER, loss= LOSS, metrics=["accuracy"])

# Fit the model and train the neural network

model.fit(x_train, y_train, epochs=EPOCHS)

# Evaluate the model

accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

# Save the model so testing does not need to be done over and over

model.save(f"optimizer_{OPTIMIZER}_epochs_{EPOCHS}_loss_{LOSS}.model")

# Next step is to read in test data created in Microsoft paint using OpenCV
# Model prediction of imported hand written paint digits

for x in range(1,6):
    img = cv.imread(f"{x}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The result is probably: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

# Next step is to create a GUI where digits can be drawn as paint png files are of poor quality giving less than desired results
