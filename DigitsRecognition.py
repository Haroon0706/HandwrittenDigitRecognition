# Import 4 essential libraries

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

model.compile(optimizer="adam", loss="sparse_categorical_crossentopy", metrics=["accuracy"])

