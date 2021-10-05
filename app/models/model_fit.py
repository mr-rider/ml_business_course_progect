import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
imort ssl
import keras
from keras.datasets import cifar10

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Load the preshuffled train and tests the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


fig = plt.figure(figsize=(20, 5))
for i in range(36):
  ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
  ax.imshow(np.squeeze(x_train[i]))

plt.show()

# Rescales the images by dividing the pixel values by 255: [0, 255] => [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255



# One-hot encodes the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

# Breaks the training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Prints the shape of the training set
print('x_train shape: ', x_train.shape)

# Prints the number of training, validation, and test images
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')


# First convolutional and pooling layers.
# Note that we need to define input_shape in the first convolutional layer only.
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))

# Second convolutional and pooling layers with a ReLU activation function
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Third convolutional and pooling layers
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Dropout layer to avoid overfitting with a 30% rate
model.add(Dropout(0.3))

# Flattens the last feature map into a vector of features
model.add(Flatten())

# Adds the first fully connected layer
model.add(Dense(500, activation='relu'))
# Another dropout layer with a 40% rate
model.add(Dropout(0.4))

# The output layer is a fully connected layer with 10 nodes and softmax activation
# to give probabilities to the 10 classes.
model.add(Dense(10, activation='softmax'))

# Prints a summary of the model a the model archicture
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
                 validation_data=(x_valid, y_valid), callbacks=[checkpointer],
                 verbose=2, shuffle=True)

model.load_weights('model.weights.best.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])