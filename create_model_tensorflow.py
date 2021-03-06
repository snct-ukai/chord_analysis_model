import numpy as np

x_train = np.load("./tmp/numpy/X_train.npy")
y_train = np.load("./tmp/numpy/Y_train.npy")

print("X_train shape:", x_train.shape)
print("Y_train shape:", y_train.shape)


import tensorflow
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape
from keras.layers import Conv1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Add
import tensorflow_addons as tfa

# redefine target data into one hot vector
classes = 156
y_train = keras.utils.to_categorical(y_train, classes)

def cba(inputs, filters, kernel_size, strides):
  x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
  x = Dropout(0.4)(x)
  x = BatchNormalization()(x)
  x = Activation(tfa.activations.rrelu)(x)
  return x

# define CNN
inputs = Input(shape=(x_train.shape[1:]))
inputs = Reshape((12, 1), input_shape = (x_train.shape[1:]))(inputs)

def layer_1(inputs):
  x_1 = cba(inputs, filters=32, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=32, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=64, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=64, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=128, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=128, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=256, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=256, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=32, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=32, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=64, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=64, kernel_size=(1), strides=(1))
  x_2 = cba(x_1, filters=128, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=128, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=256, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=256, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=32, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=32, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=64, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=64, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=128, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=128, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=256, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=256, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=32, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=32, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=64, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=64, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=128, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=128, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=256, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=256, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=32, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=32, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=64, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=64, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=128, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=128, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=256, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=256, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5])
  return x

x = layer_1(inputs)
x = Dropout(0.4)(x)

x = Activation(tfa.activations.rrelu)(x)

x_1 = cba(x, filters=512, kernel_size=(5), strides=(1))
x_2 = cba(x, filters=512, kernel_size=(4), strides=(1))
x_3 = cba(x, filters=512, kernel_size=(3), strides=(1))
x_4 = cba(x, filters=512, kernel_size=(2), strides=(1))
x_5 = cba(x, filters=512, kernel_size=(1), strides=(1))

x = Add()([x_1, x_2, x_3, x_4, x_5])
x = Activation(tfa.activations.rrelu)(x)

x = cba(x, filters=1024, kernel_size=(11), strides=(1))
x = Activation(tfa.activations.rrelu)(x)

x = cba(x, filters=512, kernel_size=(11), strides=(1))
x = Activation(tfa.activations.rrelu)(x)

x = layer_1(x)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.4)(x)
x = Dense(156, activation=tfa.activations.rrelu)(x)
x = Dense(156, activation=tfa.activations.sparsemax)(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = tensorflow.keras.optimizers.Adam(lr=0.00001, decay=1e-6, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

print("start learning...")

history = model.fit(x = x_train, y = y_train, epochs = 25000)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(history.history['loss'], color='blue')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax = fig.add_subplot(1, 2, 2)
ax.plot(history.history['accuracy'], color='red')
ax.set_title('Accuracy')
ax.set_xlabel('Epoch')
print("Learning Finished!")

plt.show()

model.save("./tmp/models/model.h5")
