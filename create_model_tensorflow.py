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
from keras.callbacks import EarlyStopping, ModelCheckpoint

# redefine target data into one hot vector
classes = 156
y_train = keras.utils.to_categorical(y_train, classes)

def cba(inputs, filters, kernel_size, strides):
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# define CNN
inputs = Input(shape=(x_train.shape[1:]))
inputs = Reshape((12, 1), input_shape = (x_train.shape[1:]))(inputs)

x_1 = cba(inputs, filters=32, kernel_size=(8), strides=(2))
x_1 = cba(x_1, filters=32, kernel_size=(1), strides=(1))
x_1 = cba(x_1, filters=64, kernel_size=(8), strides=(2))
x_1 = cba(x_1, filters=64, kernel_size=(1), strides=(1))

x_2 = cba(inputs, filters=32, kernel_size=(16), strides=(2))
x_2 = cba(x_2, filters=32, kernel_size=(1), strides=(1))
x_2 = cba(x_2, filters=64, kernel_size=(16), strides=(2))
x_2 = cba(x_2, filters=64, kernel_size=(1), strides=(1))

x_3 = cba(inputs, filters=32, kernel_size=(32), strides=(2))
x_3 = cba(x_3, filters=32, kernel_size=(1), strides=(1))
x_3 = cba(x_3, filters=64, kernel_size=(32), strides=(2))
x_3 = cba(x_3, filters=64, kernel_size=(1), strides=(1))

x_4 = cba(inputs, filters=32, kernel_size=(64), strides=(2))
x_4 = cba(x_4, filters=32, kernel_size=(1), strides=(1))
x_4 = cba(x_4, filters=64, kernel_size=(64), strides=(2))
x_4 = cba(x_4, filters=64, kernel_size=(1), strides=(1))

x = Add()([x_1, x_2, x_3, x_4])

x = cba(x, filters=128, kernel_size=(16), strides=(2))
x = cba(x, filters=128, kernel_size=(1), strides=(1))

x = GlobalAveragePooling1D()(x)
x = Dense(classes)(x)
x = Activation("softmax")(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = tensorflow.keras.optimizers.Nadam(lr=0.00001, decay=1e-6)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

print("start learning...")

history = model.fit(x = x_train, y = y_train, epochs = 2000)

import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.show()

print("Learning Finished!")