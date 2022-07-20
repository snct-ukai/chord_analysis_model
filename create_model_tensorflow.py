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

def layer_1(inputs):
  x_1 = cba(inputs, filters=32, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=32, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=64, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=64, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=32, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=32, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=64, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=64, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=32, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=32, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=64, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=64, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=32, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=32, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=64, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=64, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=32, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=32, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=64, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=64, kernel_size=(1), strides=(1))

  x_6 = cba(inputs, filters=32, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=32, kernel_size=(1), strides=(1))
  x_6 = cba(x_6, filters=64, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=64, kernel_size=(1), strides=(1))

  x_7 = cba(inputs, filters=32, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=32, kernel_size=(1), strides=(1))
  x_7 = cba(x_7, filters=64, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=64, kernel_size=(1), strides=(1))

  x_8 = cba(inputs, filters=32, kernel_size=(9), strides=(1))
  x_8 = cba(x_8, filters=32, kernel_size=(1), strides=(1))
  x_8 = cba(x_8, filters=64, kernel_size=(9), strides=(1))
  x_8 = cba(x_8, filters=64, kernel_size=(1), strides=(1))

  x_9 = cba(inputs, filters=32, kernel_size=(10), strides=(1))
  x_9 = cba(x_9, filters=32, kernel_size=(1), strides=(1))
  x_9 = cba(x_9, filters=64, kernel_size=(10), strides=(1))
  x_9 = cba(x_9, filters=64, kernel_size=(1), strides=(1))

  x_10 = cba(inputs, filters=32, kernel_size=(11), strides=(1))
  x_10 = cba(x_10, filters=32, kernel_size=(1), strides=(1))
  x_10 = cba(x_10, filters=64, kernel_size=(11), strides=(1))
  x_10 = cba(x_10, filters=64, kernel_size=(1), strides=(1))

  x_11 = cba(inputs, filters=32, kernel_size=(12), strides=(1))
  x_11 = cba(x_11, filters=32, kernel_size=(1), strides=(1))
  x_11 = cba(x_11, filters=64, kernel_size=(12), strides=(1))
  x_11 = cba(x_11, filters=64, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11])
  return x

def layer_2(inputs):
  x_1 = cba(inputs, filters=16, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=16, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=32, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=32, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=16, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=16, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=32, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=32, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=16, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=16, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=32, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=32, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=16, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=16, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=32, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=32, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=16, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=16, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=32, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=32, kernel_size=(1), strides=(1))

  x_6 = cba(inputs, filters=16, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=16, kernel_size=(1), strides=(1))
  x_6 = cba(x_6, filters=32, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=32, kernel_size=(1), strides=(1))

  x_7 = cba(inputs, filters=16, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=16, kernel_size=(1), strides=(1))
  x_7 = cba(x_7, filters=32, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=32, kernel_size=(1), strides=(1))

  x_8 = cba(inputs, filters=16, kernel_size=(9), strides=(1))
  x_8 = cba(x_8, filters=16, kernel_size=(1), strides=(1))
  x_8 = cba(x_8, filters=32, kernel_size=(9), strides=(1))
  x_8 = cba(x_8, filters=32, kernel_size=(1), strides=(1))

  x_9 = cba(inputs, filters=16, kernel_size=(10), strides=(1))
  x_9 = cba(x_9, filters=16, kernel_size=(1), strides=(1))
  x_9 = cba(x_9, filters=32, kernel_size=(10), strides=(1))
  x_9 = cba(x_9, filters=32, kernel_size=(1), strides=(1))

  x_10 = cba(inputs, filters=16, kernel_size=(11), strides=(1))
  x_10 = cba(x_10, filters=16, kernel_size=(1), strides=(1))
  x_10 = cba(x_10, filters=32, kernel_size=(11), strides=(1))
  x_10 = cba(x_10, filters=32, kernel_size=(1), strides=(1))

  x_11 = cba(inputs, filters=16, kernel_size=(12), strides=(1))
  x_11 = cba(x_11, filters=16, kernel_size=(1), strides=(1))
  x_11 = cba(x_11, filters=32, kernel_size=(12), strides=(1))
  x_11 = cba(x_11, filters=32, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11])
  return x

def layer_3(inputs):
  x_1 = cba(inputs, filters=8, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=8, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=16, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=16, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=8, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=8, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=16, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=16, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=8, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=8, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=16, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=16, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=8, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=8, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=16, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=16, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=8, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=8, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=16, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=16, kernel_size=(1), strides=(1))

  x_6 = cba(inputs, filters=8, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=8, kernel_size=(1), strides=(1))
  x_6 = cba(x_6, filters=16, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=16, kernel_size=(1), strides=(1))

  x_7 = cba(inputs, filters=8, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=8, kernel_size=(1), strides=(1))
  x_7 = cba(x_7, filters=16, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=16, kernel_size=(1), strides=(1))

  x_8 = cba(inputs, filters=8, kernel_size=(9), strides=(1))
  x_8 = cba(x_8, filters=8, kernel_size=(1), strides=(1))
  x_8 = cba(x_8, filters=16, kernel_size=(9), strides=(1))
  x_8 = cba(x_8, filters=16, kernel_size=(1), strides=(1))

  x_9 = cba(inputs, filters=8, kernel_size=(10), strides=(1))
  x_9 = cba(x_9, filters=8, kernel_size=(1), strides=(1))
  x_9 = cba(x_9, filters=16, kernel_size=(10), strides=(1))
  x_9 = cba(x_9, filters=16, kernel_size=(1), strides=(1))

  x_10 = cba(inputs, filters=8, kernel_size=(11), strides=(1))
  x_10 = cba(x_10, filters=8, kernel_size=(1), strides=(1))
  x_10 = cba(x_10, filters=16, kernel_size=(11), strides=(1))
  x_10 = cba(x_10, filters=16, kernel_size=(1), strides=(1))

  x_11 = cba(inputs, filters=8, kernel_size=(12), strides=(1))
  x_11 = cba(x_11, filters=8, kernel_size=(1), strides=(1))
  x_11 = cba(x_11, filters=16, kernel_size=(12), strides=(1))
  x_11 = cba(x_11, filters=16, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11])
  return x

def layer_4(inputs):
  x_1 = cba(inputs, filters=4, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=4, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=8, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=8, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=4, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=4, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=8, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=8, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=4, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=4, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=8, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=8, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=4, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=4, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=8, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=8, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=4, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=4, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=8, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=8, kernel_size=(1), strides=(1))

  x_6 = cba(inputs, filters=4, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=4, kernel_size=(1), strides=(1))
  x_6 = cba(x_6, filters=8, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=8, kernel_size=(1), strides=(1))

  x_7 = cba(inputs, filters=4, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=4, kernel_size=(1), strides=(1))
  x_7 = cba(x_7, filters=8, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=8, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5, x_6, x_7])
  return x

def layer_5(inputs):
  x_1 = cba(inputs, filters=2, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=2, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=4, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=4, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=2, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=2, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=4, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=4, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=2, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=2, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=4, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=4, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=2, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=2, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=4, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=4, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=2, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=2, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=4, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=4, kernel_size=(1), strides=(1))

  x_6 = cba(inputs, filters=2, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=2, kernel_size=(1), strides=(1))
  x_6 = cba(x_6, filters=4, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=4, kernel_size=(1), strides=(1))

  x_7 = cba(inputs, filters=2, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=2, kernel_size=(1), strides=(1))
  x_7 = cba(x_7, filters=4, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=4, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5, x_6, x_7])
  return x

def layer_6(inputs):
  x_1 = cba(inputs, filters=1, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=1, kernel_size=(1), strides=(1))
  x_1 = cba(x_1, filters=2, kernel_size=(2), strides=(1))
  x_1 = cba(x_1, filters=2, kernel_size=(1), strides=(1))

  x_2 = cba(inputs, filters=1, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=1, kernel_size=(1), strides=(1))
  x_2 = cba(x_2, filters=2, kernel_size=(3), strides=(1))
  x_2 = cba(x_2, filters=2, kernel_size=(1), strides=(1))

  x_3 = cba(inputs, filters=1, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=1, kernel_size=(1), strides=(1))
  x_3 = cba(x_3, filters=2, kernel_size=(4), strides=(1))
  x_3 = cba(x_3, filters=2, kernel_size=(1), strides=(1))

  x_4 = cba(inputs, filters=1, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=1, kernel_size=(1), strides=(1))
  x_4 = cba(x_4, filters=2, kernel_size=(5), strides=(1))
  x_4 = cba(x_4, filters=2, kernel_size=(1), strides=(1))

  x_5 = cba(inputs, filters=1, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=1, kernel_size=(1), strides=(1))
  x_5 = cba(x_5, filters=2, kernel_size=(6), strides=(1))
  x_5 = cba(x_5, filters=2, kernel_size=(1), strides=(1))

  x_6 = cba(inputs, filters=1, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=1, kernel_size=(1), strides=(1))
  x_6 = cba(x_6, filters=2, kernel_size=(7), strides=(1))
  x_6 = cba(x_6, filters=2, kernel_size=(1), strides=(1))

  x_7 = cba(inputs, filters=1, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=1, kernel_size=(1), strides=(1))
  x_7 = cba(x_7, filters=2, kernel_size=(8), strides=(1))
  x_7 = cba(x_7, filters=2, kernel_size=(1), strides=(1))

  x = Add()([x_1, x_2, x_3, x_4, x_5, x_6, x_7])
  return x

#x = layer_1(inputs)
#x = layer_2(x)
#x = layer_3(x)
def layer(inputs):

  x_1 = layer_6(inputs)
  x_1 = Dense(156, activation='relu')(x_1)
  x_1 = layer_6(x_1)
  x_1 = Dense(156, activation='relu')(x_1)
  x_1 = layer_6(x_1)
  x_1 = Dense(156, activation='relu')(x_1)

  x_2 = layer_5(inputs)
  x_2 = Dense(156, activation='relu')(x_2)
  x_2 = layer_5(x_2)
  x_2 = Dense(156, activation='relu')(x_2)
  x_2 = layer_5(x_2)
  x_2 = Dense(156, activation='relu')(x_2)

  x_3 = layer_4(inputs)
  x_3 = Dense(156, activation='relu')(x_3)
  x_3 = layer_4(x_3)
  x_3 = Dense(156, activation='relu')(x_3)
  x_3 = layer_4(x_3)
  x_3 = Dense(156, activation='relu')(x_3)

  x = Add()([x_1, x_2, x_3])
  return x

x = layer(inputs)
x = Dense(156, activation='relu')(x)
x = layer(x)
x = Dense(156, activation='relu')(x)

x = cba(x, filters=1, kernel_size=(5), strides=(1))
x = Dense(156, activation='relu')(x)
x = cba(x, filters=1, kernel_size=(4), strides=(1))
x = Dense(156, activation='relu')(x)
x = cba(x, filters=1, kernel_size=(3), strides=(1))
x = Dense(156, activation='relu')(x)
x = cba(x, filters=1, kernel_size=(2), strides=(1))
x = Dense(156, activation='relu')(x)
x = cba(x, filters=1, kernel_size=(1), strides=(1))

x = GlobalAveragePooling1D()(x)
x = Dense(156, activation='relu')(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = tensorflow.keras.optimizers.Adam(lr=0.00001, decay=1e-6, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

print("start learning...")

history = model.fit(x = x_train, y = y_train, epochs = 20000)

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
