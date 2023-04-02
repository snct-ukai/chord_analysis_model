import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Concatenate, Flatten
from keras.models import Model
import keras
import tensorflow
import numpy as np

x_train = np.load("./tmp/numpy/X_train.npy")
y_train = np.load("./tmp/numpy/Y_train.npy")

print("X_train shape:", x_train.shape)
print("Y_train shape:", y_train.shape)


# redefine target data into one hot vector
classes = 156
y_train = keras.utils.to_categorical(y_train, classes)


def cba(inputs, filters, kernel_size, strides):
    x = Conv1D(filters, kernel_size=kernel_size,
               strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# define CNN
inputs = Input(shape=(x_train.shape[1:]))
inputs = Reshape((12, 1), input_shape=(x_train.shape[1:]))(inputs)


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

    x = Add()([x_1, x_2, x_3, x_4, x_5])
    return x

#x = layer_1(inputs)

#x_1 = cba(x, filters=1, kernel_size=(5), strides=(1))
#x_2 = cba(x, filters=1, kernel_size=(4), strides=(1))
#x_3 = cba(x, filters=1, kernel_size=(3), strides=(1))
#x_4 = cba(x, filters=1, kernel_size=(2), strides=(1))
#x_5 = cba(x, filters=1, kernel_size=(1), strides=(1))
#
#x = Add()([x_1, x_2, x_3, x_4, x_5])
#x = cba(inputs, filters=12, kernel_size=(5), strides=(1))

#x = cba(x, filters=1, kernel_size=(1), strides=(1))


x_1 = Conv1D(filters=32, kernel_size=2, strides=1, padding='same')(inputs)
x_1 = Conv1D(filters=64, kernel_size=2, strides=1, padding='same')(inputs)
x_2 = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
x_2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
x_3 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(inputs)
x_3 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(inputs)

x = Concatenate()([x_1, x_2, x_3])
x = BatchNormalization()(x)

x = Conv1D(filters=128, kernel_size=7, strides=1, padding='same')(x)
x = MaxPooling1D()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = GlobalAveragePooling1D()(x)
x = Dense(156, activation='softmax')(x)

#x = GlobalAveragePooling1D()(x)
#x = Dense(156, activation='elu')(x)
#x = Dense(156, activation='softmax')(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = tensorflow.keras.optimizers.Adam(lr=0.00001, decay=1e-6, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

print("start learning...")

history = model.fit(x=x_train, y=y_train, epochs=400)

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
