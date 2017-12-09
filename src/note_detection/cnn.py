import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import mnist

batch_size = 128
num_classes = 128
epochs = 10

# input image dimensions
img_x, img_y = 28, 600

# Insert our data here
x_train = np.array([[1,1,1],[2,2,2],[3,3,3]])
y_train = np.array([1,2,3])
#x_test
#y_test

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_train.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)

# Convert data to right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Not sure if we need these
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
    activation='relu',
    input_shape=input_shape))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, (5,5), activation='relu'))
# Final output layer
model.add(Conv2D(128, (5,5), activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1,11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
