import numpy as np
import keras
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt

from PIL import Image
import utils
import pretty_midi


def create_model():
    img_x, img_y = 1222, 624
    input_shape = (img_x, img_y, 3)
    num_classes = 128 * 88

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
        activation='relu',
        input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(64, (5,5), activation='relu'))
    # Final output layer
    #model.add(Conv2D(128, (5,5), activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(11264, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
    
def resnet_model(bin_multiple):

    #input and reshape
    inputs = Input(shape=input_shape)
    reshape = Reshape(input_shape_channels)(inputs)

    #normal convnet layer (have to do one initially to get 64 channels)
    conv = Conv2D(64,(1,bin_multiple*note_range),padding="same",activation='relu')(reshape)
    pool = MaxPooling2D(pool_size=(1,2))(conv)

    for i in range(int(np.log2(bin_multiple))-1):
        print i
        #residual block
        bn = BatchNormalization()(pool)
        re = Activation('relu')(bn)
        freq_range = (bin_multiple/(2**(i+1)))*note_range
        print freq_range
        conv = Conv2D(64,(1,freq_range),padding="same",activation='relu')(re)

        #add and downsample
        ad = add([pool,conv])
        pool = MaxPooling2D(pool_size=(1,2))(ad)

    flattened = Flatten()(pool)
    fc = Dense(1024, activation='relu')(flattened)
    do = Dropout(0.5)(fc)
    fc = Dense(512, activation='relu')(do)
    do = Dropout(0.5)(fc)
    outputs = Dense(note_range, activation='sigmoid')(do)

    model = Model(inputs=inputs, outputs=outputs)

    return model


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def train(x_train, y_train, x_test, y_test):
    batch_size = 128
    num_classes = 128
    epochs = 10

    # input image dimensions
    img_x, img_y = 1222, 624
    
    im = Image.open("ex.jpg")
    x_train = np.array([np.array(im)])
    
    # x is spectrogram, y is MIDI
    #for i in range(len(data)):
    #    x_train.append()
    #    y_train.append() 
    #    x_test.append() 
    #    y_test.append() 
    
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
    #x_test = x_test.reshape(x_train.shape[0], img_x, img_y, 3)
    x_test = x_train

    pm = pretty_midi.PrettyMIDI("ex.mid")

    y_train = utils.pretty_midi_to_one_hot(pm)
    y_train = y_train.reshape(1, 11264)
    print "y_train"
    print(y_train.shape)
    y_test = y_train
    
    # Convert data to right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    model = create_model()
    #model = resnet_model(100)
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

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
    plt.savefig('home/Chau/DSL_Final/outputs/loss.png')
    plt.show()

x_train, y_train, x_test, y_test = [], [], [], []
train(x_train, y_train, x_test, y_test)
