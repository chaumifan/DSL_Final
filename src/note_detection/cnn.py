import numpy as np
import keras
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from  skimage.measure import block_reduce

from PIL import Image
import utils
import pretty_midi
import os, os.path


def create_model():
    img_x, img_y = 145, 49
    input_shape = (img_x, img_y, 3)
    num_classes = 128

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),
        activation='tanh',
        input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, (3,3), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(64, (5,5), activation='relu'))
    # Final output layer
    #model.add(Conv2D(128, (5,5), activation='sigmoid'))
    #model.add(Flatten())
    model.add(Flatten())
    #model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
    

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def train(x_train, y_train, x_test, y_test):
    batch_size = 64
    num_classes = 128
    epochs = 100

    # input image dimensions
    img_x, img_y = 624, 1222

    path = '/mnt/d/Workspace/EE379K/DSL_Final/models'
    #path = '/mnt/c/Users/chau/Documents/models'
    model_ckpt = os.path.join(path,'ckpt.h5')
    
    #x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
    #x_test = x_test.reshape(x_train.shape[0], img_x, img_y, 3)
    
    # Convert data to right type
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    #print(x_train)
    #print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = create_model()
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
            metrics=['accuracy'])

    history = AccuracyHistory()

    checkpoint = ModelCheckpoint(model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(patience=5,monitor='val_loss', verbose=1, mode='min')
    callbacks = [history, checkpoint,early_stop]
    
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=callbacks)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1,101), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('loss.png')
    plt.show()


def run_cnn(jpg_path, midi_path):
    # x is spectrogram, y is MIDI
    jpg_path = '/mnt/d/Workspace/EE379K/data/spectrograms'
    midi_path = '/mnt/d/Workspace/EE379K/data/split_midi'
    #jpg_path = '/mnt/c/Users/chau/Documents/spectrograms'
    #midi_path = '/mnt/c/Users/chau/Documents/split_midi'
    x_train, y_train = [], []
    img = []
    i = 0
    for filename in os.listdir(jpg_path):
        print(filename)
        # filename = "daylight_128.jpg"
        m_fn = filename.replace(".jpg", ".mid")
        if os.path.isfile(os.path.join(midi_path, m_fn)):
            pm = pretty_midi.PrettyMIDI(os.path.join(midi_path, m_fn))
            oh = utils.pretty_midi_to_one_hot(pm)
            if type(oh) is not int:
                oh = utils.slice_to_categories(oh)
                #oh = oh.reshape(1, 128)
                y_train.append(oh)
        
                im = Image.open(os.path.join(jpg_path, filename))
                im = im.crop((14, 13, 594, 301))
                resize = im.resize((49, 145), Image.NEAREST)
                resize.load()
                #result = Image.fromarray((visual * 255).astype(numpy.uint8))
                #resize.save("images/" + str(i) + ".jpg")
                arr = np.asarray(resize, dtype="float32")
                print(arr)
                #arr = block_reduce(arr, block_size=(2,2,1), func=np.mean)
                x_train.append(arr)
                #if len(x_train) > 0:
                #    break
                i += 1

    x_train = np.array(x_train)
    #x_train = x_train.reshape(len(x_train), 1)
    y_train = np.array(y_train)
    print(y_train)
    print(x_train.shape)
    print(y_train.shape)
    print(len(x_train))
    print(np.shape(x_train))
    #im_array = np.array([np.array
    #x_train = np.array(x_train)
    x_test = np.copy(x_train)
    y_test = np.copy(y_train)
    #x_train, x_test, y_train, y_test = train_test_split(
    #        x_train, y_train, test_size=0.2, random_state=1)
    print(x_train.shape)
    print(y_train.shape)
    x_train /= 255.0
    x_test /= 255.0
    train(x_train, y_train, x_test, y_test)

run_cnn("h", "p")
