import numpy as np
import keras
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.models import load_model

from  skimage.measure import block_reduce
from PIL import Image
import utils
import pretty_midi as pm 
import os, os.path
import re

model_path = "/mnt/d/Workspace/EE379K/DSL_Final/models/ckpt.h5"
image_path = "/mnt/d/Workspace/EE379K/data/spectrograms"
#image_file = "daylight_41.jpg"

midi_path = "/mnt/d/Workspace/EE379K/data/split_midi/daylight_41.mid"
midi = pm.PrettyMIDI(midi_path)

one_hot = utils.pretty_midi_to_one_hot(midi)
np.savetxt("onehot_short.csv",one_hot, delimiter=",")
y = utils.slice_to_categories(one_hot)

x = []

model = load_model(model_path)
model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

filenums = []
for image_file in os.listdir(image_path):
	print(image_file)
	
	im = Image.open(os.path.join(image_path, image_file))
	im = im.crop((14, 13, 594, 301))
	resize = im.resize((49, 145), Image.NEAREST)
	resize.load()
	arr = np.asarray(resize, dtype="float32")

	x.append(arr)
	filenums.append(int(re.search(r'\d+', image_file).group()))

x = np.array(x)
x /= 255.0

y_pred = model.predict(x)
print(np.nonzero(y))
print(y_pred)

notes_unsorted = [np.argmax(y_pred[n]) for n in range(len(y_pred))]

notes = [x for _,x in sorted(zip(filenums, notes_unsorted))]
print(notes)

i=0
for note in notes:
	one_hot = np.zeros((128, 25))
	one_hot[note, :] = 1
	mid = utils.one_hot_to_pretty_midi(one_hot)
	mid.write('sample_outputs/daylight_' + str(i) + ".mid")
	i += 1