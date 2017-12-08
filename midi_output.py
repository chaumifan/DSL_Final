import random
import numpy as np
import pandas as pd
import mido
from mido import Message, MidiFile, MidiTrack


mid = MidiFile("Bowsers_Theme.mid")
tempo = mido.bpm2tempo(120)
ticks = mido.second2tick(mid.length, 3, tempo)
ticks = int(round(ticks))

notes_oh = 128
velocity = 1
features = notes_oh + velocity

matrix = np.ndarray(shape=(ticks, features))
df = pd.DataFrame(data, column=header)
print matrix.shape

# Range for 88 key piano is note 21..108
A_1 = 21
MIDDLE_C = 60
C_9 = 108

strange = 0

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

THRESHOLD = .5
delta = 16
for sec in range(len(prob)):
    for note_num in range(128):
        if note_prob[note_num] > THRESHOLD:
            if note_num < 21 or note_num > 108:
                strange += 1
            track.append(Message("note_on", note=note_prob[note_num], time=delta))
            track.append(Message("note_off", note=note_prob[note_num], time=delta+100))

track.append(Message("program_change", program=12, time=0))

#delta = 16
#for i in range(128):
#    note = i
#    track.append(Message("note_on", note=note, velocity=100, time=delta))
#    track.append(Message("note_off", note=note, velocity=100, time=delta+100))

mid.save("result.mid")
