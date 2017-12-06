import random
from mido import Message, MidiFile, MidiTrack

notes = [64, 64+7, 64+12]

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message("program_change", program=12, time=0))

delta = 16
for i in range(100):
    note = random.choice(notes)
    track.append(Message("note_on", note=note, velocity=100, time=delta))
    track.append(Message("note_off", note=note, velocity=100, time=delta))

mid.save("new_song.mid")
