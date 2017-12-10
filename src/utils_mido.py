import numpy as np
from mido import MidiFile, MidiTrack, Message

def midi_to_piano_roll(midi, fs=100):
    piano_rolls = []
	
	for track in midi.tracks):
		one_hot = np.zeros((128, int(fs*tr
		for note in track:
			one_hot[note.pitch, int(note.start*fs)] = 1
			print('note on', note.pitch, int(note.start*fs))

			one_hot[note.pitch, int(note.end*fs)] = 0
			print('note off', note.pitch, int(note.end*fs))
		piano_rolls.append(one_hot)

	one_hot = np.zeros((128, np.max([o.shape for o in piano_rolls])))
	for p in piano_rolls:
		one_hot[:, :o.shape[1]] += o

	one_hot = np.clip(one_hot, -1, 1)
	return one_hot
'''
def piano_roll_to_midi(piano_roll):
    notes, frames = piano_roll.shape
    midi = MidiFile()
	track = MidiTrack()
	midi.tracks.append(track)
	
	piano_roll = np.stack((np.zeros((notes, 1)),
						   piano_roll,
						   np.zeros((notes, 1))))

	changes = np.nonzero(np.diff(piano_roll).T)
	for time, note in zip(*changes):
		change = piano_roll[note, time+1]
		
'''		    
