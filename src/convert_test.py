import midi_test as mt
import pretty_midi as pm

file = pm.PrettyMIDI("test.mid")
output = mt.pretty_midi_to_one_hot(file)
o_file = mt.one_hot_to_pretty_midi(output)

o_file.write("output.mid")