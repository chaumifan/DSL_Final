import utils as mt
import pretty_midi as pm
import pandas as pd

file = pm.PrettyMIDI("test.mid")
output = mt.pretty_midi_to_one_hot(file)
csv_out = pd.DataFrame(output)
csv_out.to_csv("output.csv")

o_file = mt.one_hot_to_pretty_midi(output)

o_file.write("output.mid")
