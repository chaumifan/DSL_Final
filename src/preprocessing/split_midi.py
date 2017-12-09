import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from os import listdir
from os.path import isfile, split, join

def split_midi(mid_file, target_dir, default_tempo=500000, target_segment_len=1):
  '''Split midi file into many chunks'''
  song_name = split(mid_file)[-1][:-4]
  mid = MidiFile(mid_file)

  # identify the meta messages
  metas = []
  tempo = default_tempo
  for msg in mid:
    if msg.type is 'set_tempo':
      tempo = msg.tempo
    if msg.is_meta:
      metas.append(msg)
  for meta in metas:
    meta.time = int(mido.second2tick(meta.time, mid.ticks_per_beat, tempo))

  # target segment length in seconds
  target_segment_len = 1

  target = MidiFile()
  track = MidiTrack()
  track.extend(metas)
  target.tracks.append(track)
  prefix = 0
  time_elapsed = 0
  for msg in mid:
    # Skip non-note related messages
    if msg.is_meta:
      continue
    time_elapsed += msg.time
    if msg.type is not 'end_of_track':
      msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
      track.append(msg)
    if msg.type is 'end_of_track' or time_elapsed >= target_segment_len:
      track.append(MetaMessage('end_of_track'))
      target.save(join(target_dir, song_name + '_{}.mid'.format(prefix)))
      target = MidiFile()
      track = MidiTrack()
      track.extend(metas)
      target.tracks.append(track)
      time_elapsed = 0
      prefix += 1


def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str)
  parser.add_argument('target_dir', type=str)
  return parser.parse_args()


def main():
  args = parse_args()
  input_dir = args.input_dir
  target_dir = args.target_dir

  # Get all the input midi files
  midis = [x for x in listdir(input_dir) if x.endswith('.mid')]

  for midi in midis:
    split_midi(midi, target_dir)
  

if __name__ == '__main__':
  main()
