import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from os import listdir
from os.path import isfile, split, join
import argparse

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


def merge_midi(midis, input_dir, output, default_tempo=500000):
  '''Merge midi files into one'''
  pairs = [(int(x[:-4].split('_')[-1]), x) for x in midis]
  pairs = sorted(pairs, key=lambda x: x[0])
  midis = [join(input_dir, x[1]) for x in pairs]

  mid = MidiFile(midis[0])
  # identify the meta messages
  metas = []
  # tempo = default_tempo
  tempo = default_tempo // 2
  for msg in mid:
    if msg.type is 'set_tempo':
      tempo = msg.tempo
    if msg.is_meta:
      metas.append(msg)
  for meta in metas:
    meta.time = int(mido.second2tick(meta.time, mid.ticks_per_beat, tempo))
  
  target = MidiFile()
  track = MidiTrack()
  track.extend(metas)
  target.tracks.append(track)
  for midi in midis:
    mid = MidiFile(midi)
    for msg in mid:
      if msg.is_meta:
        continue
      if msg.type is not 'end_of_track':
        msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
        track.append(msg)

  track.append(MetaMessage('end_of_track'))
  target.save(output)


def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str)
  parser.add_argument('target_dir', type=str)
  parser.add_argument('-l', '--length', default=1, type=float)
  parser.add_argument('-m', '--merge', action='store_true')
  return parser.parse_args()


def main():
  args = parse_args()
  input_dir = args.input_dir
  target_dir = args.target_dir
  length = args.length

  # Get all the input midi files
  midis = [x for x in listdir(input_dir) if x.endswith('.mid')]

  if args.merge:
    merge_midi(midis, input_dir, target_dir)
  else:
    for midi in midis:
      print(midi)
      try:
        split_midi(join(input_dir, midi), target_dir, target_segment_len=length)
      except:
        print('\tProblem!')
  

if __name__ == '__main__':
  main()
