import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

def main():
  mid_file = '../data/midi/daylight.mid'
  song_name = 'test/daylight'

  mid = MidiFile(mid_file)
  
  # identify the meta messages
  metas = []
  tempo = 500000
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
      print(msg)
    if msg.type is 'end_of_track' or time_elapsed >= target_segment_len:
      print('----------------')
      track.append(MetaMessage('end_of_track'))
      target.save(song_name + '_{}.mid'.format(prefix))
      target = MidiFile()
      track = MidiTrack()
      track.extend(metas)
      target.tracks.append(track)
      time_elapsed = 0
      prefix += 1


if __name__ == '__main__':
  main()
