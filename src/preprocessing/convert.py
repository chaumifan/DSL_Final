import sys
from os import listdir
from os.path import join, isfile
from subprocess import call
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import audiospec
import argparse

def midi_to_wav(midi_file, output, soundfont='/usr/share/sounds/sft/FluidR3_GM.sf2'):
  '''Convert midi file on disk to wav'''
  command = ['fluidsynth', '-a', 'alsa', '-F', output, soundfont, midi_file]
  return call(command)


def wav_to_spectrogram(wav_file, output, segment=None):
  '''Convert wav file to spectogram'''
  rate, data = wavfile.read(wav_file)
  if segment is not None:
    data = data[segment[0] * rate:segment[1] * rate]
  audiospec.plotstft(data[:, 0], rate, plotpath=output, plot_artifacts=False)


def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str)
  parser.add_argument('wav_dir', type=str)
  parser.add_argument('spec_dir', type=str)
  parser.add_argument('-s', '--soundfont', default='/usr/share/sounds/sf2/FluidR3_GM.sf2', type=str)
  return parser.parse_args()


def main():
  '''Convert directory of midi files into wav and spectrogram files'''
  args = parse_args()

  input_dir = args.input_dir
  wav_dir = args.wav_dir
  spec_dir = args.spec_dir
  soundfont = args.soundfont

  # Get all the input midi files
  midis = [x for x in listdir(input_dir) if x.endswith('.mid')]
  # Get existing wav files
  wavs = [x for x in listdir(wav_dir) if x.endswith('.wav')]
  # Get existing spectrograms
  specs = [x for x in listdir(spec_dir) if x.endswith('.jpg')]
  
  for midi in midis:
    print(midi)
    name = midi[:-4]
    # if name + '.wav' not in wavs:
    #   midi_to_wav(join(input_dir, midi), join(wav_dir, name + '.wav'), soundfont=soundfont)
    if name + '.jpg' not in specs:
      wav_to_spectrogram(join(wav_dir, name + '.wav'), join(spec_dir, name + '.jpg'))
  print('Done!')


if __name__ == '__main__':
  main()
