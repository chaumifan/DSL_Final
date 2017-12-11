"""
Created on Sun Dec 10 16:50:00 2017

@author: Dhruv
"""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_cqt(song, path):
  plt.figure(figsize=(7.5, 3.75))
  y, sr = librosa.load(song)
  C = librosa.cqt(y, sr=sr)
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                            sr=sr)
  plt.axis('off')
  plt.savefig(path, bbox_inches="tight")
  plt.close('all')

