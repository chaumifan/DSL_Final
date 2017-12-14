# Music Transcription using Machine Learning

This projected was created by Kevin Chau, Kevin Tai, Rohan Kondetimmanahalli, and Dhruv Verma. A blog post detailing more about the project can be found here: https://medium.com/@dhruvverma/music-transcription-using-a-convolutional-neural-network-b115968829f4

## Background

An outline for our project was as follows:
1. Find raw audio file as input
2. Create a spectrogram from the raw audio file
3. Time slice spectrogram image into intervals
4. Feed the CNN a slice of the image as input
5. Take the output of the CNN and turn it into a MIDI file
6. Restitch each of the sliced outputs into one MIDI file

The restitched MIDI file is the transcribed version of the initial raw audio file.

## Installing

Tools required to run this project: Tensorflow, Keras, fluidsynth, Pillow, mido, pretty-midi
