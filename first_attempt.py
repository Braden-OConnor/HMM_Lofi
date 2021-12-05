# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:04:22 2021

@author: oconn
"""

import pretty_midi
import numpy as np
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
import IPython
import pickle
from hmmlearn import hmm
import datetime


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


pm = pretty_midi.PrettyMIDI('LofiPianoSample1.mid')


plt.figure(figsize=(8, 4))
plot_piano_roll(pm = pm, start_pitch = 30, end_pitch = 100)


roll = pm.get_piano_roll()


notes = []
note_lens = []

for i in range(0,len(roll[0])):
    note_slice = [0,0,0,0,0,0,0,0,0,0]
    size = 0
    for j in range (0,len(roll)):
        velocity = roll[j][i]
        if velocity > 0:
            note_slice[size] = j
            size = size + 1
    notes.append(np.array(note_slice))
    note_lens.append(10)      
    
    
remodel = hmm.GaussianHMM(n_components=75, covariance_type="diag", n_iter=12, min_covar=0.1) 
model = remodel.fit(notes) 

test = model.sample(4000,None)

song = []
for i in range (0,len(test[0])):
    npa = np.array(test[0][i])
    npa = np.unique(npa.astype(int))
    song.append(npa)    

song_gen = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0, is_drum=False, name='my piano')
song_gen.instruments.append(inst)

holder = song

test = []
test.append(song[0])
test.append(song[1])
test.append(song[2])



for i in range(0,len(song)):
    row = np.unique(song[i])
    for j in range(0,len(row)):
       note_len = 1
       curr_pitch = row[j]
       if curr_pitch > 0:
           hold = 1
           while ((i+hold) < len(song)) and (curr_pitch in song[i+hold]):
               
               note_len = note_len + 1
               song[i+hold] = np.delete(song[i+hold], np.where(song[i+hold] == curr_pitch))
               hold = hold + 1

       inst.notes.append(pretty_midi.Note(100, curr_pitch, i/100, (i+note_len)/100))
print(inst.notes)
      
song_gen.instruments.append(inst)
    
    
fs = 16000
IPython.display.Audio(song_gen.synthesize(fs=16000), rate=16000)

header = "test" + str(datetime.datetime.now().day) + str(datetime.datetime.now().minute) + ".mid"
song_gen.write(header)










































