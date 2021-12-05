# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 15:41:14 2021

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

def plot_piano_roll(pm, start_pitch, end_pitch, fs=2):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

rolls = []

for i in range(0,5):
    pm = pretty_midi.PrettyMIDI(str(i+1) + "_clean.mid")
    #plt.figure(figsize=(8, 4))
    #plot_piano_roll(pm = pm, start_pitch = 40, end_pitch = 110)
    rolls.append(pm.get_piano_roll(fs = 2))
    
notes = []
note_lens = []

    
for i in range(0,len(rolls)):   
    current_song = rolls[i]
   
    size = 0
    for j in range(0,len(current_song[0])):
        note_slice = [0,0,0,0,0,0,0,0,0,0]
        size = 0
        for k in range(0,128):            
            velocity = current_song[k][j]
            if velocity > 0:
                note_slice[size] = k
                size = size + 1
        if (size > 0):
            notes.append(note_slice)
    #print(i)
    #print(j+1)
    note_lens.append(j+1)
    
    
remodel = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=12, min_covar=0.1) 
model = remodel.fit(notes, note_lens) 

test = model.sample(20,None)

song = []
for i in range (0,len(test[0])):
    npa = np.array(test[0][i])
    npa = np.unique(npa.astype(int))
    song.append(npa)    

song_gen = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0, is_drum=False, name='my piano')
song_gen.instruments.append(inst)

for i in range(0,len(song)):
    row = np.unique(song[i])
    row = np.delete(row,0)
    for j in range(0,len(row)):
       note_len = 1
       curr_pitch = row[j]
       if curr_pitch > 0:
           hold = 1
           while ((i+hold) < len(song)) and (curr_pitch in song[i+hold]):
               
               note_len = note_len + 1
               song[i+hold] = np.delete(song[i+hold], np.where(song[i+hold] == curr_pitch))
               hold = hold + 1

       inst.notes.append(pretty_midi.Note(100, curr_pitch, i/2, (i+note_len)/2))
#print(inst.notes)
      
song_gen.instruments.append(inst)


header = "test" + str(datetime.datetime.now().day) + str(datetime.datetime.now().minute)+ str(datetime.datetime.now().second) + ".mid"
song_gen.write(header)



plt.figure(figsize=(8, 4))
plot_piano_roll(song_gen, 50, 100)


print(song)




















