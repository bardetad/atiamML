# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:10:26 2017

Script to create datasets with fixed pitch:
DATASET 1 (f0 = 1500) FM-synthesisa + Moog filter:
1. Carrier frequency j (3 carriers)
2. Ratio k 
3. Index r
4. Filter resonance Q
5. Filter frequency f

DATASET 2 (fundamental = 110) Additive synthesis with BiQuad filter:
1. Harmonicity = sum(s3.bn/in((2n+j)*f*i), j = [0,1,0.1]
2. # Partials k
3. Filter (Q)
4. Filter (f)
@author: bavo
"""
import pyo
import numpy as np
import os

# Start server
s = pyo.Server(audio="offline")

# path to your sound folder
#folder_path = "/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/training_2/"
folder_path = '/media/bavo/1E84369D843676FD/Documents and Settings/bavov/Documents/ATIAM/4_Informatique/MachineLearning_Project/Datasets/toy_dataset_3/'

# create it if it does not exist
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

# Server parameters
sr = 44100
chnls = 1
fformat = 'WAVE'
dur = 0.2
samptype = 0  # 16 bits int

# Simple selector :-))
data = 1


# set server parameters
s.setSamplingRate(sr)
s.setNchnls(chnls)
s.setVerbosity(1)
s.boot()

if data == 1:
    f0 = 1500
    for j in np.arange(1, 5, 0.5):
        print(j)
        freq1 = f0
        freq2 = f0 + j * f0 / 10
        freq3 = f0 - j * f0 / 10
        freq_ar = [freq1, freq2, freq3]
        for k in np.arange(0.1, 0.4, 0.05):
            ratio_ar = [k, k, k]
            for r in np.arange(0, 30, 4):
                for Q in np.arange(1, 11, 1):
                    for filter_f in np.arange(1, 11, 1):
                        # Set sound parameters
                        sound = 'set1_' + str(int(j / 0.5 - 1)) +  '_' \
                            + str(int(k / 0.05 - 1)) + '_' \
                            + str(int(r / 4 + 1)) + '_' \
                            + str(int(Q)) + '_' \
                            + str(int(filter_f)) + '_' \
                            + '.wav'

                        s.recordOptions(dur=dur, filename=os.path.join(folder_path + sound),
                                        fileformat=fformat, sampletype=samptype)

                        # Create sound
                        sig = pyo.FM(carrier=freq_ar,
                                     ratio=ratio_ar, index=float(r), mul=1)
                        sig_filtered = pyo.MoogLP(sig, freq=float(
                            filter_f) * sr / 200, res=float(Q), mul=1).out()

                        # start the render
                        s.start()
                        # cleanup
                        s.recstop()
            #            s.shutdown()
else:
    f0 = 110
    for j in np.arange(0, 1, 0.1):
        print(j)
        freq_ar = [f0]
        for k in np.arange(0, 20, 1):
            freq_new = (2 * k + j) * f0
            freq_ar.append(freq_new)
            for Q in np.arange(1, 11, 1):
                for filter_f in np.arange(1, 11, 1):
                    # Set sound parameters
                    sound = 'set1_' + str(int(j * 10)) +  '_' \
                        + str(int(k)) + '_' \
                        + str(int(Q)) + '_' \
                        + str(int(filter_f)) + '_' \
                        + '.wav'

                    s.recordOptions(dur=dur, filename=os.path.join(folder_path + sound),
                                    fileformat=fformat, sampletype=samptype)

                    # Create sound
                    sin_sig = pyo.Sine(freq=freq_ar, mul=1)
                    sig_filtered = pyo.Biquad(sin_sig, freq=float(
                        filter_f * sr / (2 * 100)), q=float(Q), type=2).out()

                    # start the render
                    s.start()
                    # cleanup
                    s.recstop()
        #            s.shutdown()
