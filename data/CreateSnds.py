# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:10:26 2017

Script to create toy data-set 1 with six latent space factors
1. Pitch =  70 + exp(i), with i = [0,10,0.5]
2. Harmonicity = sum(sin((2n+j)*f*i), j = [0,1,0.1]
3. # Partials k
4. Noisiness: sig = alpha*sin() + beta*noise()
5. Filter (Q)
6. Filter (f)

@author: bavo
"""
from pyo import *
import numpy as np
import os

# Start server
s = Server(audio="offline")

# path to your sound folder
folder_path = "/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/training_2/"

# create it if it does not exist
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

# Server parameters
sr = 44100
chnls = 1
fformat = 'WAVE'
dur = 0.2
samptype = 0 # 16 bits int
f0 = 110

for j in np.arange(0,1,0.1):
    freq_ar = [f0]
    for k in np.arange(0,11,1):
        freq_new =  (2*k + j)*f0
        freq_ar.append(freq_new)
        for beta in np.arange(0,1.1,0.1):
            for Q in np.arange(1,11,1):
                for filter_f in np.arange(1,11,1):
                    # Set sound parameters
                    sound = 'set1_' + str(int(j*10)) +  '_' \
                        + str(int(k)) + '_' \
                        + str(int(beta*10)) + '_' \
                        + str(int(Q)) + '_' \
                        + str(int(filter_f)) + '_' \
                        + '.wav'
                    
                    # set server parameters
                    s.setSamplingRate(sr)
                    s.setNchnls(chnls)
                    s.boot()
                    s.recordOptions(dur=dur, filename=os.path.join(folder_path + sound),
                                    fileformat=fformat, sampletype=samptype)
                   
                    # Create sound
                    sin_sig = Sine(freq=freq_ar, mul = 1)*(1-beta)
                    noise_sig = Noise(1)*(beta)
                    temp = sin_sig + noise_sig
                    sig_filtered = Biquad(temp, freq = float(filter_f*sr/(2*100)), q = float(Q), type=2).out()
                    
                    # start the render
                    s.start()
                    # cleanup
                    s.recstop() 
                    s.shutdown()