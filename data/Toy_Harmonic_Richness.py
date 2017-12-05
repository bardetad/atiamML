#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:50:14 2017

This script generates a Toy Dataset to test the learning of our VAE.
The data represents harmonical richness. 
The values increase smoothly in number of partials. The spectral slope is
fixed to 1/n with n the partial number.
The first signal is a pure sinwave at fond_f Hz, the second signal is the
first one added to a pure sinwave at 2*fond_f Hz and a small amplitude. Then
the same partial increases until it reaches 1/n (1/2) and passes on the 3d 
partial... and so on... so that the number of signals adds up to dataset_size.
 
@author: Leki
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


folder_path = '/Users/Leki/Documents/Atiam/Info/ProjectML/ToyDataset/';
fond_f = 220;                   # Fondamental Frequency
Fe =44100;                      # Sampling rate
fmax = Fe/4;                    # Maximum frequency authorized
num_harm = int(fmax/fond_f);    # Number of harmonical partials before fmax
time_len = 1;                   # Temporal signal time in seconds
Nfft = 4096;                
dataset_size = 10000;
num_gains = dataset_size/num_harm;
WRITE_WAV= False;
PLOT = False,

t = np.linspace(0,1,np.ceil(time_len*Fe));      # Time vector
Spectrums = np.zeros([Nfft/4,dataset_size]);    # Spectrums  : OUTPUT
Labels = ["" for x in range(dataset_size)];     # Labels     : OUTPUT


yout = np.sin(2*np.pi*fond_f*t);                # Signal generated

if WRITE_WAV :
    sf.write('0_sinus_0.wav', yout, Fe);            # Write wav to listen


inc = 0;
for harmonic in range(2,num_harm+1):
    yprev = yout;
    
    # Ploting
    if PLOT :
        plt.plot(10**(Spectrums[:,inc]/20));    # Plot in linear
        plt.show()
        
    for harmonic_gain in range(1,num_gains+1):
        inc = inc + 1;
        
        # Applying gain
        gain = float(harmonic_gain)/num_gains;      # value in [1/num_gains, 1]
        gain = gain/harmonic;                           # Decroissance en 1/n
        gain = float(int(gain*harmonic*10**4)*10**(-4)); # Precision reduction
        ytemp = gain*np.sin(2*np.pi*fond_f*harmonic*t);   
        
        # Adding previous and normalizing
        yout = yprev + ytemp;            
        yout = yout/max(abs(yout));                     
        label = str(harmonic) + '_' + str(gain); 
        
        
        if WRITE_WAV : 
            sf.write(label + '.wav', yout, Fe);
        
        # Compute Magnitude Spectrum
        spectrum_temp = yout[0:Nfft]*np.hanning(Nfft);
        spectrum_temp = 20*np.log10(np.fft.fft(spectrum_temp,Nfft));
        spectrum_temp = spectrum_temp[0:Nfft/4];
        
        # Store in numpy array
        Spectrums[:,inc] = spectrum_temp;
        Labels[inc] = label;


#Save to npz file
toy_dataset_dict = {'Spectrums': Spectrums, 'labels': Labels}
np.savez(folder_path + 'toy_dataset_spectral_richness.npz', **toy_dataset_dict)
