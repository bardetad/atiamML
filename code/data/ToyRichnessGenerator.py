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
the same partial's ampltidue increases until it reaches 1/n (1/2) and passes 
on the 3d partial, and so on. So that the number of signals adds up to 
dataset_size.
 
OUTPUTS :
    Spectrums : 1024 x N values containing N spectra 
                or 2048 x N values containing the associated phases
                    1024 : number of frequency bins
                    N : dataset size
                
    Labels :    2 x N Parameter values associated to spectras and phases
                    [1,:] : Number of harmonics
                    [2,:] : Gain of the last harmonic (compared to 1/n)
    
@author: Alexis
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

PHASE = False    # Boolean to add phase on spectra
PLOT = False     # Boolean to plot for each new harmonic
WRITE_WAV = False  # Boolean to write WAV files along the way

folder_path = './DummyDataset/'
fond_f = 220                # Fondamental Frequency
Fe = 44100                   # Sampling rate
fmax = Fe / 4                 # Maximum frequency authorized
num_harm = int(fmax / fond_f)  # Number of harmonical partials before fmax
time_len = 1                # Temporal signal time in seconds
Nfft = 4096                 # Fourrier transform number of points
dataset_size = 100        # Desired dataset size
num_gains = dataset_size / num_harm  # Caused number of gains for each harmonic

# Initialize variables
#______________________________________________________________________________

t = np.linspace(0, 1, np.ceil(time_len * Fe))      # Time vector

if PHASE == True:
    Spectrums = np.zeros([Nfft / 4 * 2, dataset_size])  # Output 1
else:
    Spectrums = np.zeros([Nfft / 4, dataset_size])

Labels = np.zeros((2, dataset_size))               # Output 2


yout = np.sin(2 * np.pi * fond_f * t)  # Temporal signal, initialised at a pure
# sine wave at fond_f Hz

if WRITE_WAV:
    sf.write(folder_path + '0_sinus_0.wav', yout, Fe)

# Compute Magnitude Spectrum
spectrum_temp = yout[0:Nfft] * np.hanning(Nfft)
spectrum_temp = np.fft.fft(spectrum_temp, Nfft)
modules_temp = 20 * np.log10(np.abs(spectrum_temp))
modules_temp = modules_temp[0:Nfft / 4]

if PHASE == True:
    phases_temp = np.angle(spectrum_temp)
    Spectrums[Nfft / 4:, 0] = phases_temp[0:Nfft / 4]

Spectrums[:Nfft / 4, 0] = modules_temp


Labels[0, 0] = 1
Labels[1, 0] = 1

# Iterate on harmonics
#______________________________________________________________________________

inc = 0
for harmonic in range(2, num_harm + 1):
    yprev = yout

    # Ploting
    if PLOT:
        plt.plot(10**(Spectrums[:, inc] / 20))    # Plot in linear
        plt.show()

    for harmonic_gain in range(1, num_gains + 1):
        inc = inc + 1

        # Applying gain
        gain = float(harmonic_gain) / num_gains   # value in [1/num_gains; 1]
        gain = gain / harmonic                    # 1/n spectral slope
        gain = float(int(gain * 10**4) * 10**(-4))  # Precision reduction
        ytemp = gain * np.sin(2 * np.pi * fond_f * harmonic * t)

        # Adding previous and normalizing
        yout = yprev + ytemp
        yout = yout / max(abs(yout))

        if WRITE_WAV:
            sf.write(folder_path + str(Labels[:, inc]) + '.wav', yout, Fe)

        # Compute Magnitude Spectrum
        spectrum_temp = yout[0:Nfft] * np.hanning(Nfft)
        spectrum_temp = np.fft.fft(spectrum_temp, Nfft)
        modules_temp = 20 * np.log10(np.abs(spectrum_temp))
        modules_temp = modules_temp[0:Nfft / 4]

        # Compute Phase Spectrum
        if PHASE == True:
            phases_temp = np.angle(spectrum_temp)
            phases_temp = phases_temp[0:Nfft / 4]
            Spectrums[Nfft / 4:, inc] = phases_temp

        # Store in numpy array
        Spectrums[:Nfft / 4, inc] = modules_temp
        Labels[0, inc] = harmonic
        Labels[1, inc] = gain * harmonic

Spectrums = Spectrums[:, :inc]   # Keep only the computed spectrums

# Save to npz file
#______________________________________________________________________________

# positive dB
specPos = np.zeros(np.shape(Spectrums))
phase_l = '-'
if PHASE:
    phase_l = '-phase-'
    specPos[Nfft / 4:, :] = Spectrums[Nfft / 4:, :]

specPos[:Nfft / 4, :] = Spectrums[:Nfft / 4, :] - \
    np.min(Spectrums[:Nfft / 4, :])
toy_dataset_dict = {'Spectrums': specPos, 'labels': Labels}
np.savez(folder_path + 'toy-spectral-richness-v2-db' + phase_l + 'pos.npz',
         **toy_dataset_dict)

# dB normalized
specDbNorm = specPos
specDbNorm[:Nfft / 4, :] = specPos[:Nfft / 4, :] / \
    np.max(specPos[:Nfft / 4, :])
toy_dataset_dict = {'Spectrums': specDbNorm, 'labels': Labels}
np.savez(folder_path + 'toy-spectral-richness-v2-db' + phase_l + 'norm.npz',
         **toy_dataset_dict)

# linear
specLin = np.zeros(np.shape(Spectrums))
specLin[:Nfft / 4, :] = 10**(Spectrums[:Nfft / 4, :] / 20)

if PHASE:
    specLin[Nfft / 4:, :] = Spectrums[Nfft / 4:, :] - \
        np.min(Spectrums[Nfft / 4:, :])
    specLin[Nfft / 4:, :] = specLin[Nfft / 4:, :] * \
        np.max(specLin[:Nfft / 4, :]) / np.max(specLin[Nfft / 4:, :])

toy_dataset_dict = {'Spectrums': specLin, 'labels': Labels}
np.savez(folder_path + 'toy-spectral-richness-v2' + phase_l + 'lin.npz',
         **toy_dataset_dict)

# linear normalized
specLinNorm = specLin
specLinNorm[:Nfft / 4, :] = specLin[:Nfft / 4, :] / \
    np.max(specLin[:Nfft / 4, :])
# specLinNorm[Nfft / 4:, :] = specLin[Nfft / 4:, :] - \
#     np.min(specLin[Nfft / 4:, :])
# specLinNorm[Nfft / 4:, :] = specLin[Nfft / 4:, :] / \
#     np.max(specLin[Nfft / 4:, :])

toy_dataset_dict = {'Spectrums': specLinNorm, 'labels': Labels}
np.savez(folder_path + 'toy-spectral-richness-v2-lin' + phase_l + 'norm.npz',
         **toy_dataset_dict)
