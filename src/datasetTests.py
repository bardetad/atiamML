# @pierrotechnique
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#import soundfile as sf

fs = 44100. # Sampling frequency
f1 = 110. # Fundamental frequency (n = 1)
n_max = 48 # Little n_max for testing
#n_max = int((0.25*fs)/f1) # Max harmonic below fs/4
N = 4096 # Target number of points
T = float(N)/fs # Corresponding temporal period
t = np.linspace(0,T,N) # Corresponding time vector
delf = fs/N # Discrete frequency step
Nbeta = int(f1/delf)
betaVect = np.linspace(0.,f1,Nbeta,endpoint=False) # Inharmonicity factor
#Nphi = int(20*np.pi)
#phiVect = np.linspace(0.,2*np.pi,Nphi,endpoint=False) # Phase
nVect = [i+2 for i in xrange(n_max-1)] # Modal indices
f = np.linspace(0,0.25*fs,N/4) # Frequency vector
w = np.hanning(N)

# Initialize collector array
data = np.zeros(((((n_max-1)*Nbeta)+1),1024),dtype='float32')
# Initialize parameter value array
params = np.zeros(((((n_max-1)*Nbeta)+1),2))
lbls = np.array(['beta','n'])
a1 = np.sin(2*np.pi*f1*t) # Calculate fundamental
A1 = abs(np.fft.fft(a1*w)) # Amplitude spectrum of fundamental
data[0] = A1[0:1024] # Store from 0 to 11025 Hz only
params[0][1] = 1 # Set n = 1 for first entry
i = 1 # Initialize collector index counter

#for phi in phiVect:
for beta in betaVect:
    a = np.sin(2*np.pi*f1*t) # Reset fundamental for each new beta value
    for n in nVect:
        a = a + np.sin(2*np.pi*n*(f1+beta)*t) # Sum modes
        A = abs(np.fft.fft(a*w)) # Amplitude spectrum
        data[i] = A[0:1024] # Store it
        params[i][0] = beta # Store beta parameter value
        params[i][1] = n # Store n parameter value
        if (i%100 == 0): # Plot some example spectra along the way
            plt.plot(f,(data[i]))
            plt.show()
            filename = '../data/beta'+str(int(beta))+'_n'+str(n)+'_phi'+'.wav'
#           sf.write(filename,(a/max(abs(a))),int(fs),) # Save sound
        i += 1 # Update index counter

data = data/data.max() # Normalize

# Save data arrays
np.savez('../data/beta10_n48.npz',data=data,lbls=lbls,params=params)