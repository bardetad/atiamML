# @pierrotechnique
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

fs = 44100. # Sampling frequency
f1 = 110. # Fundamental frequency (n = 1)
#n_max = 48 # Little n_max for testing
n_max = int((0.25*fs)/f1) # Max harmonic below fs/4
N = 4096 # Target number of points
T = float(N)/fs # Corresponding temporal period
t = np.linspace(0,T,N) # Corresponding time vector
delf = fs/N # Discrete frequency step
Nbeta = int(f1/delf)
betaVect = np.linspace(0.,f1,Nbeta,endpoint=False) # Inharmonicity factor
Namp = 48
ampVect = np.linspace(0.,1.,Namp)
nVect = [i+2 for i in xrange(n_max-1)] # Modal indices
f = np.linspace(0,0.25*fs,N/4) # Frequency vector
w = np.hanning(N)

# Initialize collector array
data = np.zeros(((((n_max-1)*Nbeta*Namp)+1),1024),dtype='float32')
# Initialize parameter value array
params = np.zeros(((((n_max-1)*Nbeta*Namp)+1),3))
lbls = np.array(['beta','n','amp'])
a1 = np.sin(2*np.pi*f1*t) # Calculate fundamental
A1 = abs(np.fft.fft(a1*w)) # Amplitude spectrum of fundamental
data[0] = A1[0:1024] # Store from 0 to 11025 Hz only
params[0][1] = 1 # Set n = 1 for first entry
i = 1 # Initialize collector index counter

for amp in ampVect:
    for beta in betaVect:
        a = np.sin(2*np.pi*f1*t) + amp*np.random.randn(N)
        for n in nVect:
            a = a + np.sin(2*np.pi*n*(f1+beta)*t)
            A = abs(np.fft.fft(a*w)) # Amplitude spectrum
            data[i] = A[0:1024] # Store it
            params[i][0] = beta # Store beta parameter value
            params[i][1] = n # Store n parameter value
            params[i][2] = amp
            if (i%2000 == 0): # Plot some spectra
                plt.plot(f,(data[i]))
                plt.show()
                plt.plot(t,(a/max(abs(a))))
                plt.show()
#                filename = '../data/beta'+str(int(beta))+'_n'+str(n)+'_amp'
#                filename += str(int(amp))+str(int(10*(amp-int(amp))))+'.wav'
#                sf.write(filename,(a/max(abs(a))),int(fs)) # Save sound
            i += 1 # Update index counter

data = data/data.max() # Normalize

# Save data arrays
np.savez('../data/beta10_n100_amp48.npz',data=data,lbls=lbls,params=params)