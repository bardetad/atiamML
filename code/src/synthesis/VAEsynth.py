# @pierrotechnique
# -*- coding: utf-8 -*-

import OSC
import numpy as np
import torch
import VAE
import matplotlib.pyplot as plt

direct = '/Users/pierrotechnique/Documents/School/UPMC/M2ATIAM/Informatique'
direct += '/MML/Projet/atiamML/data/trained/'

aTrain='Nwu1/toy-spectral-richness-v2-lin-phase-norm_NPZ_E<2048-relu6-1000-mu'
aTrain+='Sig-10>_D<10-relu6-1000-sigmoid-2048>_beta1_mb100_lr0dot001_ep100'
ampVAE = VAE.loadVAE(aTrain,direct)
miniBatchSize = ampVAE.mb_size
zDim = ampVAE.decoder.dimZ
z = np.zeros((1,zDim),dtype='float32')
i = 0
# ---If phase comes from a separately trained VAE:---

#pTrain = 'Nwu1/pierrotechniquePhase23700_NPZ_E<1024-relu6-512-muSig-3>_D<3-'
#pTrain += 'relu6-512-muSig-1024>_beta1_mb100_lr0dot001_ep128'
#phaseVAE = VAE.loadVAE(pTrain,direct)
# ---Loading from dataset---
#setData = np.load('../data/pierrotechniqueAmpPhase23700.npz')
#setData = setData['spectra']
#setDim = setData.shape[1]
#phaseData = np.load('../data/pierrotechniquePhase23700.npz')
#phaseData = phaseData['spectra']

def griffLim(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    S = np.log1p(S) # ln(1 + S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    for k in range(Nfft): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        x = np.fft.ifft(S,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return (p + np.pi)/(2*np.pi) # Rescaled between 0. and 1. for Max


inOSC = OSC.OSCServer(('127.0.0.1',7000))
outOSC = OSC.OSCClient()
outOSC.connect(('127.0.0.1',9004)) # Match port number to Max out

def cntrlHandler(addr,tags,data,client_address):
    global i
    zAx = data[0]
    zVal = data[1]
#    S = np.zeros(4096,dtype='complex') # If using ifft/istft algorithm
    if (zAx < zDim):
        z[0,zAx] = zVal
        ampVAE.decode(torch.autograd.Variable(torch.from_numpy(z)))
#        X = ampVAE.X_mu.data.numpy()[0] # Gaussian
        X = ampVAE.X_sample.data.numpy()[0] # Bernoulli
        # ---If VAE was trained in dB:---
#        X = X/max(abs(X)) # Normalize
#        X = 10**X # Convert
#        X = X/max(X) # Renormalize
        # ---If no phase information is available:---
#        X = X/max(abs(X)) # Normalize
#        p = griffLim(X) # Griffin-Lim phase approximation
        # ---If phase comes from a separately trained VAE:---
#        X = X/max(abs(X)) # Normalize amplitude
#        phaseVAE.decode(torch.autograd.Variable(torch.from_numpy(z)))
#        p = phaseVAE.X_mu.data.numpy()[0]
#        p = (p/(2*max(abs(p)))) + 0.5 # Rescaled between 0. and 1. for Max
        # ---If phase was trained concatenated to amplitude:---
        p = X[1024:2048]
#        p *= 2*np.pi
#        p -= np.pi
#        p = (p/(max(abs(p)))) # Rescaled between 0. and 1. for Max
        X = X[0:1024]
        X = X/max(abs(X))
#        X = X/max(abs(X)) # Normalize amplitude
        # ---If loading from dataset---
#        k = int(0.5*(zVal+1.)*(setDim-1))
#        X = setData[0:1024,k]
#        p = setData[1024:2048,k]
#        p = (p/(2*max(abs(p)))) + 0.5 # Rescaled between 0. and 1. for Max
        # ---If using ifft/istft algorithm---
#        p *= 2*np.pi
#        p -= np.pi
#        S[0:1024] = X*np.exp(1j*p)
#        S[3072:4096] = np.flip(S[0:1024],0)
#        reS = np.real(S)
#        imS = np.imag(S)
        # ---If you wanna see what's goin' on---
#        if (i%4==0):
#            plt.plot(reS)
#            plt.show()
        i += 1
        # ---If using ifft/istft algorithm---
#        for j in range(32):
#            imag = OSC.OSCMessage()
#            imag.setAddress('/imag')
#            real = OSC.OSCMessage()
#            real.setAddress('/real')
#            imag.append(list(imS[(j*128):((j+1)*128)]))
#            real.append(list(reS[(j*128):((j+1)*128)]))
#            outOSC.send(imag)
#            outOSC.send(real)
        #---If using oscillator bank algorithm---
        for j in range(4):
            phs = OSC.OSCMessage()
            phs.setAddress('/phs')
            amp = OSC.OSCMessage()
            amp.setAddress('/amp')
            phs.append(list(p[(j*256):((j+1)*256)]))
            outOSC.send(phs)
            amp.append(list(X[(j*256):((j+1)*256)]))
            outOSC.send(amp)
        clr = OSC.OSCMessage()
        clr.setAddress('/clr')
        clr.append(1)
        outOSC.send(clr)

inOSC.addMsgHandler('/cntrl',cntrlHandler)
inOSC.serve_forever()