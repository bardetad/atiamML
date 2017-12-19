# @pierrotechnique
# -*- coding: utf-8 -*-

import OSC
import numpy as np
import torch
import sys
import VAE
import matplotlib.pyplot as plt

savename = 'toy-dataset-1_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig'

savename += '-1024>_beta1_mb50_lr0dot001_ep800'
directory = '/Users/pierrotechnique/Documents/School/UPMC/M2ATIAM/Informatique'
directory += '/MML/Projet/atiamML/src/'
model = VAE.loadVAE(savename,directory)
miniBatchSize = model.mb_size
zDim = model.decoder.dimZ
z = np.zeros((1,zDim),dtype='float32')
i = 0

def griffLim(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    S = np.log1p(S) # ln(1 + S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    for k in range(Nfft): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        x = np.fft.ifft(S,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return (p + np.pi)/(2*np.pi)

inOSC = OSC.OSCServer(('127.0.0.1',4737))
outOSC = OSC.OSCClient()
outOSC.connect(('127.0.0.1',3747))

def cntrlHandler(addr,tags,data,client_address):
    global i
    zAx = data[0]
    zVal = data[1]
    if (zAx > (zDim-1)):
        print('Control dimension beyond latent space dimensionality!')
    else:
        z[0,zAx] = zVal
        model.decode(torch.autograd.Variable(torch.from_numpy(z)))
        X = model.X_mu.data.numpy()[0]
        Y = X/max(X)
        Z = 10**Y
        s = Z/max(Z)
        p = griffLim(s)
        i += 1
        if (i%10==0):
            plt.plot(s)
            plt.show()
            plt.plot(p)
            plt.show()
        for j in range(4):
            amp = OSC.OSCMessage(address='/amp')
            phs = OSC.OSCMessage(address='/phs')
            amp.append(list(s[(j*256):((j+1)*256)]))
            phs.append(list(p[(j*256):((j+1)*256)]))
            outOSC.send(amp)
            outOSC.send(phs)
        clr = OSC.OSCMessage(address='/clr')
        clr.append(1)
        outOSC.send(clr)

inOSC.addMsgHandler('/cntrl',cntrlHandler)
inOSC.serve_forever()
#inOSC.server_activate()
#inOSC.server_close()