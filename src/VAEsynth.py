# @pierrotechnique
# -*- coding: utf-8 -*-

import OSC
import numpy as np
import torch
import sys
import VAE

# from VAE: decoder, zDim, miniBatchSize,
zDim = 3
z = np.zeros((zDim),dtype='float32')

savename = 'toy-dataset-1_NPZ_E_1024-relu6-600-muSig-10__D_10-relu6-600-muSig'
savename += '-1024__beta1_mb50_lr0dot001_ep250'
directory = '/Users/pierrotechnique/Documents/School/UPMC/M2ATIAM/Informatique'
directory += '/MML/Projet/atiamML/src/'
model = VAE.loadVAE(savename,directory)
miniBatchSize = model.mb_size

def latentVar(val,miniBatchSize):
    dim = val.size
    z_np = np.tile(val,miniBatchSize).reshape((miniBatchSize,dim))
    z_t = torch.autograd.Variable(torch.from_numpy(z_np))
    return z_t

def griffLim(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    S = np.log1p(S) # ln(1 + S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    for i in range(Nfft): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        x = np.fft.ifft(S,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return p

inOSC = OSC.OSCServer(('127.0.0.1',1234))
outOSC = OSC.OSCClient()
outOSC.connect(('127.0.0.1',4321))

def cntrlHandler(addr,tags,data,client_address):
#    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
#    txt += str(data)
    zAx = data[0]
    zVal = data[1]
    if (zAx > (zDim-1)):
        print('Control dimension beyond latent space dimensionality!')
    else:
        z[zAx] = zVal
        model.decode(latentVar(z,miniBatchSize))
#        X,_ = P(latentVar(z),miniBatchSize)
        X = model.X_mu
        X = X.mean(dim=0).data.numpy()
        for k in range(len(X)):
            if (X[k] < 0.):
                X[k] = sys.float_info.epsilon
        p = griffLim(X)
        Xp = np.ravel(np.column_stack((X,p)))
        spctrm = OSC.OSCMessage('/spctrm')
        spctrm.append(list(Xp))
        outOSC.send(spctrm)

inOSC.addMsgHandler('/cntrl',cntrlHandler)
inOSC.serve_forever()
#inOSC.server_activate()
#inOSC.server_close()