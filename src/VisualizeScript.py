#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:36:40 2017

@author: bavo
"""
import os
os.chdir('/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/1_VAE_model/')

#%%
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import VAE_model
reload(VAE_model)

# Load Model
VAE_test = VAE_model.VAE_Vanilla()
VAE_test.load_state_dict(torch.load('model_2_Toy1_2', map_location={'cuda:0': 'cpu'}))

# Load data
toydataset_1 = VAE_model.ToyDataset_1(npz_file='toy_dataset_2b_BVK.npz', 
                            root_dir='/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE_test.mb_size, shuffle=True, num_workers=4)

dataiter = iter(trainloader)
sample_dict = dataiter.next()
X = sample_dict['image']
X = Variable(X)
X = X.float()
loss, X_sample = VAE_test.forward(X)

#%%                              Test 1: compare in- and output
idx = 90
image_out = X_sample[idx].data.numpy()
image_in = X[idx].data.numpy()
images = [image_in, image_out]

# Plot
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(images[0])
ax2 = fig.add_subplot(212)
ax2.plot(images[1])

#%%                             Test 2: analyse z-space (random z)
z = Variable(torch.randn(VAE_test.mb_size, VAE_test.Z_dim))

samples = VAE_test.P(z).data.numpy()

# Plot
idx = 0
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.plot(samples[idx])
ax2 = fig.add_subplot(412)
ax2.plot(samples[idx+1])
ax2 = fig.add_subplot(413)
ax2.plot(samples[idx+2])
ax2 = fig.add_subplot(414)
ax2.plot(samples[idx+3])

#%%                                     Signal Reconstruction

def griffLim1(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    S = np.log1p(S) # ln(1 + S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    for i in xrange(250): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        x = np.fft.ifft(S,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return np.real(x)

def griffLim2(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    S = np.log1p(S) # ln(1 + S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    S0 = np.sqrt(S)
    for i in xrange(250): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        Si = np.linalg.norm(S)
        x = np.fft.ifft(S0*S/Si,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return np.real(x)

samples_recon = 10**(samples/20)
#spectrum = np.array([[0], samples_recon[0,:]])

sig1 = griffLim1(samples_recon[0,:])
sig2 = griffLim2(samples_recon[0,:])

D = np.fft.rfft(sig1,2024)
log_D = 20*np.log10(np.abs(D)**2)
#%%                           Test 3: analyse z=space (Guassian mesh)
from scipy.stats import norm

# Create Guassian distribution mesh
x, y = np.mgrid[0:1:1.0/np.sqrt(VAE_test.mb_size), 0:1:1.0/np.sqrt(VAE_test.mb_size)]
x = norm.ppf(x)
y = norm.ppf(y) 
x[0,:]= - 2
y[:,0] = -2
x = x.reshape(VAE_test.mb_size,)
y = y.reshape(VAE_test.mb_size,)
z = np.array([x,y]).T
z = Variable(torch.from_numpy(z).float())

# Decode mesh to images
samples = VAE_test.P(z).data.numpy()

# Plot
fig = plt.figure()
for i in range(1,100):
    ax = fig.add_subplot(10,10,i)
    ax.plot(samples[i,:])
fig.show()
    
    