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
VAE_test.load_state_dict(torch.load('model_6_Toy1_MSE', map_location={'cuda:0': 'cpu'}))

# Load data
toydataset_1 = VAE_model.ToyDataset_1(npz_file='toy_dataset_2b_BVK.npz', 
                            root_dir='/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE_test.mb_size, shuffle=True, num_workers=4)

dataiter = iter(trainloader)
sample_dict = dataiter.next()
X = sample_dict['image']
X = Variable(X)
X = X.float()
loss, X_mu = VAE_test.forward(X)

#%%                              Test 1: compare in- and output
idx = 0
image_out = X_mu[idx].data.numpy()
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

X_mu, X_var = VAE_test.P(z)
X_mu = X_mu.data.numpy()

# Plot
idx = 0
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.plot(X_mu[idx])
ax2 = fig.add_subplot(412)
ax2.plot(X_mu[idx+1])
ax2 = fig.add_subplot(413)
ax2.plot(X_mu[idx+2])
ax2 = fig.add_subplot(414)
ax2.plot(X_mu[idx+3])

#%%                            Test 3: Signal Reconstruction and granular synth
import pyo

def griffLim(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    a = S.copy()
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    for i in xrange(250): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        x = np.fft.ifft(S,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return x

#X_mu = X_mu.data.numpy()
samples = X_mu
samples_recon = 10**(samples/20)

sig2 = griffLim(samples_recon[0,:])
sig2_real = np.real(sig2)
sig_mul = sig2_real*10

s = pyo.Server(sr=44100, duplex = 0)
s.boot()
s.start()
tab = pyo.DataTable(size=1024, init=sig_mul.tolist())
tab.view()
env = pyo.HannTable()
pos = pyo.Phasor(1024/44100, 0, 1024)
dur = pyo.Noise(.001, .1)
g = pyo.Granulator(tab, env, 1, pos, dur, 24, mul=1).out()
#s.gui(locals())

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

#%%                         Visualize (PCA)
from sklearn.decomposition import PCA as sklearnPCA
from mpl_toolkits.mplot3d import Axes3D

n_components = 2

# Load data
toydataset_1 = VAE_model.ToyDataset_1(npz_file='toy_dataset_2b_BVK.npz', 
                            root_dir='/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE_test.mb_size, shuffle=True, num_workers=4)
dataiter = iter(trainloader)
sample_dict = dataiter.next()
image_in = sample_dict['image']
label_in = sample_dict['label']
X = Variable(image_in).float()
loss, X_sample = VAE_test.forward(X)

data_in = image_in.numpy()
labels_in = label_in.numpy()

z_mu, z_var = VAE_test.Q(X)
z_mu = z_mu.data.numpy()
z_var = z_var.data.numpy()

sklearn_pca = sklearnPCA(n_components)
PCA_proj = sklearn_pca.fit_transform(z_mu)

fig = plt.figure(figsize=(12, 8))
for k in range(5):
    ax = fig.add_subplot(3,2,k+1)
    for i in range(10):
        bool_ar = labels_in[:,k] == i
        color_ar = np.random.random_sample((3,))
        for j in range(len(PCA_proj[bool_ar,0])):
            x = PCA_proj[bool_ar, 0][j]
            y = PCA_proj[bool_ar, 1][j]
            ax.text(x, y, str(i), bbox=dict(color = color_ar, alpha=0.5), fontsize=12)
    add_x = np.std(PCA_proj[:,0])/2
    add_y = np.std(PCA_proj[:,1])/2
    ax.set_xlim(min(PCA_proj[:,0])-add_x ,max(PCA_proj[:,0])+add_x )
    ax.set_ylim(min(PCA_proj[:,1])-add_y,max(PCA_proj[:,1])+add_y)

plt.show()

    