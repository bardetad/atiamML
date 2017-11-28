# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:26:00 2017

@author: bavo
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
os.chdir("/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project")

class VAE_Vanilla_test(nn.Module):
    
    mb_size = 400
    Z_dim = 2
    h_dim = 128
    X_dim = 784     # =28*28 
    lr = 1e-3
    
    def __init__(self):
        super(VAE_Vanilla_test, self).__init__()
        
        self.Q_in  = nn.Linear(784,128)
        self.Q_mu  = nn.Linear(128,VAE_Vanilla_test.Z_dim)
        self.Q_var = nn.Linear(128,VAE_Vanilla_test.Z_dim)
        
        self.P_z   = nn.Linear(VAE_Vanilla_test.Z_dim,128)
        self.P_out = nn.Linear(128,784)
    
    def forward(self, X):
        #       Q        
        h = F.relu(self.Q_in(X))
        z_mu = self.Q_mu(h)
        z_var = self.Q_var(h)
        #       Sample
        z = self.sample_z(z_mu, z_var)
        #       P
        X_sample = self.P(z)
        #       Loss
        loss = self.Loss(X_sample, z_mu, z_var)
        return X_sample
    def Q(self,X):
        #       Q        
        h = F.relu(self.Q_in(X))
        z_mu = self.Q_mu(h)
        z_var = self.Q_var(h)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(self.mb_size, self.Z_dim))
        return mu + torch.exp(log_var / 2) * eps
    
    def P(self,z):
        h = F.relu(self.P_z(z))
        X = F.sigmoid(self.P_out(h))
        return X
    
    def Loss(self,X_sample, z_mu, z_var):
        recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False) / VAE_test.mb_size
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss = recon_loss + kl_loss
        return loss
        
def plot_images(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(20,20)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.plot()

#%%                                     Load model
VAE_test = VAE_Vanilla_test()
#VAE_test.load_state_dict(torch.load('./model_100'))
VAE_test.load_state_dict(torch.load('model_2_warmup', map_location={'cuda:0': 'cpu'}))
       
#%%                                     Load data
trainset = torchvision.datasets.MNIST(root='./data', train=True,download=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=VAE_test.mb_size,shuffle=False, num_workers=0)
dataiter = iter(trainloader)
image_in, labels = dataiter.next()
X = image_in.view(VAE_test.mb_size,VAE_test.X_dim)
X = Variable(X)

#%%                              Test 1: compare in- and output
X_sample = VAE_test.forward(X)
image_out = X_sample[0].data.numpy()
image_out = image_out.reshape(VAE_test.mb_size,28,28)
image_in = torch.squeeze(image_in)
image_in = image_in.numpy()
images = [image_in, image_out]

# Plot
for j in range(2):
    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(np.sqrt(VAE_test.mb_size),np.sqrt(VAE_test.mb_size))
    gs.update(wspace=0.05, hspace=0.05)
    samples = images[j]
    plot_images(samples)

#%%                             Test 2: analyse z-space (random z)
z = Variable(torch.randn(VAE_test.mb_size, VAE_test.Z_dim))

samples = VAE_test.P(z).data.numpy()

# Plot
plot_images(samples)

# plot one z
temp = torch.from_numpy(np.array([-2.0,-2.0]))
temp = Variable(temp.float())
temp2 = VAE_test.P(temp).data.numpy()
plt.imshow(temp2.reshape(28,28))

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
plot_images(samples)

#%%                         Visualize (PCA)
from sklearn.decomposition import PCA as sklearnPCA
from mpl_toolkits.mplot3d import Axes3D

size = 800
image_size = 28*28
n_components = 2

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=size,shuffle=False, num_workers=0)
dataiter = iter(trainloader)
image_in, labels = dataiter.next()
X = image_in.view(size,image_size)
X = Variable(X)

data_in = image_in.view(size, image_size).numpy()
labels_in = labels.numpy()

VAE_test = VAE_Vanilla_test()

z_mu, z_var = VAE_test.Q(X)
z_mu = z_mu.data.numpy()
z_var = z_var.data.numpy()

sklearn_pca = sklearnPCA(n_components)
#PCA_proj = z_mu
#PCA_proj = sklearn_pca.fit_transform(data_in)
PCA_proj = sklearn_pca.fit_transform(z_mu)

fig = plt.figure(figsize=(12, 8))
#ax = Axes3D(fig)
ax = fig.add_subplot(111)
for i in range(10):
    bool_ar = labels_in == i
    color_ar = np.random.rand(3,1)
    for j in range(len(PCA_proj[bool_ar,0])):
        x = PCA_proj[bool_ar, 0][j]
        y = PCA_proj[bool_ar, 1][j]
#        z = PCA_proj[bool_ar, 2][j]
        ax.text(x, y, str(i), bbox=dict(color = color_ar, alpha=0.5), fontsize=12)
add_x = np.std(PCA_proj[:,0])/2
add_y = np.std(PCA_proj[:,1])/2
#add_z = np.std(PCA_proj[:,2])/2
ax.set_xlim(min(PCA_proj[:,0])-add_x ,max(PCA_proj[:,0])+add_x )
ax.set_ylim(min(PCA_proj[:,1])-add_y,max(PCA_proj[:,1])+add_y)
#ax.set_zlim(min(PCA_proj[:,2])-add_z,max(PCA_proj[:,2])+add_z)

plt.show()





