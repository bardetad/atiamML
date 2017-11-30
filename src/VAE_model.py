# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:45:30 2017

Class VAE_Vanilla:      VAE model
- X_dim: input layer (spectrogram size)
- h_dim: hidden layer
- Z_dim: latent space layer

Class Toy_Dataset_1:    to enable torch.data.utils functionality on .npz input

@author: wiseodd with modif from bavo
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE_Vanilla(nn.Module):
    
    mb_size = 10
    
    X_dim = 513
    h_dim = 128
    Z_dim = 6
    
    lr = 1e-3
    
    def __init__(self):
        super(VAE_Vanilla, self).__init__()
        
        self.beta = 0        
        
        self.Q_in  = nn.Linear(self.X_dim,128)
        self.Q_mu  = nn.Linear(128,self.Z_dim)
        self.Q_var = nn.Linear(128,self.Z_dim)
        
        self.P_z   = nn.Linear(self.Z_dim,128)
        self.P_out = nn.Linear(128,self.X_dim)
    
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
        loss = self.Loss(X_sample, X, z_mu, z_var)
        return loss

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(self.mb_size, self.Z_dim))
#        eps = eps.cuda()
        return mu + torch.exp(log_var / 2) * eps
    
    def P(self,z):
        h = F.relu(self.P_z(z))
        X_sample = F.sigmoid(self.P_out(h))
        return X_sample
    
    def Loss(self,X_sample, X, z_mu, z_var):
        recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False) / self.mb_size
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss = recon_loss + self.beta*kl_loss
        return loss
        
class ToyDataset_1(Dataset):

    def __init__(self, npz_file, root_dir):
        npz_dict = np.load(root_dir + npz_file)
        self.imgs_stack = npz_dict['images']
        self.labels_stack = npz_dict['labels']
        self.root_dir = root_dir

    def __len__(self):
        return len(self.labels_stack)

    def __getitem__(self, idx):
        image = self.imgs_stack[:,:,idx]
        label = self.labels_stack[idx]
        sample = {'image': image, 'label': label}

        return sample
