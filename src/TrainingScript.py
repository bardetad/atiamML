# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:23:29 2017

@author: bavo
"""
import os
os.chdir('/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/1_VAE_model/')

#%%
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
#import torchvision
#import torchvision.transforms as transforms
import VAE_model
reload(VAE_model)

VAE = VAE_model.VAE_Vanilla()
VAE.cuda()

toydataset_1 = VAE_model.ToyDataset_1(npz_file='toy_dataset_2b_BVK.npz', 
                            root_dir='/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE.mb_size, shuffle=True, num_workers=4)

#trainset = torchvision.datasets.MNIST(root='./data', train=True,download=False, transform=transforms.ToTensor())
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=VAE.mb_size,shuffle=False, num_workers=0)

#dataiter = iter(trainloader)*zcxtoy_dataset_1_BVK.npz
#image = dataiter.next()

params = list(VAE.parameters())
optimizer = optim.Adam(params, lr=VAE.lr)

runs = 20
for epoch in range(runs):
    for i, sample_batch in enumerate(trainloader, 0):
        # Load input
        images = sample_batch['image']
        if images.shape[0] != VAE.mb_size:
            continue
        X = images.float()
        X = Variable(X.cuda())
#        X = Variable(X)
        
        # Housekeeping
        optimizer.zero_grad()       

        # Q -> Sample -> P -> Loss
        loss = VAE(X)
        
        # Backward
        loss.backward()
    
        # Update
        optimizer.step()     
    # Print every now and then
    print('Iter-{}; Loss: {:.4}'.format(epoch, loss.data[0]))

#%%
# Save model 
torch.save(VAE.state_dict(), './model_6_Toy1_MSE')
# Load model
VAE = VAE_Vanilla()
VAE.load_state_dict(torch.load('./model_2'))