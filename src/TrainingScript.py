# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:23:29 2017

@author: bavo
"""

import torch
import torch.optim as optim
from torch.autograd import Variable
from VAE_model import VAE_Vanilla
from VAE_model import ToyDataset_1

VAE = VAE_Vanilla()
#VAE.cuda()

toydataset_1 = ToyDataset_1(npz_file='toy_dataset_1.npz', 
                            root_dir='/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/training/')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE.mb_size, shuffle=True, num_workers=4)

params = list(VAE.parameters())
optimizer = optim.Adam(params, lr=VAE.lr)

runs = 30
VAE.beta = 1
for epoch in range(runs):
#    VAE.beta = epoch*1.0/runs
    for i, sample_batch in enumerate(trainloader, 0):
        # Load input        
        images = sample_batch['image']
        if images.shape[0] != VAE.mb_size:
            continue
        X = images.view(VAE.mb_size,VAE.X_dim).float()
#        X = Variable(X.cuda())
        X = Variable(X)
        
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
torch.save(VAE.state_dict(), './model_6_Toy1')
# Load model
VAE = VAE_Vanilla()
VAE.load_state_dict(torch.load('./model_2'))
