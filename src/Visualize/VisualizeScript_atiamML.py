#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:36:40 2017

@author: bavo
"""

# Load data iterator and model
import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../dataset/')

import torch
import VisualizeFunctions as VF
from ManageDataset import NPZ_Dataset
#from VAE_model import VAE
#from VAE_model import loadVAE
from VAE import VAE
from VAE import loadVAE

# Load Model
#model_folder = '/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/atiamML/data/savedVAE/VAE_model/'
model_folder = '/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/atiamML/data/savedVAE/'
data_folder = '/media/bavo/1E84369D843676FD/Users/bavov/Documents/ATIAM/4_Informatique/MachineLearning_Project/Datasets/'
data_set_ar = ['MDCT', 'Spectrums', 'SpectrumsPhase', ]
data_name_ar = ['dataset2BVK.npz', 'dataset3BVK.npz']
data_set = data_set_ar[2]
data_name = data_name_ar[1]
Z_dim = 20

model_name = 'SpectrumsPhasedataset3BVK_NPZz20_E<2050-relu6-400-muSig-20>_D<20-relu6-400-muSig-2050>_beta1_mb50_lr0dot0001_ep200'
VAE_test = loadVAE(model_name, model_folder)

# Load data
toydataset_1 = NPZ_Dataset(npz_file=data_name, root_dir= data_folder + data_set + '/', dataName='images')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE_test.mb_size, shuffle=True, num_workers=4)

########## Visualize in/out ##########
idx = 0
PCAdim = 2
frameNb = 500
X_np, labels_np, X_mu_np = VF.RunVAEOnce(trainloader, VAE_test)

for idx in range(5):
    VF.plotInOut(X_np, X_mu_np, idx)

VF.plotRandZ(VAE_test, idx)

VF.plotLinearZ(VAE_test, frameNb)

VF.CreateZMesh(VAE_test)

factor = 3
nb_labels = 500
VF.PlotPCA(VAE_test, trainloader, PCAdim, factor, nb_labels)   

########## Visualize loss ##########
VAE.beta_wu = 1
loss_dataset, label_ar = VF.CalculateDatasetLoss(trainloader, VAE_test)

VF.BPlotLabelLoss(loss_dataset, label_ar, VAE_test)

VF.plotLoss(VAE_test)
VF.plotLoss_depreciated(model_folder, data_set, data_name, Z_dim)

########## Reconstruct Audio ##########
VF.MoveAudio(data_folder, data_name, labels_np, idx)

nbFrames = 40
outputpath = '' # Save in current working dir
VF.SpecToAudio(X_mu_np, idx, nbFrames, outputpath)

VF.SpecPhaseToAudio(X_mu_np, idx, nbFrames, outputpath)

VF.MDCTToAudio(X_mu_np, idx, nbFrames, outputpath)

#VF.SpecToGranulator(X_mu_np, idx)



