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
import numpy as np
import VisualizeFunctions as VF
from ManageDataset import NPZ_Dataset
#from VAE_model import VAE
#from VAE_model import loadVAE
from VAE import VAE
from VAE import loadVAE

# Load Model
#model_folder = '/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/atiamML/data/savedVAE/'
model_folder = '/home/bavo/Desktop/results_atiam/'
data_folder = '/media/bavo/1E84369D843676FD/Users/bavov/Documents/ATIAM/4_Informatique/MachineLearning_Project/Datasets/'
data_set_ar = ['MDCT', 'Spectrums', 'SpectrumsPhase', ]
data_name_ar = ['Spectrumsdataset2NormBVK.npz', 'Spectrumsdataset2DbNormBVK.npz'\
                ,'Spectrumsdataset3NormBVK.npz','SpectrumsPhasedataset2NormBVK.npz'\
                ,'MDCTdataset2Norm_BVK.npz']
data_set = data_set_ar[1]
data_name = data_name_ar[2]
Z_dim = 5

model_name = 'SpectrumsSpectrumsdataset3NormBVK_NPZ_E<1025-relu6-600-muSig-32>_D<32-relu6-600-muSig-1025>_beta1_mb100_lr0dot0001_ep500'
VAE_test = loadVAE(model_name, model_folder)

# Load data
toydataset_1 = NPZ_Dataset(npz_file=data_name, root_dir= data_folder + data_set + '/', dataName='images')
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE_test.mb_size, shuffle=True, num_workers=4)

#%%                 ########## Visualize in/out ##########
idx = 0
frameNb = 500
X_np, labels_np, X_mu_np = VF.RunVAEOnce(trainloader, VAE_test)

#%% Plot reconstruction 
for idx in range(2):
    VF.plotInOut(X_np, X_mu_np, idx+5)

#%% Plot random z
VF.plotRandZ(VAE_test, idx)

#%% Plot linear variation of 2 dim of z
zdim_y = 0
zdim_x = 1
zdim_xrange = 50
outputfolder = './imgs/'
# Chaque image: zdim_y varie de [-10,10] pour #frameNb 
# Une image per zdim_x qui varie de [-5,5] pour zdim_xrange
VF.plotLinearZ(VAE_test, frameNb, zdim_y, zdim_x, zdim_xrange, outputfolder)

#%% Create Gaussian mesh sampling z-space
VF.CreateZMesh(VAE_test)

#%% Plot PCA of nb_labels of dataset
PCAdim = 2
factor = 0
nb_labels = 2000
sklearn_pca = VF.PlotPCA(VAE_test, trainloader, data_set, PCAdim, factor, nb_labels) 

#%%         ########## Visualize loss ##########
VAE.beta_wu = 1
loss_dataset, label_ar = VF.CalculateDatasetLoss(trainloader, VAE_test)

#%% Calculate boxplot 
VF.BPlotLabelLoss(loss_dataset, label_ar, VAE_test)
#%%

VF.plotLoss(VAE_test)
#VF.plotLoss_depreciated(model_folder, data_set, data_name, Z_dim)
#%%         ########## Reconstruct Audio ##########

VF.MoveAudio(data_folder, data_name, labels_np, idx)

#%%
import librosa
from torch.autograd import Variable

def griffLim_stft(S, Phase = 0): 
    Nfft = S.shape[0]*2 - 2
    S = np.log1p(np.abs(S))  
    a = np.zeros_like(S)
    a = np.exp(S) - 1
    if not type(Phase) == int:
        Phase = np.append(Phase, np.random.random_sample((Nfft/2-Phase.shape[0]+1,Phase.shape[1])),axis=0)
        p = Phase
    else:
        p = 2*np.pi*np.random.random_sample(a.shape) -np.pi
    for i in xrange(250):
        S = a*np.exp(1j*p)
        x = librosa.istft(S)
        spec = librosa.stft(x, n_fft = Nfft)
        p = np.angle(spec)
    return x

nbFrames = 40
idx = 0
outputpath = '' # Save in current working dir
#VF.SpecToAudio(X_mu_np, idx, nbFrames, outputpath)

a=np.zeros((1,2))
a[0,0] = 2.5
a[0,1] = 0 

test = sklearn_pca.inverse_transform(a)
z = Variable(torch.from_numpy(test)).float()
VAE_test.decode(z)
X_mu = VAE_test.X_mu
X_mu_np = X_mu.data.numpy()

Nfft = 4096
S = np.repeat(X_mu_np[0,:][np.newaxis].T,nbFrames,axis=1)
S = np.append(S, np.zeros((Nfft/2-S.shape[0]+1,nbFrames)),axis=0)
x = griffLim_stft(S)*300
output_name = 'ReconAudio_'
librosa.output.write_wav(outputpath + output_name + 'gl.wav', x, 44100)
    
#%%
VF.SpecPhaseToAudio(X_mu_np, idx, nbFrames, outputpath)

#%%
VF.MDCTToAudio(X_mu_np, idx, nbFrames, outputpath)

#VF.SpecToGranulator(X_mu_np, idx)



