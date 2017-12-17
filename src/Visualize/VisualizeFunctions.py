#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:36:40 2017

@author: bavo
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import librosa
import pyo

# Perform one run of VAE
def RunVAEOnce(trainloader, VAE_test):
    dataiter = iter(trainloader)
    sample_dict = dataiter.next()
    
    X = sample_dict['image']
    labels = sample_dict['label']
    
    X = Variable(X)
    X = X.float()
    VAE_test.forward(X)
    X_mu = VAE_test.X_mu
    
    X_np = X.data.numpy()
    X_mu_np = X_mu.data.numpy()
    labels_np = labels.numpy()
    return X_np, labels_np, X_mu_np

###############################  Test 1: compare in- and output ###############################
def plotInOut(X_np, X_mu_np, idx):
    image_out = X_mu_np[idx]
    image_in = X_np[idx]
    images = [image_in, image_out]
    
    # Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(images[0])
    ax2 = fig.add_subplot(212)
    ax2.plot(images[1])

# analyse z-space (random z)
def plotRandZ(VAE_test, idx):
    z = Variable(torch.randn(VAE_test.mb_size, VAE_test.encoder.dimZ))
    
    VAE_test.decode(z)
    X_mu = VAE_test.X_mu
    X_mu = X_mu.data.numpy()
    
    # Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax1.plot(X_mu[idx])
    ax2 = fig.add_subplot(412)
    ax2.plot(X_mu[idx+1])
    ax2 = fig.add_subplot(413)
    ax2.plot(X_mu[idx+2])
    ax2 = fig.add_subplot(414)
    ax2.plot(X_mu[idx+3])
    
def plotLinearZ(VAE_test, frameNb):
    VAE_test.eval()
    for j in range(VAE_test.decoder.dimZ):
        if j > 5:
            break
        tensorParamValues = torch.FloatTensor(
                frameNb, VAE_test.decoder.dimZ).zero_()
        for i in range(frameNb):
            tensorParamValues[i][j] = float(i * 20) / float(frameNb) - 10
        sample = Variable(tensorParamValues)
        VAE_test.decode(sample)
        image = VAE_test.X_mu.cpu()
        plt.figure()
        plt.imshow((image.data.view(frameNb, VAE_test.decoder.dimX)).numpy())
    
    tensorParamValues = torch.FloatTensor(
                frameNb, VAE_test.decoder.dimZ).zero_()
    for i in range(frameNb):
        tensorParamValues[i][:] = float(i * 20) / float(frameNb) - 10
    sample = Variable(tensorParamValues)
    VAE_test.decode(sample)
    image = VAE_test.X_mu.cpu()
    plt.figure()
    plt.imshow((image.data.view(frameNb, VAE_test.decoder.dimX)).numpy())
        

# analyse z=space (Guassian mesh)
def CreateZMesh(VAE_test):
    # Create Guassian distribution mesh
    x, y = np.mgrid[0:1:1.0/np.floor(np.sqrt(VAE_test.mb_size)), 0:1:1.0/np.floor(np.sqrt(VAE_test.mb_size))]
    x = norm.ppf(x)
    y = norm.ppf(y) 
    x[0,:]= - 2
    y[:,0] = -2
    x = x.reshape(len(x)**2,)
    y = y.reshape(len(y)**2,)
    z = np.array([x,y]).T
    if VAE_test.decoder.dimZ > 2:
        # Append zeros for z > 2
        z = np.append(z,np.zeros((len(x),VAE_test.decoder.dimZ - 2)),axis=1)
        z = np.append(z,np.zeros((VAE_test.mb_size - len(x),VAE_test.decoder.dimZ)), axis = 0)
    z = Variable(torch.from_numpy(z).float())
    
    # Decode mesh to images
    VAE_test.decode(z)
    X_mu = VAE_test.X_mu
    samples = X_mu.data.numpy()
    
    # Plot
    fig = plt.figure()
    for i in range(0,len(x)):
        ax = fig.add_subplot(np.sqrt(len(x)),np.sqrt(len(x)),i+1)
        ax.plot(samples[i,:])
    fig.show()

# visualize with PCA
def PlotPCA(VAE_test, labels_np, PCAdim):
    z_mu = VAE_test.z_mu
    z_mu = z_mu.data.numpy()
    
    sklearn_pca = sklearnPCA(PCAdim)
    PCA_proj = sklearn_pca.fit_transform(z_mu)
    
    fig = plt.figure(figsize=(12, 8))
    for k in range(5):
        ax = fig.add_subplot(3,2,k+1)
        for i in range(10):
            bool_ar = labels_np[:,k] == i
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
    
###############################  Test 2: Calculate/visualize loss ###############################
    
# Calculate mean and variance for loss/dataset/factor of variation
def CalculateDatasetLoss(trainloader, VAE_test):
    loss_dataset = np.zeros((len(trainloader.dataset),3))
    dataiter = iter(trainloader)
    sample_dict = dataiter.next()
    labels = sample_dict['label']
    label_ar = np.zeros((1,labels.shape[1]))            
    
    for i, sample in enumerate(trainloader):
        X = sample['image'].float()
        label = sample['label'].numpy()
        X = Variable(X)
        
        # Actual VAE
        VAE_test.forward(X)
        lossVariable, recon, kl = VAE_test.loss(X)
        loss_dataset[i,0] = lossVariable.data[0]
        loss_dataset[i,1] = recon.data[0]
        loss_dataset[i,2] = kl.data[0]
        label_ar = np.append(label_ar,label,axis=0)
    label_ar = np.delete(label_ar,0,axis=0)
    return loss_dataset, label_ar

def BPlotLabelLoss(loss_dataset, label_ar, VAE_test):
    if label_ar.shape[1] == 4:
        label_steps = [10,20,10,10]
        start_step = [0,0,1,1]
    elif label_ar.shape[1] == 5:
        label_steps = [8,7,8,10,10]
        start_step = [1,1,1,1,1]
    else:
        print('no matching VAE dimension')
        return
    
    box_mat = []
    for i in range(label_ar.shape[1]):
        box_vec = np.zeros((label_steps[i],1))
        for j in range(label_steps[i]):
            bool_ar = label_ar[:,i] == j + start_step[i]
            mean = np.mean(loss_dataset[:,0]*bool_ar.T)
            box_vec[j,0] = mean
        box_mat.append(box_vec)
    
    fig = plt.figure(1, figsize=(9, 6))  
    ax1 = fig.add_subplot(111)
    plt.boxplot(box_mat)
    ax1.set_title('Loss for 4 variational factors')
    ax1.set_ylabel('Loss')
    if label_ar.shape[1] == 4:
        ax1.set_xticklabels(['Inharmonicity', '# Harmonics', 'Filter Q', 'Filter freq.'])
    elif label_ar.shape[1] == 5:
        ax1.set_xticklabels(['Carrier', 'Ratio', 'Index', 'Filter Q', 'Filter freq.'])
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    #ax1.set_xlim(0.5, 4)
    #ax1.set_ylim(-1, 1)
    plt.show()

# plot loss evolution
def plotLoss(model_folder, data_set, data_name, Z_dim):
    loss_vector = np.load(model_folder + data_set + '_' + data_name + str(Z_dim) + 'loss_.npy')
    fig, ax = plt.subplots()
    x_axis = range(0,len(loss_vector))
    ax.plot(x_axis, loss_vector[0:,0], 'k--', label='Total loss')
    ax.plot(x_axis, loss_vector[0:,1], 'k:', label='Reconstruction loss')
    ax.plot(x_axis, loss_vector[0:,2], 'k', label='Kull/Leib loss')
    ax.legend(loc='upper center', shadow=True)
    
# ###############################  Test 3:  Signal Reconstruction and granular synth ###############################
def MoveAudio(data_folder, dataname, labels_np, idx):
    label_in = labels_np[idx]
    
    if len(label_in) == 4:
        filename = 'set1_' + str(int(label_in[0])) + '_' + str(int(label_in[1])) + '_' + str(int(label_in[2])) + '_' + str(int(label_in[3])) + '_.wav'
    elif len(label_in) == 5:
        filename = 'set1_' + str(int(label_in[0])) + '_' + str(int(label_in[1])) + '_' + str(int(label_in[2])) + '_' + str(int(label_in[3])) + '_' + str(int(label_in[4])) + '_.wav'
    
    if dataname == 'dataset2BVK.npz':
        folder = 'toy_dataset_2'
    elif dataname == 'dataset3BVK.npz':
        folder = 'toy_dataset_3'
    
    y, sr = librosa.load(data_folder + folder + '/' + filename, sr=None)
    output_name = 'test_stft'
    librosa.output.write_wav(output_name + '.wav', y, 44100) 

def SpecToAudio(X_mu_np, idx):
    Nfft = 4096
    nbFrames = 200
    S = np.repeat(X_mu_np[idx,:][np.newaxis].T,nbFrames,axis=1)
    S = np.append(S, np.zeros((Nfft/2-S.shape[0]+1,nbFrames)),axis=0)
    x = griffLim_stft(S)
    output_name = 'ReconAudio_'
    librosa.output.write_wav(output_name + 'gl.wav', x, 44100)
    librosa.output.write_wav(output_name + 'nophase.wav', librosa.istft(S), 44100)
    
def griffLim_stft(S): 
    Nfft = S.shape[0]*2 - 2
    S = np.log1p(np.abs(S))  
    a = np.zeros_like(S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) -np.pi
    for i in xrange(250):
        S = a*np.exp(1j*p)
        x = librosa.istft(S)
        spec = librosa.stft(x, n_fft = Nfft)
        p = np.angle(spec)
    return x

def MDCTToAudio(X_mu_np, idx, nbFrames):
    x = X_mu_np[idx,:]
    y = imdct4(x)
    output_name = 'ReconAudio_'
    librosa.output.write_wav(output_name + 'MDCT.wav', y, 44100)
    
def imdct4(x):
    N = x.shape[0]
    if N%2 != 0:
        raise ValueError("iMDCT4 only defined for even-length vectors.")
    M = N // 2
    N2 = N*2
    
    t = np.arange(0,M)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N2)
    c = np.take(x,2*t) + 1j * np.take(x,N-2*t-1)
    c = 0.5 * w * c
    c = np.fft.fft(c,M)
    c = ((8 / np.sqrt(N2))*w)*c
    
    rot = np.zeros(N2)
    
    rot[2*t] = np.real(c[t])
    rot[N+2*t] = np.imag(c[t])
    
    t = np.arange(1,N2,2)
    rot[t] = -rot[N2-t-1]
    
    t = np.arange(0,3*M)
    y = np.zeros(N2)
    y[t] = rot[t+M]
    t = np.arange(3*M,N2)
    y[t] = -rot[t-3*M]
    return y

def SpecToGranulator(X_mu_np, idx):
    Nfft = 4096
    S = X_mu_np[idx,:]
    S = np.append(S[np.newaxis].T, np.zeros((Nfft/2-S.shape[0]+1,1)),axis=0)
    x = np.fft.irfft(S, n=Nfft)
    s = pyo.Server(sr=44100, duplex = 0)
    s.boot()
    s.start()
    tab = pyo.DataTable(size=1024, init=x.tolist())
    tab.view()
    env = pyo.HannTable()
    pos = pyo.Phasor(1024/44100, 0, 1024)
    dur = pyo.Noise(.001, .1)
    g = pyo.Granulator(tab, env, 1, pos, dur, 24, mul=1).out()
    s.gui(locals(),exit=True)

    