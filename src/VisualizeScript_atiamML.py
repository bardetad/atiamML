#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:36:40 2017

@author: bavo
"""
import os
#os.chdir('/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/1_VAE_model/')
os.chdir('/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/atiamML/src/')

#%%                                             Load data iterator and model
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import imp
VAE_mod = imp.load_source('VAE_model', 'VAE_model.py')
#NPZ_Dataset_mod = imp.load_source('ManageDataset', '../atiamML/src/dataset/ManageDataset.py')
#import VAE_model as VAE_mod
NPZ_Dataset_mod = imp.load_source('ManageDataset', './dataset/ManageDataset.py')


# Load Model
#X_dim = 1025
#Z_dim = 4
#IOh_dims_Enc = [1025, 400, 4]
#IOh_dims_Dec = [4, 400, 1025]
#NL_types_Enc = ['relu6']
#NL_types_Dec = ['relu6']
#mb_size = 50
#beta = 1
#lr = 1e-3
#bernoulli = False
#gaussian = True
#VAE_test = VAE_mod.VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, beta, lr, bernoulli, gaussian)
model_folder = '/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/atiamML/data/savedVAE/'
model_name = 'Spectrumsdataset2BVK_NPZ_E<1025-relu6-400-muSig-4>_D<4-relu6-400-muSig-1025>_beta1_mb50_lr0dot0001_ep60'
#model_name = 'Spectrumsdataset3BVK_NPZ_E<1025-relu6-400-muSig-5>_D<5-relu6-400-muSig-1025>_beta1_mb50_lr0dot0001_ep60'

data_folder = '/media/bavo/1E84369D843676FD/Users/bavov/Documents/ATIAM/4_Informatique/MachineLearning_Project/Datasets/'
data_set = 'Spectrums/'
data_name = 'dataset2BVK.npz'


VAE_test = VAE_mod.loadVAE(model_name, model_folder)

# Load data
toydataset_1 = NPZ_Dataset_mod.NPZ_Dataset(npz_file=data_name, root_dir= data_folder + data_set)
trainloader = torch.utils.data.DataLoader(toydataset_1, batch_size=VAE_test.mb_size, shuffle=True, num_workers=4)

#%%                                             Perform one run of VAE
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
#%%                              Perform full run of VAE and calculate loss over each sample
#                           Eval 1: Calculate mean and variance for loss/dataset/factor of variation

loss_dataset = np.zeros((len(trainloader.dataset),3))
label_ar = np.zeros((1,VAE_test.encoder.dimZ))            

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

if VAE_test.encoder.dimZ == 4:
    label_steps = [10,20,10,10]
    start_step = [0,0,1,1]
else:
    label_steps = [8,7,8,10,10]
    start_step = [1,1,1,1,1]

box_mat = []
for i in range(VAE_test.encoder.dimZ):
    box_vec = np.zeros((label_steps[i],1))
    for j in range(label_steps[i]):
        bool_ar = label_ar[:,i] == j + start_step[i]
        mean = np.mean(loss_dataset[:,0]*bool_ar.T)
        box_vec[j,0] = mean
    box_mat.append(box_vec)

fig = plt.figure(1, figsize=(9, 6))  
ax1 = fig.add_subplot(111)
bp = plt.boxplot(box_mat)
ax1.set_title('Loss for 4 variational factors')
ax1.set_ylabel('Loss')
ax1.set_xticklabels(['Inharmonicity', '# Harmonics', 'Filter Q', 'Filter freq.'])
#ax1.set_xticklabels(['Carrier', 'Ratio', 'Index', 'Filter Q', 'Filter freq.'])
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
#ax1.set_xlim(0.5, 4)
#ax1.set_ylim(-1, 1)
plt.show()

#%%                              Eval 2: Evaluate loss evolution
loss_vector = np.load(model_folder + data_set[:-1] + '_' + data_name + 'loss_.npy')

fig, ax = plt.subplots()
x_axis = range(5,60)
ax.plot(x_axis, loss_vector[5:,0], 'k--', label='Total loss')
ax.plot(x_axis, loss_vector[5:,1], 'k:', label='Reconstruction loss')
ax.plot(x_axis, loss_vector[5:,2], 'k', label='Kull/Leib loss')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

#%%                              Test 1: compare in- and output
import librosa

idx = 1
image_out = X_mu_np[idx]
image_in = X_np[idx]
label_in = labels_np[idx]
images = [image_in, image_out]

# Listen to input
filename = 'set1_' + str(int(label_in[0])) + '_' + str(int(label_in[1])) + '_' + str(int(label_in[2])) + '_' + str(int(label_in[3])) + '_.wav'
#filename = 'set1_' + str(int(label_in[0])) + '_' + str(int(label_in[1])) + '_' + str(int(label_in[2])) + '_' + str(int(label_in[3])) + '_' + str(int(label_in[4])) + '_.wav'
dataset = 'toy_dataset_2/'
y, sr = librosa.load(data_folder + dataset + filename, sr=None)
output_name = 'test_stft'
librosa.output.write_wav(output_name + '.wav', y, 44100)

# Plot
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(images[0])
ax2 = fig.add_subplot(212)
ax2.plot(images[1])

#%%                             Test 2: analyse z-space (random z)
z = Variable(torch.randn(VAE_test.mb_size, VAE_test.encoder.dimZ))

VAE_test.decode(z)
X_mu = VAE_test.X_mu
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
import librosa

def griffLim_stft(S): 
    Nfft = S.shape[0]
    S = np.append(np.zeros((1,100)),S,axis=0)
    a = np.abs(S)
    p = 2*np.pi*np.random.random_sample(a.shape) 
    for i in xrange(250):
        S = a*np.exp(1j*p)
        x = librosa.istft(S)
        spec = librosa.stft(x, Nfft)
        p = np.angle(spec)
    return x, S

X_mu_np = X_mu.data.numpy()
samples_recon = X_mu_np

sig_spec = np.repeat(samples_recon[80,:][np.newaxis].T,200,axis=1)
S = sig_spec
Nfft = 4096
S = np.append(S, np.zeros((4096/2-1025+1,200)),axis=0)

#t = np.linspace(0,2,2*44100)
#test = np.sin(2*np.pi*440*t)
##D = np.fft.fft(test[2000:],Nfft)[0:Nfft/2+1]
##log_D = 20*np.log10(np.abs(D)**2)
##S = np.repeat(log_D[np.newaxis].T,430,axis=1)
#D2 = librosa.stft(test, 4096)[0:1025,5]
#S = np.repeat(D2[np.newaxis].T,430,axis=1)
#S = np.append(S, np.zeros((4096/2-1025+1,430)),axis=0)

S = np.log1p(np.abs(S))  
a = np.zeros_like(S)
a = np.exp(S) - 1
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(200):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, n_fft = Nfft))
    
output_name = 'test_stft'
librosa.output.write_wav(output_name + '.wav', y, 44100)
librosa.output.write_wav(output_name + '2.wav', librosa.istft(a), 44100)

#%%
import pyo

s = pyo.Server(sr=44100, duplex = 0)
s.boot()
s.start()
tab = pyo.DataTable(size=1024, init=sig_mul.tolist())
tab.view()
env = pyo.HannTable()
pos = pyo.Phasor(1024/44100, 0, 1024)
dur = pyo.Noise(.001, .1)
g = pyo.Granulator(tab, env, 1, pos, dur, 24, mul=1).out()
s.gui(locals(),exit=True)

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
VAE_test.decode(z)
X_mu = VAE_test.X_mu
samples = X_mu.data.numpy()

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
toydataset_1 = VAE_model.ToyDataset_1(npz_file='toy_dataset_1_BVK.npz', 
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

VAE_test.encode(X)
z_mu = VAE_test.z_mu
z_mu = z_mu.data.numpy()

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

    