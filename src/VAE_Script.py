# @pierrotechnique
# -*- coding: utf-8 -*-
#%%
import tensorflow.examples.tutorials.mnist as tetm
mnist = tetm.input_data.read_data_sets('../data/MNIST',one_hot=True)
#? How are tensorflow datasets structured?
#? What does the one_hot option do?

#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
#import matplotlib.gridspec as gridspec

dataset = np.load('../data/beta10_n100_amp24.npz')
inp = dataset['data']
lbls = dataset['lbls']
params = dataset['params']

#X_dim = mnist.train.images.shape[1] # Input variable (X) dimensionality
#? Why does this correspond to shape[1]?
X_dim = np.size(inp,axis=1)
Nit = 256
miniBatchSize = int(0.125*np.size(inp,axis=0))
z_dim = 3 # Latent space dimensionality
h_dim = X_dim/2 # Hidden layer (h) dimensionality
learningRate = 0.001

def extractMiniBatch(data,miniBatchSize):
    i = np.random.randint((np.size(data,axis=0)-miniBatchSize))
    X = data[i:(i+miniBatchSize)]
    return X

def varInit(size): # Initializes network input variables
    inpDim = size[0] # Assumes size = [inpDim,outDim]
    stdDev = 1./np.sqrt(inpDim/2.)
    return torch.autograd.Variable(
            torch.randn(size)*stdDev,requires_grad=True)

# Encoding
wXh = varInit(size=[X_dim,h_dim]) # Weights X into h
bXh = torch.autograd.Variable(torch.zeros(h_dim),requires_grad=True) # Bias

whz_mu = varInit(size=[h_dim,z_dim]) # Weights h into z (mu(X))
bhz_mu = torch.autograd.Variable(torch.zeros(z_dim),requires_grad=True)

whz_sigma = varInit(size=[h_dim,z_dim]) # Weights h into z (sigma(X))
bhz_sigma = torch.autograd.Variable(torch.zeros(z_dim),requires_grad=True)

def Q(X): # Two-layer encoder network
    h = torch.nn.functional.relu(torch.mm(X,wXh) + bXh)
    z_mu = torch.mm(h,whz_mu) + bhz_mu
    z_sigma = torch.mm(h,whz_sigma) + bhz_mu
    return z_mu,z_sigma

def zParam(mu,sigma): # z latent variable reparameterization trick
    eps = torch.autograd.Variable(torch.randn(miniBatchSize,z_dim))
#    return mu + torch.exp(torch.log(sigma)/2.)*eps
    return mu + torch.exp(0.5*sigma)*eps

# Decoding
wzh = varInit(size=[z_dim,h_dim]) # Weights z into h
bzh = torch.autograd.Variable(torch.zeros(h_dim),requires_grad=True)

whXo_mu = varInit(size=[h_dim,X_dim]) # Weights h into Xo_mu
bhXo_mu = torch.autograd.Variable(torch.zeros(X_dim),requires_grad=True)

whXo_sigma = varInit(size=[h_dim,X_dim]) # Weights h into Xo_sigma
bhXo_sigma = torch.autograd.Variable(torch.zeros(X_dim),requires_grad=True)

def P(z): # Two-layer decoder network
    h = torch.nn.functional.relu(torch.mm(z,wzh) + bzh)
    Xo_mu = torch.nn.functional.leaky_relu(torch.mm(h,whXo_mu) + bhXo_mu)
    Xo_sigma = torch.nn.functional.relu6(
            torch.mm(h,whXo_sigma) + bhXo_sigma)
#    X = torch.nn.functional.sigmoid(torch.mm(h,whX) + bhX)
    return Xo_mu,Xo_sigma

# Training
param = [wXh,bXh,whz_mu,bhz_mu,whz_sigma,bhz_sigma,wzh,bzh,
         whXo_mu,bhXo_mu,whXo_sigma,bhXo_sigma]
solver = torch.optim.Adam(param,lr=learningRate)

for it in xrange(Nit):
#    X,_ = mnist.train.next_batch(miniBatchSize)
    X = extractMiniBatch(inp,miniBatchSize)
    X = torch.autograd.Variable(torch.from_numpy(X))
    # Forward
    z_mu,z_sigma = Q(X)
    z = zParam(z_mu,z_sigma)
    Xo_mu,Xo_sigma = P(z)
    # Loss
#    reconLoss = torch.nn.functional.binary_cross_entropy(
#            Xout,X) #pytorch  
    reconLoss = -0.5*X_dim*torch.sum(2*np.pi*Xo_sigma)
    reconLoss -= torch.sum(torch.sum((X-Xo_mu).pow(2))/((2*Xo_sigma.exp())))
    reconLoss /= (miniBatchSize*X_dim) # Gaussian distribution log-likelihood
    # Takes into account the fact that sigma here represents log(sigma^2)
#    klLoss = 0.5*torch.sum(torch.exp(z_sigma)+(z_mu**2)-1.-z_sigma) # wiseodd
    klLoss = -0.5*torch.sum(-(z_sigma.exp())-(z_mu.pow(2))+1.+z_sigma)
    klLoss /= (miniBatchSize*X_dim) # pytorch
    loss = -reconLoss + klLoss
    # Backward
    loss.backward()
    # Update
    solver.step()
    # Clear parameter gradients (manually)
    for p in param:
        if p.grad is not None:
            data = p.grad.data
            p.grad = torch.autograd.Variable(
                    data.new().resize_as_(data).zero_())
    if ((it%int(0.125*Nit)) == 0):
        print('Loss: '+str(loss.data[0]))

print('Final loss: '+str(loss.data[0]))           
print('Finished training, brah!')

#%%

#samples = X.data.numpy()
#rndm = torch.autograd.Variable(torch.randn(miniBatchSize,z_dim))
#zSample = (z_sigma*rndm) + z_mu
zSample = zParam(z_mu,z_sigma)
samples,_ = P(zSample)
sample = samples[0].data.numpy()

#def PCA(X,k=2):
#    X_mu = torch.mean(X,0)
#    X = X - X_mu.expand_as(X)
#    U,S,V = torch.svd(X)
#    return torch.mm(X,torch.t(U[:][:k]))


def griffLim(S): # Griffin-Lim algorithm for signal reconstruction
    Nfft = S.shape[0]
    S = np.log1p(S) # ln(1 + S)
    a = np.exp(S) - 1
    p = 2*np.pi*np.random.random_sample(a.shape) - np.pi # Init random phase
    for i in range(Nfft): # Iterate to approximate phase
        S = a*np.exp(1j*p)
        x = np.fft.ifft(S,Nfft)
        p = np.angle(np.fft.fft(x,Nfft))
    return np.real(x)

#fig = plt.figure(figsize=(8,8))
#gs = gridspec.GridSpec(8,8)
#gs.update(wspace=0.1,hspace=0.1)
#
#for i,sample in enumerate(samples):
#    ax = plt.subplot(gs[i])
#    plt.axis('off')
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#    ax.set_aspect('equal')
#    plt.imshow(sample.reshape(28,28),cmap='Greys_r')

f = np.linspace(0,11025.,1024) # Frequency vector
plt.plot(f,sample)
plt.show()
plt.plot(f,inp[1])
plt.show()
#for i,sample in enumerate(samples):
#    if (i%500 == 0):
#        plt.plot(f,sample)
#        plt.show()
#        x = griffLim(sample)
#        plt.plot(x/max(abs(x)))
#        plt.show()
#        filename = '../data/recon2_'+str(i)+'.wav'
#        sf.write(filename,(x/max(abs(x))),11025,) # Save sound
    
#%%

# Signal reconstruction by Alexis Adoniadis

#sig = 0;
#t = np.linspace(0,_time,_time*_Fe)
#
#for band in range(_Nfft/_factor):
##Take the mean of the band's frequency
#fosc = (band+0.5)/len(_spectrum)*_Fe/_factor;
#sig = sig + np.sin(2*np.pi*fosc*t)*_spectrum[band];
#
#return sig/max(abs(sig))

# Griffin-Lim algorithm courtesy of Victor Wetzel by way of Philippe Esling
        
#def griffinlim(self, S):
#        """Returns a sound reconstructed from a spectrogram with NFFT points,
#        Griffin-Lim algorithm
#        INPUT:
#            - S: spectrogram (array)
#        OUTPUT:
#            - x: signal"""
#        # ---- INIT ----
#        # Create empty STFT & Back from log amplitude
#        n_fft = S.shape[0]
#        S = np.log1p(np.abs(S))
#
#        #a = np.zeros_like(S)
#        a = np.exp(S) - 1
#        
#        # Phase reconstruction
#        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
#        
#        # LOOP: iterative reconstruction
#        for i in range(100):
#            S = a * np.exp(1j*p)
#            x = lib.istft(S)
#            p = np.angle(lib.stft(x, self.n_fft))
#    
#        return x