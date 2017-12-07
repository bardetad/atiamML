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

dataset = np.load('../data/beta10_n48.npz')
inp = dataset['data']
lbls = dataset['lbls']
params = dataset['params']

#X_dim = mnist.train.images.shape[1] # Input variable (X) dimensionality
#? Why does this correspond to shape[1]?
X_dim = np.size(inp,axis=1)
miniBatchSize = int(0.25*np.size(inp,axis=0))
z_dim = 2 # Latent space dimensionality
h_dim = miniBatchSize # Hidden layer (h) dimensionality
Nit = 1000
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
    return mu + torch.exp(sigma/2.)*eps

# Decoding
wzh = varInit(size=[z_dim,h_dim]) # Weights z into h
bzh = torch.autograd.Variable(torch.zeros(h_dim),requires_grad=True)

whX = varInit(size=[h_dim,X_dim]) # Weights h into X
bhX = torch.autograd.Variable(torch.zeros(X_dim),requires_grad=True)

def P(z,wzh): # Two-layer decoder network
    h = torch.nn.functional.relu(torch.mm(z,wzh) + bzh)
    X = torch.nn.functional.sigmoid(torch.mm(h,whX) + bhX)
    return X

# Training
param = [wXh,bXh,whz_mu,bhz_mu,whz_sigma,bhz_sigma,wzh,bzh,whX,bhX]
solver = torch.optim.Adam(param,lr=learningRate)

for it in xrange(Nit):
#    X,_ = mnist.train.next_batch(miniBatchSize)
    X = extractMiniBatch(inp,miniBatchSize)
    X = torch.autograd.Variable(torch.from_numpy(X))
    # Forward
    z_mu,z_sigma = Q(X)
    z = zParam(z_mu,z_sigma)
    Xout = P(z,wzh)
    # Loss
#    reconLoss = torch.nn.functional.binary_cross_entropy(
#            Xout,X,size_average=False)/miniBatchSize # wiseodd
    #? What does size_average do in this function?
    reconLoss = torch.nn.functional.binary_cross_entropy(
            Xout,X) #pytorch
#    klLoss = 0.5*torch.sum(torch.exp(z_sigma)+(z_mu**2)-1.-z_sigma) # wiseodd
    klLoss = -0.5*torch.sum(-(z_sigma.exp())-(z_mu.pow(2))+1.+z_sigma)
    klLoss /= (miniBatchSize*X.size()[1]) # pytorch
    loss = reconLoss + klLoss
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
    if ((it % (0.1*Nit)) == 0):
        print('Loss: '+str(loss.data[0]))
            
print('Finished training, brah!')

#%%

#samples = X.data.numpy()
rndm = torch.autograd.Variable(torch.randn(miniBatchSize,z_dim))
zSample = (z_sigma*rndm) + z_mu
samples = P(zSample,wzh)
samples = samples.data.numpy()

#def PCA(X,k=2):
#    X_mu = torch.mean(X,0)
#    X = X - X_mu.expand_as(X)
#    U,S,V = torch.svd(X)
#    return torch.mm(X,torch.t(U[:][:k]))
#
#
#z_mu2d = torch.autograd.Variable(PCA(z_mu.data,2))
#z_sigma2d = torch.autograd.Variable(PCA(z_sigma.data,2))
#rndm_2d = torch.autograd.Variable(torch.randn(miniBatchSize,2))
#zSample_2d = (z_sigma2d*rndm_2d) + z_mu2d
#wzh_2d = torch.autograd.Variable(PCA(torch.t(wzh.data),2))
#samples_2d = P(zSample_2d,torch.t(wzh_2d))
#samples_2d = samples_2d.data.numpy()

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
for i,sample in enumerate(samples):
    if (i%10 == 0):
        plt.plot(f,sample)
        plt.show()
        x = griffLim(sample)
        plt.plot(x)
        plt.show()
#        za = int(z[i][0].data.numpy())
#        zb = int(z[i][1].data.numpy())
#        zc = int(z[i][2].data.numpy())
#        zs = 'za'+str(za)+'_zb'+str(zb)+'_zc'+str(zc)
#        filename = '../data/recon_'+zs+'.wav'
#        sf.write(filename,(x/max(abs(x))),44100,) # Save sound
    
#%%

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