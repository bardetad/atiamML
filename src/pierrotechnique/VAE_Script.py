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
import sys
#import matplotlib.gridspec as gridspec

dataset = np.load('../data/beta10_n100_fm24.npz')
inp = dataset['data']
lbls = dataset['lbls']
params = dataset['params']

#X_dim = mnist.train.images.shape[1] # Input variable (X) dimensionality
#? Why does this correspond to shape[1]?
X_dim = np.size(inp,axis=1)
Ndata = np.size(inp,axis=0)
Nit = 32
miniBatchSize = int(float(Ndata)/Nit)
Nepoch = int(float(Ndata)/miniBatchSize) + 1
z_dim = 3 # Latent space dimensionality
h_dim = X_dim/2 # Hidden layer (h) dimensionality
learningRate = 0.001

def extractMiniBatch(data,miniBatchSize):
    i = np.random.randint((np.size(data,axis=0)-miniBatchSize))
    np.random.shuffle(data) # Make this actually random!
    X = data[i:(i+miniBatchSize)] # NO NO NO!
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

for epoch in xrange(Nepoch):
    X = extractMiniBatch(inp,miniBatchSize)
    X = torch.autograd.Variable(torch.from_numpy(X))
    for it in xrange(Nit):
        # Forward
        z_mu,z_sigma = Q(X)
        z = zParam(z_mu,z_sigma)
        Xo_mu,Xo_sigma = P(z)
        # Loss 
        reconLoss = -0.5*X_dim*torch.sum(2*np.pi*Xo_sigma)
#        reconLoss -= torch.sum(
#                torch.sum((X-Xo_mu).pow(2))/((2*Xo_sigma.exp())))
        # Test without second sum
        reconLoss -= torch.sum(((X-Xo_mu).pow(2))/((2*Xo_sigma.exp())))
        reconLoss /= (miniBatchSize*X_dim) 
        # Gaussian distribution log-likelihood
        # Takes into account the fact that sigma here represents log(sigma^2)
        klLoss = -0.5*torch.sum(-(z_sigma.exp())-(z_mu.pow(2))+1.+z_sigma)
        beta = float(it*epoch)/(Nit*Nepoch)
        klLoss *= beta
        klLoss /= (miniBatchSize*X_dim)
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
            print('Epoch '+str(epoch)+'/'+str(
                    Nepoch)+', iteration '+str(it)+'/'+str(Nit))
            print('Reconst. loss (Gaussian log-likelihood): '+str(
                    reconLoss.data[0]))
            print('K.L. divergence: '+str(klLoss.data[0]))
            print('Total loss: '+str(loss.data[0]))
    print('Total epoch '+str(epoch)+'/'+str(
            Nepoch)+' loss: '+str(loss.data[0]))

print('Total final loss: '+str(loss.data[0]))           
print('Finished training, brah!')

#%%
zSample = zParam(z_mu,z_sigma)
zMax = zSample.max(dim=0)[0]
zMin = zSample.min(dim=0)[0]
Nex = 4
zDiag = np.array([
        np.linspace(zMin[0].data.numpy(),zMax[0].data.numpy(),Nex),
        np.linspace(zMin[1].data.numpy(),zMax[1].data.numpy(),Nex),
        np.linspace(zMin[2].data.numpy(),zMax[2].data.numpy(),Nex)])

def latentVar(val,miniBatchSize):
    dim = val.size
    z_np = np.tile(val,miniBatchSize).reshape((miniBatchSize,dim))
    z_t = torch.autograd.Variable(torch.from_numpy(z_np))
    return z_t

f = np.linspace(0,11025.,1024) # Frequency vector

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

X,_ = P(latentVar(np.array([0.,0.,4.],dtype='float32'),miniBatchSize))
#X,_ = P(latentVar(zDiag[:,0],miniBatchSize))
X = X.mean(dim=0).data.numpy()
for k in range(len(X)):
    if (X[k] < 0.):
        X[k] = sys.float_info.epsilon
plt.plot(f,20*np.log10(X))
plt.plot(f,X)
plt.ylim((-67,0))
plt.show()
x = griffLim(X)
plt.plot(x)
plt.show()

#for i in range(Nex):
#    X,_ = P(latentVar(zDiag[:,i],miniBatchSize))
#    X = X.mean(dim=0).data.numpy()
#    for k in range(len(X)):
#        if (X[k] < 0.):
#            X[k] = sys.float_info.epsilon
#    plt.plot(f,20*np.log10(X))
#    plt.plot(f,X)
#    plt.ylim((-67,0))
#    plt.show()
#    x = griffLim(X)
#    plt.plot(x)
#    plt.show()
#    filename = '../data/recon_zDiag_'+str(i)+'.wav'
#    sf.write(filename,(x/max(abs(x))),44100) # Save sound

#def PCA(X,k=2):
#    X_mu = torch.mean(X,0)
#    X = X - X_mu.expand_as(X)
#    U,S,V = torch.svd(X)
#    return torch.mm(X,torch.t(U[:][:k]))

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