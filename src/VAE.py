# VAE = { X -> Encoder-> Z -> Decoder ~> X }
# X_dim (int): dimension of X (input) - e.g. 513
# Z_dim (int): dimension of Z (latent) - e.g. 6
# IOh_dims_Enc (int*): hidden layers' IO encoder dims - e.g. [513 128 6]
# IOh_dims_Dec (int*): hidden layers' IO decoder dims - e.g. [6 128 513]
# NL_types_*** (string*): types on nonLinearity for each layer of Encoder/Decoder
#  e.g. NL_types_Enc = ['relu'] ; NL_types_Dec = ['relu', 'sigmoid']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from EncoderDecoder import Encoder, Decoder


class VAE(nn.Module):

    def __init__(self, X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size=64, beta=1):

        # superclass init
        super(VAE, self).__init__()
        self.created = False

        self.encoder = Encoder(X_dim, IOh_dims_Enc, Z_dim)
        self.decoder = Decoder(Z_dim, IOh_dims_Dec, X_dim)
        if (self.encoder.created == False or self.decoder.created == False):
            print "ERROR_VAE: Wrong encoder/decoder structure"
            return None

        # check if NL_types length & layers number are the same
        self.NL_funcE = NL_types_Enc
        self. NL_funcD = NL_types_Dec
        # in Encoder
        if len(self.NL_funcE) != self.encoder.nb_h:
            print "ERROR_VAE: not enough or too many NL functions in encoder"
            return None
        # in Decoder
        if len(self.NL_funcD) != self.decoder.nb_h:
            print "ERROR_VAE: not enough or too many NL functions in decoder"
            return None

        # check if each elemt of NL_types exists in 'torch.nn.functional' module
        # in Encoder
        for index_h in range(self.encoder.nb_h):
            try:
                getattr(F, self.NL_funcE[index_h])
            except AttributeError:
                pass
                print "ERROR_VAE: Wrong encoder NL function name"
                return None
        # in Decoder
        for index_h in range(self.decoder.nb_h):
            try:
                getattr(F, self.NL_funcD[index_h])
            except AttributeError:
                pass
                print "ERROR_VAE: Wrong encoder NL function name"
                return None

        # learning rate
        self.lr = 1e-3
        # minibacth size
        self.mb_size = mb_size
        self.beta = beta

        self.z_mu = None
        self.z_logSigma = None
        self.z = None
        self.X_sample = None

        self.created = True

    def forward(self, X):
        if self.created == False:
            print "ERROR_VAE_forward: VAE not correctly created"
            return None
        # compute z from X
        # the size -1 is inferred from other dimensions
        self.z = self.encode(X.view(-1, self.encoder.dimX))
        # compute X_sample from z
        self.X_sample = self.decode(self.z)
        return self.X_sample

    def encode(self, X):
        # first layer takes X in input
        var_h = getattr(F, self.NL_funcE[0])(self.encoder.h_layers[0](X))
        # then var_h goes through deeper layers
        for i in range(self.encoder.nb_h - 1):
            var_h = getattr(F, self.NL_funcE[
                            i + 1])(self.encoder.h_layers[i + 1](var_h))

        # get z's mu and logSigma
        self.z_mu = self.encoder.h_mu(var_h)
        self.z_logSigma = self.encoder.h_logSigma(var_h)

        # reparametrization trick
        return self.reparametrize()

    def decode(self, z):
        # first layer takes z in input
        var_h = getattr(F, self.NL_funcD[0])(self.decoder.h_layers[0](z))
        # then var_h goes through deeper layers
        for i in range(self.decoder.nb_h - 1):
            var_h = getattr(F, self.NL_funcD[i + 1])(
                self.decoder.h_layers[i + 1](var_h))

        return var_h

    def reparametrize(self):
        # eps = Variable(torch.randn(self.mb_size, self.encoder.dimZ))
        # return self.z_mu + torch.exp(self.z_logSigma / 2) * eps
        std = self.z_logSigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.z_mu)

    def loss(self, X):
        bce = F.binary_cross_entropy(
            self.X_sample, X.view(-1, self.encoder.dimX))
        kld = - 0.5 * torch.sum(1 + self.z_logSigma -
                                self.z_mu.pow(2) - self.z_logSigma.exp())
        # Normalise by same number of elements as in reconstruction
        kld /= self.mb_size * self.encoder.dimX
        loss = bce + self.beta * kld
        return loss

        # def save(self, name):

        # def load(self, name):
