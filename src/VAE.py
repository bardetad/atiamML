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
from EncoderDecoder import Encoder, Decoder


class VAE(nn.Module):

    def __init__(self, X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec):

        # superclass init
        super(VAE, self).__init__()
        self.created = False

        self.encoder = Encoder(X_dim, IOh_dims_Enc, Z_dim)
        self.decoder = Decoder(Z_dim, IOh_dims_Dec, X_dim)
        if (self.encoder.created == False or self.decoder.created == False):
            print "ERROR_VAE: Wrong encoder/decoder structure"
            return None

        # check if NL_types length & layers number are the same
        # in Encoder
        if len(NL_types_Enc) != self.encoder.nb_h:
            print "ERROR_VAE: not enough or too many NL functions in encoder"
            return None
        # in Decoder
        if len(NL_types_Dec) != self.decoder.nb_h:
            print "ERROR_VAE: not enough or too many NL functions in decoder"
            return None

        # check if each elemt of NL_types exists in 'torch.nn.functional' module
        # in Encoder
        for index_h in range(self.encoder.nb_h):
            try:
                getattr(F, NL_types_Enc[index_h])
            except AttributeError:
                pass
                print "ERROR_VAE: Wrong encoder NL function name"
                return None
        # in Decoder
        for index_h in range(self.decoder.nb_h):
            try:
                getattr(F, NL_types_Dec[index_h])
            except AttributeError:
                pass
                print "ERROR_VAE: Wrong encoder NL function name"
                return None

        self.lr = 1e-3
        self.mb_size = 10
        self.beta = 0

        self.created = True

    # def forward(self):

    # def encode(self, X):

    # def reparameterize(self, mu, logSigma):

    # def Loss(self,X_sample, X, z_mu, z_var):
