# X -> Encoder -> Q(Z|X)
# inputDim (int): dimension of X (input) - e.g. 513
# dimValues (int*): hidden layers' IO dimensions - e.g. [513 128 6] for a 1 hLayer + 1(mu-sigma) layer NN
# outputDim (int): dimension of Z (output) - e.g. 6

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Encoder(nn.Module):

    def __init__(self, inputDim, dimValues, outputDim):

        # superclass init
        super(Encoder, self).__init__()
        self.created = False

        # dimension of inputs X
        self.dimX = inputDim

        # dimension of outputs Z
        self.dimZ = outputDim

        # Encoder NN structure:
        # define HIDDEN layers number
        self.nb_h = len(dimValues) - 2
        # check if args match
        if self.nb_h < 1:
            print "ERROR_Encoder: Not enough dimension values"
            return None
        elif self.dimX != dimValues[0]:
            print "ERROR_Encoder: X & NN input dimensions mismatched"
            return None
        elif self.dimZ != dimValues[len(dimValues) - 1]:
            print "ERROR_Encoder: Z & NN output dimensions mismatched"
            return None

        # store IO dimensions for each layers in a list
        #& create initialized hidden layers of the NN
        self.inDim_h = []
        self.outDim_h = []

        self.weights_h = []
        self.bias_h = []
        self.weight_mu = None
        self.bias_mu = None
        self.weight_logSigma = None
        self.bias_logSigma = None

        for index_h in range(self.nb_h):
            self.inDim_h.append(dimValues[index_h])
            self.outDim_h.append(dimValues[index_h + 1])
            self.weights_h.append(xavier_init(
                size=[self.inDim_h[index_h], self.outDim_h[index_h]]))
            self.bias_h.append(Variable(torch.zeros(
                self.outDim_h[index_h]), requires_grad=True))

        # LAST LAYER is made by hand whereas for bernoulli DECODER IT'S NOT
        self.weight_mu = xavier_init(
            size=[self.outDim_h[self.nb_h - 1], self.dimZ])
        self.bias_mu = Variable(torch.zeros(self.dimZ), requires_grad=True)
        self.weight_logSigma = xavier_init(
            size=[self.outDim_h[self.nb_h - 1], self.dimZ])
        self.bias_logSigma = Variable(
            torch.zeros(self.dimZ), requires_grad=True)

        self.created = True

    def getInfo(self):
        print('\nEncoder net : ')
        for idx in range(self.nb_h):
            print('layer ' + str(idx) + ': size ' + str(self.weights_h[idx].size(0)))

#------------------------------------------------------------------------------

# Z -> Decoder -> P(X|Z)
# inputDim (int): dimension of Z (input) - e.g. 6
# dimValues (int*): hidden layers' IO dimensions - e.g. [6 128 513] for a 1 hLayer NN
# outputDim (int): dimension of approximate X (output) - e.g. 513
# bernoulli (bool) : if true, decode directly from NN
# gaussian (bool) : if true, same structure as encoder with a mu and logSigma


class Decoder(nn.Module):

    def __init__(self, inputDim, dimValues, outputDim, bernoulli=True, gaussian=False):

        # superclass init
        super(Decoder, self).__init__()
        self.created = False

        # dimension of inputs Z
        self.dimZ = inputDim

        # dimension of outputs X
        self.dimX = outputDim

        # decoder type flags
        self.bernoulli = bernoulli
        self.gaussian = gaussian

        # Decoder NN structure:
        # define HIDDEN layers number
        if self.bernoulli and not self.gaussian:
            self.nb_h = len(dimValues) - 1
        elif self.gaussian and not self.bernoulli:
            self.nb_h = len(dimValues) - 2
        else:
            print("ERROR_Decoder: Decoder type unknown")
            raise
        # check if args match
        if self.nb_h < 1:
            print "ERROR_Decoder: Not enough dimension values"
            return None
        elif self.dimZ != dimValues[0]:
            print "ERROR_Decoder: Z & NN input dimensions mismatched"
            return None
        elif self.dimX != dimValues[len(dimValues) - 1]:
            print "ERROR_Decoder: X & NN output dimensions mismatched"
            return None

        # store IO dimensions for each layers in a list
        #& create hidden layers of the NN (private)
        self.inDim_h = []
        self.outDim_h = []

        self.weights_h = []
        self.bias_h = []

        if gaussian and not bernoulli:
            self.weight_mu = None
            self.bias_mu = None
            self.weight_logSigma = None
            self.bias_logSigma = None

        for index_h in range(self.nb_h):
            self.inDim_h.append(dimValues[index_h])
            self.outDim_h.append(dimValues[index_h + 1])
            self.weights_h.append(xavier_init(
                size=[self.inDim_h[index_h], self.outDim_h[index_h]]))
            self.bias_h.append(Variable(torch.zeros(
                self.outDim_h[index_h]), requires_grad=True))

        if gaussian and not bernoulli:
            # LAST LAYER is made by hand whereas for gaussian decoder
            self.weight_mu = xavier_init(
                size=[self.outDim_h[self.nb_h - 1], self.dimX])
            self.bias_mu = Variable(torch.zeros(self.dimX), requires_grad=True)
            self.weight_logSigma = xavier_init(
                size=[self.outDim_h[self.nb_h - 1], self.dimX])
            self.bias_logSigma = Variable(
                torch.zeros(self.dimX), requires_grad=True)

        self.created = True

    def getInfo(self):
        print('\nDecoder net : ')
        for idx in range(self.nb_h):
            print('layer ' + str(idx) + ': size ' + str(self.weights_h[idx].size(0)))

#------------------------------------------------------------------------------

# Xavier init


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1 / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)
