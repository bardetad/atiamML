# X -> Encoder -> Q(Z|X)
# inputDim (int): dimension of X (input) - e.g. 513
# dimValues (int*): hidden layers' IO dimensions - e.g. [513 128 6] for a 1 hLayer + 1(mu-sigma) layer NN
# outputDim (int): dimension of Z (output) - e.g. 6

import torch
import torch.nn as nn


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
        #& create hidden layers of the NN (private)
        self.inDim_h = []
        self.outDim_h = []
        self.h_layers = nn.ModuleList()
        for index_h in range(self.nb_h):
            self.inDim_h.append(dimValues[index_h])
            self.outDim_h.append(dimValues[index_h + 1])
            self.h_layers.append(
                nn.Linear(self.inDim_h[index_h], self.outDim_h[index_h]))

        # LAST LAYER is made by hand whereas for decoder IT'S NOT
        self.h_mu = nn.Linear(self.outDim_h[self.nb_h - 1], self.dimZ)
        self.h_logSigma = nn.Linear(self.outDim_h[self.nb_h - 1], self.dimZ)

        self.created = True

    def getInfo(self):
        print('\nEncoder net : ')
        for idx in range(self.nb_h):
            print(str(idx) + ' -> ' + str(self.h_layers[idx]))
        print('mu ->' + str(self.h_mu))
        print('logSigma ->' + str(self.h_logSigma))


# Z -> Decoder -> P(X|Z)
# inputDim (int): dimension of Z (input) - e.g. 6
# dimValues (int*): hidden layers' IO dimensions - e.g. [6 128 513] for a 1 hLayer NN
# outputDim (int): dimension of approximate X (output) - e.g. 513
class Decoder(nn.Module):

    def __init__(self, inputDim, dimValues, outputDim):

        # superclass init
        super(Decoder, self).__init__()
        self.created = False

        # dimension of inputs Z
        self.dimZ = inputDim

        # dimension of outputs X
        self.dimX = outputDim

        # Decoder NN structure:
        # define HIDDEN layers number
        self.nb_h = len(dimValues) - 1
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
        self.h_layers = nn.ModuleList()
        for index_h in range(self.nb_h):
            self.inDim_h.append(dimValues[index_h])
            self.outDim_h.append(dimValues[index_h + 1])
            self.h_layers.append(
                nn.Linear(self.inDim_h[index_h], self.outDim_h[index_h]))

        self.created = True

    def getInfo(self):
        print('\nDecoder net : ')
        for idx in range(self.nb_h):
            print(str(idx) + ' -> ' + str(self.h_layers[idx]))
