import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable

from torchvision.utils import save_image
import torch.utils.data
from torchvision import datasets, transforms

from EncoderDecoder import Encoder, Decoder

#------------------------------------------------------------------------------
# VAE = { X -> Encoder-> Z -> Decoder ~> X }
# X_dim (int): dimension of X (input) - e.g. 513
# Z_dim (int): dimension of Z (latent) - e.g. 6
# IOh_dims_Enc (int*): hidden layers' IO encoder dims - e.g. [513 128 6]
# IOh_dims_Dec (int*): hidden layers' IO decoder dims - e.g. [6 128 513]
# NL_types_*** (string*): types on nonLinearity for each layer of Encoder/Decoder
#  e.g. NL_types_Enc = ['relu'] ; NL_types_Dec = ['relu', 'sigmoid']


class VAE(nn.Module):

    def __init__(self, X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size=64, beta=4, lr=1e-3, bernoulli=True, gaussian=False):

        # superclass init
        super(VAE, self).__init__()
        self.created = False

        self.IOh_dims_Enc = IOh_dims_Enc
        self.IOh_dims_Dec = IOh_dims_Dec

        self.encoder = Encoder(X_dim, self.IOh_dims_Enc, Z_dim)
        self.decoder = Decoder(Z_dim, self.IOh_dims_Dec,
                               X_dim, bernoulli, gaussian)
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
        self.lr = lr
        # minibacth size
        self.mb_size = mb_size
        self.beta = beta

        self.z_mu = None
        self.z_logSigma = None
        self.z = None
        self.X_sample = None
        self.X_mu = None
        self.X_logSigma = None

        self.parameters = []
        for nb_h in range(self.encoder.nb_h):
            self.parameters.append(self.encoder.weights_h[nb_h])
            self.parameters.append(self.encoder.bias_h[nb_h])
        self.parameters.append(self.encoder.weight_mu)
        self.parameters.append(self.encoder.bias_mu)
        self.parameters.append(self.encoder.weight_logSigma)
        self.parameters.append(self.encoder.bias_logSigma)

        for nb_h in range(self.decoder.nb_h):
            self.parameters.append(self.decoder.weights_h[nb_h])
            self.parameters.append(self.decoder.bias_h[nb_h])

        if self.decoder.gaussian and not self.decoder.bernoulli:
            self.parameters.append(self.decoder.weight_mu)
            self.parameters.append(self.decoder.bias_mu)
            self.parameters.append(self.decoder.weight_logSigma)
            self.parameters.append(self.decoder.bias_logSigma)

        # flags on vae creation
        self.created = True
        self.trained = False

        # flags on vae state
        self.saved = False
        self.loaded = False

        self.epoch_nb = 0

    def forward(self, X):
        if self.created == False:
            print "ERROR_VAE_forward: VAE not correctly created"
            return None
        # compute z from X
        # the size -1 is inferred from other dimensions
        self.z = self.encode(X.view(-1, self.encoder.dimX))

        self.decode(self.z)

    def encode(self, X):
        # first layer takes X in input
        var_h = getattr(F, self.NL_funcE[0])(torch.mm(X, self.encoder.weights_h[
            0]) + self.encoder.bias_h[0].repeat(X.size(0), 1))
        # then var_h goes through deeper layers
        for i in range(self.encoder.nb_h - 1):
            var_h = getattr(F, self.NL_funcE[
                            i + 1])(torch.mm(var_h, self.encoder.weights_h[i + 1]) + self.encoder.bias_h[i + 1].repeat(var_h.size(0), 1))

        # get z's mu and logSigma
        self.z_mu = torch.mm(var_h, self.encoder.weight_mu) + \
            self.encoder.bias_mu.repeat(var_h.size(0), 1)
        self.z_logSigma = torch.mm(var_h, self.encoder.weight_logSigma) + \
            self.encoder.bias_logSigma.repeat(var_h.size(0), 1)

        # reparametrization trick
        return self.reparametrize()

    def decode(self, z):
        # first layer takes z in input
        var_h = getattr(F, self.NL_funcD[0])(torch.mm(z, self.decoder.weights_h[
            0]) + self.decoder.bias_h[0].repeat(z.size(0), 1))
        # then var_h goes through deeper layers
        for i in range(self.decoder.nb_h - 1):
            var_h = getattr(F, self.NL_funcD[
                            i + 1])(torch.mm(var_h, self.decoder.weights_h[i + 1]) + self.decoder.bias_h[i + 1].repeat(var_h.size(0), 1))

        if self.decoder.bernoulli and not self.decoder.gaussian:
            self.X_sample = var_h
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            # get X_sample's mu and logSigma
            self.X_mu = torch.mm(var_h, self.decoder.weight_mu) + \
                self.decoder.bias_mu.repeat(var_h.size(0), 1)
            self.X_logSigma = torch.mm(
                var_h, self.decoder.weight_logSigma) + self.decoder.bias_logSigma.repeat(var_h.size(0), 1)
        else:
            print("ERROR VAE: wrong decoder type")
            raise

    def reparametrize(self):
        std = self.z_logSigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.z_mu)

    def loss(self, X):
        if self.decoder.bernoulli and not self.decoder.gaussian:
            # Bernoulli
            recon = F.binary_cross_entropy(
                self.X_sample, X.view(-1, self.encoder.dimX))
            recon /= self.mb_size * self.encoder.dimX
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            # gaussian
            X_sigma = torch.exp(self.X_logSigma)
            firstTerm = torch.log(2 * np.pi * X_sigma)
            secondTerm = ((self.X_mu - X)**2) / X_sigma
            recon = 0.5 * torch.sum(firstTerm + secondTerm)
            recon /= (self.mb_size * self.encoder.dimX)
        else:
            print("ERROR_VAE: VAE type unknown")
            raise
        kld = torch.mean(0.5 * torch.sum(torch.exp(self.z_logSigma) +
                                         self.z_mu**2 - 1. - self.z_logSigma, 1))
        loss = recon + self.beta * kld
        return loss

    def trainVAE(self, train_loader, epochNb):

        self.train()
        # check mb_size
        if train_loader.batch_size != self.mb_size:
            print("ERROR_VAE_train: batch sizes of data and vae mismatched")
            raise

        optimizer = optim.Adam(self.parameters, self.lr)

        if epochNb <= self.epoch_nb:
            print("ERROR_VAE_train: vae already trained to " +
                  str(self.epoch_nb) + " epochs")
            print("Try a bigger epochs number")
            raise
        for epoch in range(self.epoch_nb + 1, epochNb + 1):

            lossValue = 0

            for i, sample_batched in enumerate(train_loader):
                batch_length = sample_batched['image'].size(
                    1) * sample_batched['image'].size(0)

                # make sure the size of the batch corresponds to
                # mbSize*dataSize
                if (batch_length != self.mb_size * self.encoder.dimX):
                    print("ERROR: sizes of data and vae input mismatched")
                    print("batch_length = " + str(batch_length))
                    print("vae input length = " +
                          str(self.mb_size * self.encoder.dimX))
                    raise

                # convert 'double' tensor to 'float' tensor
                X = sample_batched['image'].view(
                    self.mb_size, self.encoder.dimX).float()
                X = Variable(X)
                self(X)
                # compute loss between data input and sampled data
                lossVariable = self.loss(X)

                lossVariable.backward()
                lossValue += lossVariable.data[0]

                optimizer.step()

                # Housekeeping
                for p in self.parameters:
                    if p.grad is not None:
                        data = p.grad.data
                        p.grad = Variable(data.new().resize_as_(data).zero_())

            print('====> Epoch: {} Average loss: {:.8f}'.format(
                  epoch, lossValue / len(train_loader.dataset)))
        self.trained = True
        self.epoch_nb = epochNb

    def save(self, datasetName, saveDir):
        # transform .npz to avoid dot in name (consider only .npz for now)
        name = datasetName.replace(".npz", "_NPZ")
        # add infos on vae structure
        encoderInfo = '_E'
        encoderInfo += '<'
        for numLayerE in range(self.encoder.nb_h):
            encoderInfo += str(self.encoder.inDim_h[numLayerE]) + '-'
            encoderInfo += self.NL_funcE[numLayerE] + '-'
        encoderInfo += str(self.encoder.outDim_h[numLayerE]
                           ) + '-' + 'muSig' + '-'
        encoderInfo += str(self.encoder.dimZ) + '>'
        name += encoderInfo

        decoderInfo = '_D'
        decoderInfo += '<'
        for numLayerD in range(self.decoder.nb_h):
            decoderInfo += str(self.decoder.inDim_h[numLayerD]) + '-'
            decoderInfo += self.NL_funcD[numLayerD] + '-'
        if not self.decoder.bernoulli and self.decoder.gaussian:
            decoderInfo += str(self.decoder.outDim_h[numLayerD]
                               ) + '-' + 'muSig' + '-'
        decoderInfo += str(self.decoder.dimX) + '>'
        name += decoderInfo

        betaInfo = '_beta' + str(self.beta)
        name += betaInfo

        # add infos on training state
        mbSizeInfo = '_mb' + str(self.mb_size)
        name += mbSizeInfo
        lrInfo = '_lr' + str(self.lr).replace(".", "dot")
        name += lrInfo
        epochInfo = '_ep' + str(self.epoch_nb)
        name += epochInfo
        # save it directory
        save_path = saveDir + name

        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        torch.save(self, save_path)
        print('Saved VAE state into ' + save_path)
        self.saved = True
        return name

    def getParams(self):
        listParams = []
        listParams.append(self.encoder.dimX)
        listParams.append(self.decoder.dimZ)
        listParams.append(self.IOh_dims_Enc)
        listParams.append(self.IOh_dims_Dec)
        listParams.append(self.NL_funcE)
        listParams.append(self.NL_funcD)
        listParams.append(self.mb_size)
        listParams.append(self.beta)
        listParams.append(self.lr)
        listParams.append(self.epoch_nb)
        listParams.append(self.decoder.bernoulli)
        listParams.append(self.decoder.gaussian)

        return listParams

    def generate(self, frameNb, saveDir):
        self.eval()
        # tensorParamValues = torch.FloatTensor(
        #     frameNb, self.decoder.dimZ).zero_()
        tensorParamValues = torch.FloatTensor(
            frameNb, 1).zero_()
        for i in range(frameNb):
            # ramp between 0 and 1 for all z dimensions (not enough...)
            tensorParamValues[i][:] = float(i * 20) / float(frameNb) - 10

        sample = Variable(tensorParamValues)
        self.decode(sample)
        if self.decoder.bernoulli and not self.decoder.gaussian:
            image = self.X_sample.cpu()
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            image = self.X_mu.cpu()
        save_image(image.data.view(frameNb, self.decoder.dimX),
                   saveDir + 'z0LinearRamp.png')


def loadVAE(vaeSaveName, load_dir):
    if not os.path.exists(load_dir):
        print("ERROR_VAE_Load: " + load_dir + " invalid directory")
        raise
    savefile_path = load_dir + vaeSaveName
    if not os.path.exists(savefile_path):
        print("ERROR_VAE_Load: " + vaeSaveName + " invalid file")
        raise
    vae = torch.load(savefile_path)

    # check if vae and vaeSameName match
    paramsFromFilename = getParamsFromName(vaeSaveName)
    paramsFromVAE = vae.getParams()
    if paramsFromFilename != paramsFromVAE:
        print("ERROR_LOAD: vae loaded and fileName mismatched")
        raise
    else:
        vae.loaded = True
        return vae

#------------------------------------------------------------------------------


def getParamsFromName(vaeSaveName):
    # e.g.vaeSaveName =
    # 'dummyDataset100_NPZ_E<1024-relu-401-muSig-6>_D<6-relu-399-sigmoid-1024>_beta4_mb10_lr0dot001_ep11'
    s_split = vaeSaveName.split("_")
    datasetName_s = s_split[0]
    datasetType_s = s_split[1]
    encoderNet_s = s_split[2]
    decoderNet_s = s_split[3]
    beta_s = s_split[4]
    mbSize_s = s_split[5]
    lr_s = s_split[6]
    epoch_s = s_split[7]

    # retrieve encoder net dimensions and
    IOh_dims_Enc = []
    NL_types_Enc = []
    # remove labels, only keep values
    encoderNet_s = encoderNet_s.replace(
        "E", "").replace("<", "").replace(">", "")
    encoderNet_tab = encoderNet_s.split("-")
    for i in range(len(encoderNet_tab)):
            # send heaven index val in IOh_dims_Enc
        if i % 2 == 0:
            IOh_dims_Enc.append(int(encoderNet_tab[i]))
        # send odd index val in NL_types_Enc
        else:
            NL_types_Enc.append(encoderNet_tab[i])
    NL_types_Enc.remove('muSig')

    # retrieve decoder net dimensions and
    IOh_dims_Dec = []
    NL_types_Dec = []
    # remove labels, only keep values
    decoderNet_s = decoderNet_s.replace(
        "D", "").replace("<", "").replace(">", "")
    decoderNet_tab = decoderNet_s.split("-")
    for i in range(len(decoderNet_tab)):
        # send heaven index val in IOh_dims_Dec
        if i % 2 == 0:
            IOh_dims_Dec.append(int(decoderNet_tab[i]))
        # send odd index val in NL_types_Dec
        else:
            NL_types_Dec.append(decoderNet_tab[i])
    # check if the decoder is gaussian or bernoulli
    if NL_types_Dec[-1] == 'muSig':
        NL_types_Dec.remove('muSig')
        gaussian = True
        bernoulli = False
    else:
        bernoulli = True
        gaussian = False

    X_dim = IOh_dims_Enc[0]
    Z_dim = IOh_dims_Dec[0]
    beta = int(beta_s.replace("beta", ""))
    mb_size = int(mbSize_s.replace("mb", ""))
    lr = float(lr_s.replace("lr", "").replace("dot", "."))
    epoch_nb = int(epoch_s.replace("ep", ""))

    listParams = []
    listParams.append(X_dim)
    listParams.append(Z_dim)
    listParams.append(IOh_dims_Enc)
    listParams.append(IOh_dims_Dec)
    listParams.append(NL_types_Enc)
    listParams.append(NL_types_Dec)
    listParams.append(mb_size)
    listParams.append(beta)
    listParams.append(lr)
    listParams.append(epoch_nb)
    listParams.append(bernoulli)
    listParams.append(gaussian)

    return listParams
