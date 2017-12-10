# VAE = { X -> Encoder-> Z -> Decoder ~> X }
# X_dim (int): dimension of X (input) - e.g. 513
# Z_dim (int): dimension of Z (latent) - e.g. 6
# IOh_dims_Enc (int*): hidden layers' IO encoder dims - e.g. [513 128 6]
# IOh_dims_Dec (int*): hidden layers' IO decoder dims - e.g. [6 128 513]
# NL_types_*** (string*): types on nonLinearity for each layer of Encoder/Decoder
#  e.g. NL_types_Enc = ['relu'] ; NL_types_Dec = ['relu', 'sigmoid']

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from EncoderDecoder import Encoder, Decoder


class VAE(nn.Module):

    def __init__(self, X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size=64, beta=1, lr=1e-3, bernoulli = True, gaussian = False):

        # superclass init
        super(VAE, self).__init__()
        self.created = False

        self.encoder = Encoder(X_dim, IOh_dims_Enc, Z_dim)
        self.decoder = Decoder(Z_dim, IOh_dims_Dec, X_dim, bernoulli, gaussian)
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

        if self.decoder.bernoulli and not self.decoder.gaussian:
            self.X_sample = var_h
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            # get X_sample's mu and logSigma
            self.X_mu = self.decoder.h_mu(var_h)
            self.X_logSigma = self.decoder.h_logSigma(var_h)
        else:
            print("ERROR VAE: wrong decoder type")
            raise

    def reparametrize(self):
        # eps = Variable(torch.randn(self.mb_size, self.encoder.dimZ))
        # return self.z_mu + torch.exp(self.z_logSigma / 2) * eps
        std = self.z_logSigma.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.z_mu)

    def loss(self, X):
        if self.decoder.bernoulli and not self.decoder.gaussian:
            # Bernoulli
            # recon = F.binary_cross_entropy(
            #     self.X_sample, X.view(-1, self.encoder.dimX))
            recon = F.binary_cross_entropy(self.X_sample, X, size_average=False)
        elif self.decoder.gaussian and not self.decoder.bernoulli:
            # gaussian
            recon = F.mse_loss(self.X_mu, X)/(self.mb_size * self.encoder.dimX)
        else:
            print("ERROR_VAE: VAE type unknown")
            raise
        # kld = - 0.5 * torch.sum(1 + self.z_logSigma -
                                # self.z_mu.pow(2) - self.z_logSigma.exp())
        kld = 0.5 * torch.sum(torch.exp(self.z_logSigma) + self.z_mu**2 - 1. - self.z_logSigma)
        # Normalise by same number of elements as in reconstruction
        kld /= self.mb_size * self.encoder.dimX
        loss = recon + self.beta * kld
        # loss /= (self.mb_size * self.encoder.dimX)
        return loss

    def train(self, train_loader, epochNb):

        # check mb_size
        if train_loader.batch_size != self.mb_size:
            print("ERROR_VAE_train: batch sizes of data and vae mismatched")
            raise

        optimizer = optim.Adam(self.parameters(), self.lr)

        if epochNb < self.epoch_nb:
            print("ERROR_VAE_train: vae already trained to" +
                  str(self.epoch_nb) + "epochs")
            print("Try a bigger epochs number")
            raise
        for epoch in range(self.epoch_nb + 1, epochNb + 1):

            # self.train()
            lossValue = 0

            for i, sample_batched in enumerate(train_loader):
                batch_length = sample_batched['image'].size(
                    1) * sample_batched['image'].size(0)

                # make sure the size of the batch corresponds to
                # mbSize*dataSize
                if (batch_length != self.mb_size * self.encoder.dimX):
                    print("ERROR: sizes of data and vae input mismatched")
                    raise

                # convert 'double' tensor to 'float' tensor
                X = sample_batched['image'].view(
                    self.mb_size, self.encoder.dimX).float()
                X = Variable(X)
                optimizer.zero_grad()
                self(X)
                # compute loss between data input and sampled data
                lossVariable = self.loss(X)
                lossVariable.backward()
                lossValue += lossVariable.data[0]
                optimizer.step()

                # time.sleep(0.1)

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, lossValue / len(train_loader.dataset)))
        if (epoch == epochNb):
            self.trained = True
            self.epoch_nb = epoch
        else:
            print("ERROR_VAE_train: wrong loop...")
            raise

    def save(self, name, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # transform .npz to avoid dot in name (consider only .npz for now)
        name = name.replace(".npz", "_NPZ")
        # add infos on vae structure
        encoderInfo = '_Encoder'
        encoderInfo += '<'
        for numLayerE in range(self.encoder.nb_h):
            encoderInfo += str(self.encoder.inDim_h[numLayerE]) + '-'
            encoderInfo += self.NL_funcE[numLayerE] + '-'
        encoderInfo += str(self.encoder.outDim_h[numLayerE]
                           ) + '-' + 'mulogSigma' + '-'
        encoderInfo += str(self.encoder.dimZ) + '>'
        name += encoderInfo

        decoderInfo = '_Decoder'
        decoderInfo += '<'
        for numLayerD in range(self.decoder.nb_h):
            decoderInfo += str(self.decoder.inDim_h[numLayerD]) + '-'
            decoderInfo += self.NL_funcD[numLayerD] + '-'
        if not self.decoder.bernoulli and self.decoder.gaussian:
            decoderInfo += str(self.decoder.outDim_h[numLayerD]
                               ) + '-' + 'mulogSigma' + '-'
        decoderInfo += str(self.decoder.dimX) + '>'
        name += decoderInfo

        # add infos on training state
        mbSizeInfo = '_mbSize' + str(self.mb_size)
        name += mbSizeInfo
        lrInfo = '_lr' + str(self.lr).replace(".", "dot")
        name += lrInfo
        epochInfo = '_epoch' + str(self.epoch_nb)
        name += epochInfo
        save_path = save_dir + name
        torch.save(self.state_dict(), save_path)
        print('Saved VAE state into ' + save_path)
        self.saved = True
        return name

    def load(self, vaeSaveName, load_dir):
        if not os.path.exists(load_dir):
            print("ERROR_VAE_Load: " + load_dir + " invalid directory")
            raise
        load_path = load_dir + vaeSaveName
        if not os.path.exists(load_path):
            print("ERROR_VAE_Load: " + vaeSaveName + " invalid file")
            raise

        # retrieve VAE structure
        X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, self.NL_funcE, self.NL_funcD, self.mb_size, self.beta, self.lr, self.epoch_nb, bernoulli, gaussian = self.retrievParamsFromName(
            vaeSaveName)
        self.encoder = Encoder(X_dim, IOh_dims_Enc, Z_dim)
        self.decoder = Decoder(Z_dim, IOh_dims_Dec, X_dim, bernoulli, gaussian)

        print(self.retrievParamsFromName(vaeSaveName))
        # self(Xdim, Zdim, IOhdimsEnc, IOhdimsDec, NLtypesEnc, NLtypesDec, mbsize, beta, lr)

        self.load_state_dict(torch.load(load_path))
        # self.eval()
        print('Loaded VAE state from ' + load_path)
        self.loaded = True
        time.sleep(2)

    def retrievParamsFromName(self, vaeSaveName):
        # e.g.vaeSaveName =
        # 'dummyDataset100_NPZ_Encoder<1024-relu-401-mulogSigma-6>_Decoder<6-relu-399-sigmoid-1024>_mbSize10_lr0dot001_epoch11'
        s_split = vaeSaveName.split("_")
        datasetName_s = s_split[0]
        datasetType_s = s_split[1]
        encoderNet_s = s_split[2]
        decoderNet_s = s_split[3]
        mbSize_s = s_split[4]
        lr_s = s_split[5]
        epoch_s = s_split[6]

        # retrieve encoder net dimensions and
        IOh_dims_Enc = []
        NL_types_Enc = []
        # remove labels, only keep values
        encoderNet_s = encoderNet_s.replace(
            "Encoder", "").replace("<", "").replace(">", "")
        encoderNet_tab = encoderNet_s.split("-")
        for i in range(len(encoderNet_tab)):
            # send heaven index val in IOh_dims_Enc
            if i % 2 == 0:
                IOh_dims_Enc.append(int(encoderNet_tab[i]))
            # send odd index val in NL_types_Enc
            else:
                NL_types_Enc.append(encoderNet_tab[i])
        NL_types_Enc.remove('mulogSigma')

        # retrieve decoder net dimensions and
        IOh_dims_Dec = []
        NL_types_Dec = []
        # remove labels, only keep values
        decoderNet_s = decoderNet_s.replace(
            "Decoder", "").replace("<", "").replace(">", "")
        decoderNet_tab = decoderNet_s.split("-")
        for i in range(len(decoderNet_tab)):
            # send heaven index val in IOh_dims_Dec
            if i % 2 == 0:
                IOh_dims_Dec.append(int(decoderNet_tab[i]))
            # send odd index val in NL_types_Dec
            else:
                NL_types_Dec.append(decoderNet_tab[i])
        # check if the decoder is gaussian or bernoulli
        if NL_types_Dec[-1] == 'mulogSigma':
            NL_types_Dec.remove('mulogSigma')
            gaussian = True
            bernoulli = False
        else:
            bernoulli = True
            gaussian = False

        X_dim = IOh_dims_Enc[0]
        Z_dim = IOh_dims_Dec[0]
        mb_size = int(mbSize_s.replace("mbSize", ""))
        beta = 1
        lr = float(lr_s.replace("lr", "").replace("dot", "."))
        epoch_nb = int(epoch_s.replace("epoch", ""))

        return X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, beta, lr, epoch_nb, bernoulli, gaussian
