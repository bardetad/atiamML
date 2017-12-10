import unittest
import sys
import os
import shutil
# Add the src folder path to the sys.path list
sys.path.append('../src')
sys.path.append('../src/dataset')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import misc
import numpy
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from VAE import VAE
from ManageDataset import NPZ_Dataset


mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)


class TestVAECreation(unittest.TestCase):

    def test_good_VAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertTrue(model.created)

    def test_wrong_EncoderStructure(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_DecoderStructure(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_EncoderNLFunctionsNb(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu', 'relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_DecoderNLFunctionsNb(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_EncoderNLfunctionsSyntax(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['reLu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_DecoderNLfunctionsSyntax(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoide']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_good_gaussianVAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec, bernoulli=False, gaussian=True)
        self.assertTrue(model.created)

    def test_wrong_gaussianVAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc,
                    IOh_dims_Dec, NL_types_Enc, NL_types_Dec, bernoulli=False, gaussian=True)
        self.assertFalse(model.created)


class TestVAEFunctions(unittest.TestCase):

    def test_VAE_lonelyForward(self):
        mb_size = 64
        # test on mnist dataset
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X), volatile=True)

        # define vae structure
        X_dim = mnist.train.images.shape[1]
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 400, Z_dim]
        IOh_dims_Dec = [Z_dim, 400, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        optimizer.zero_grad()
        out = None
        out = vae(X)
        vae.encoder.getInfo()
        vae.decoder.getInfo()
        self.assertTrue((vae.created == True) and (
            out.size()[1] == X_dim and out.size()[0] == mb_size))

    def test_gaussianVAE_Learning(self):
        mb_size = 1
        # test on mnist dataset
        X, _ = mnist.train.next_batch(mb_size)

        # define vae structure
        X = Variable(torch.from_numpy(X))
        X_dim = mnist.train.images.shape[1]
        Z_dim = 1
        IOh_dims_Enc = [X_dim, 50, Z_dim]
        IOh_dims_Dec = [Z_dim, 50, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)

        optimizer = optim.Adam(vae.parameters(), lr=1e-3)

        fig = plt.figure()
        ims = []

        for i in range(200):
            optimizer.zero_grad()
            if vae.decoder.gaussian:
                vae(X)
                out = vae.X_mu
            else:
                raise
            loss = vae.loss(X)
            if i == 0:
                initialLoss = loss.data[0]
            print("Loss -> " + str(loss.data[0]))
            loss.backward()
            optimizer.step()

            # update plot
            gen = out.data.numpy()
            gen_2D = numpy.reshape(gen[0], (28, 28)) * 255
            im = plt.imshow(gen_2D, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)

        plt.show()

        self.assertTrue((vae.created == True) and (loss.data[0] < initialLoss))

    def test_VAE_trainLoop(self):
        mb_size = 10
        epoch_nb = 10

        # define dataset
        datasetName = 'dummyDataset100Bernoulli.npz'
        datasetDir = './dummyDataset/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums', 'labels')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)

        # define vae structure
        X_dim = 1024
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 400, Z_dim]
        IOh_dims_Dec = [Z_dim, 400, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        vae.train(train_loader, epoch_nb)
        self.assertTrue((vae.created == True) and (vae.trained == True))

    def test_VAE_saveState(self):
        mb_size = 10
        epoch_nb = 11
        datasetName = 'dummyDataset100Bernoulli.npz'
        datasetDir = './dummyDataset/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums', 'labels')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)

        # define vae structure
        X_dim = 1024
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 401, Z_dim]
        IOh_dims_Dec = [Z_dim, 399, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        vae.train(train_loader, epoch_nb)
        # save it
        if vae.trained == True:
            vae.save(datasetName, datasetDir)

        self.assertTrue((vae.created == True) and (
            vae.trained == True) and (vae.saved == True))

    def test_VAE_loadState(self):
        # try to retrieve all infos on vae from name file

        mb_size = 10
        epoch_nb = 11
        datasetName = 'dummyDataset100Bernoulli.npz'
        datasetDir = './dummyDataset/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums', 'labels')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)

        # create random vae structure
        X_dim = 10
        Z_dim = 64
        IOh_dims_Enc = [X_dim, 15, Z_dim]
        IOh_dims_Dec = [Z_dim, 18, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        # now try to load another vae
        vae.load('dummyDataset100_NPZ_Encoder<1024-relu-401-mulogSigma-6>_Decoder<6-relu-399-sigmoid-1024>_mbSize10_lr0dot001_epoch11', datasetDir)

        if vae.loaded == True:
            print(vae)

        self.assertTrue((vae.created == True) and (vae.loaded == True))

    def test_gaussianVAE_trainsaveload(self):
        mb_size = 100
        epoch_nb = 5
        # if exists remove 'saveloadTest' folder
        if os.path.exists('./saveloadTest'):
            shutil.rmtree('./saveloadTest')
        # create a VAE
        X_dim = 1024
        Z_dim = 10
        IOh_dims_Enc = [X_dim, 200, Z_dim]
        IOh_dims_Dec = [Z_dim, 200, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)
        # prepare dataset
        datasetName = 'dummyDataset100Gaussian.npz'
        datasetDir = './dummyDataset/'
        saveDir = './saveloadTest/'
        testDataset = NPZ_Dataset(datasetName,
                                  datasetDir, 'Spectrums', 'labels')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)
        # train it for 10 epochs
        vae.train(train_loader, epoch_nb)
        # save it
        savefile = vae.save(datasetName, saveDir)
        # reload the savefile of VAE
        vae.load(savefile, saveDir)
        # continue training
        print(vae)
        vae.train(train_loader, 20)

    # def test_VAE_invalidLoad(self):

    # def test_VAE_invalidEpochNb(self):

    # def test_VAE_loadWrongFile(self):


suiteVAECreation = unittest.TestLoader().loadTestsFromTestCase(TestVAECreation)
print "\n\n------------------- VAE Creation Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteVAECreation)
suiteVAEFunctions = unittest.TestLoader().loadTestsFromTestCase(TestVAEFunctions)
print "\n\n------------------- VAE functions Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteVAEFunctions)
