import unittest
import sys
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


class TestVAEFunctions(unittest.TestCase):

    def test_VAE_lonelyForward(self):
        mb_size = 64
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X), volatile=True)
        X_dim = mnist.train.images.shape[1]
        Z_dim = 40
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

    def test_VAE_Learning(self):
        mb_size = 1
        X, _ = mnist.train.next_batch(mb_size)

        X = Variable(torch.from_numpy(X))
        X_dim = mnist.train.images.shape[1]
        Z_dim = 1
        IOh_dims_Enc = [X_dim, 50, Z_dim]
        IOh_dims_Dec = [Z_dim, 50, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        optimizer = optim.Adam(vae.parameters(), lr=1e-3)

        fig = plt.figure()
        ims = []

        for i in range(200):
            optimizer.zero_grad()
            out = vae(X)
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
        testDataset = NPZ_Dataset('dummyDataset_100.npz',
                                  './dummyDataset/', 'Spectrums', 'labels')
        train_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=mb_size, shuffle=True)

        X_dim = 1024
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 100, Z_dim]
        IOh_dims_Dec = [Z_dim, 100, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size)

        vae.train(train_loader, epoch_nb)
        self.assertTrue((vae.created == True) and (vae.trained == True))


suiteVAECreation = unittest.TestLoader().loadTestsFromTestCase(TestVAECreation)
print "\n\n------------------- VAE Creation Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteVAECreation)
suiteVAEFunctions = unittest.TestLoader().loadTestsFromTestCase(TestVAEFunctions)
print "\n\n------------------- VAE functions Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteVAEFunctions)
