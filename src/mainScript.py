# main script to define and train VAE in a single python command
import sys
import argparse

sys.path.append('./')
sys.path.append('./dataset/')

import torch

from VAE import VAE
from VAE import loadVAE
from ManageDataset import NPZ_Dataset

# SCRIPT MODES:
# if "train": train a vae from scratch.
# if "load": load (from savefile) a vae already trained and visualize outputs

parser = argparse.ArgumentParser(description='generic VAE training and saving')


parser.add_argument('-mode', type=str, default='train', metavar='boolean',
                    help='mode among "train", "load". (default: "train"):\
                    \n "train": train a vae from scratch.\
                    \n "load": load (from savefile) a vae already trained and visualize outputs')
parser.add_argument('-vae-path', type=str,
                    default='../data/dummySave/\
                    dummyDataset100Gaussian_NPZ_E<1024-relu6-600-muSig-1>\
                    _D<1-relu6-600-muSig-1024>_beta4_mb50_lr0dot001_e10')

# VAE dimensions
parser.add_argument('-encoderIOdims', nargs='+', type=int, metavar='int list',
                    help='<Required> IO dimensions of encoder net (example: 1024 600 6)')
parser.add_argument('-decoderIOdims', nargs='+', type=int, metavar='int list',
                    help='<Required> IO dimensions of decoder net (example: 6 600 1024)')

# VAE Non linear functions
parser.add_argument('-encoderNL', nargs='+', type=str, metavar='string list',
                    help='<Required> encoder nonlinear activations \
                    (example: "relu6" for 1 layer or "relu6" "sigmoid" for 2 layers)')
parser.add_argument('-decoderNL', nargs='+', type=str, metavar='string list',
                    help='<Required> decoder nonlinear activations \
                    (example: "relu6" for 1 layer or "relu6" "sigmoid" for 2 layers)')

# VAE type
parser.add_argument('-type', type=str, default='gaussian', metavar='bernoulli/gaussian',
                    help='chose type of vae: either gaussian or bernoulli (default: "gaussian")')

# load Dataset and save VAE state settings
parser.add_argument('-dataset-path', type=str, default='../data/dummyDataset100.npz',
                    metavar='path', help='datasetName.npz file path \
                    (default: "../data/dummyDataset100.npz")')
parser.add_argument('-dataKey', type=str, default='images',
                    metavar='key', help='key for data in .npz dataset (default: "images")')
parser.add_argument('-save-path', type=str, default='../data/dummySave/',
                    metavar='path', help='VAE save path after training (default: "../data/dummySave").')

# training settings
parser.add_argument('-mb-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('-epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-beta', type=int, default=1, metavar='N',
                    help='beta coefficient for regularization (default: 1)')
parser.add_argument('-Nwu', type=int, default=1, metavar='N',
                    help='epochs number for warm-up (default: 1 -> no warm-up)')


args = parser.parse_args()
mode = args.mode

# copy parser args into variables
mb_size = args.mb_size
epoch_nb = args.epochs
beta = args.beta
Nwu = args.Nwu

# prepare dataset
datasetName = args.dataset_path.split("/")[-1]
datasetDir = args.dataset_path.replace(datasetName, "")
saveDir = args.save_path
testDataset = NPZ_Dataset(datasetName,
                          datasetDir, args.dataKey)

if mode == "train":

    train_loader = torch.utils.data.DataLoader(
        testDataset, batch_size=mb_size, shuffle=True)

    X_dim = args.encoderIOdims[0]
    Z_dim = args.decoderIOdims[0]
    IOh_dims_Enc = args.encoderIOdims
    IOh_dims_Dec = args.decoderIOdims
    NL_types_Enc = args.encoderNL
    NL_types_Dec = args.decoderNL
    if args.type == 'bernoulli':
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=True, gaussian=False, beta=beta, Nwu=Nwu)
    elif args.type == 'gaussian':
        vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
                  IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True, beta=beta, Nwu=Nwu)
    else:
        print("ERROR script: Chose VAE type -> either bernoulli or gaussian")

    # train it for N epochs
    vae.trainVAE(train_loader, epoch_nb)

    # save it
    vae.save(datasetName, saveDir)

    vae.generate(10000, saveDir)

elif mode == "load":

    # get savefile path
    directory = args.vae_path
    savefile = directory.split("/")[-1]
    directory = directory.replace(savefile, "")
    # load vae
    vaeLoaded = loadVAE(savefile, directory)

    # retrain
    # train_loader = torch.utils.data.DataLoader(
    # testDataset, batch_size=vae.mb_size, shuffle=True)
    # vae.trainVAE(train_loader, epoch_nb)
    # vae.save(datasetName, saveDir)

    vaeLoaded.generate(1000, directory, 50)
