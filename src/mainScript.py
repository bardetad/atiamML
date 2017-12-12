# main script to define and train VAE in a single python command
import sys
import argparse

sys.path.append('./')
sys.path.append('./dataset/')

import torch

from VAE import VAE
from ManageDataset import NPZ_Dataset

parser = argparse.ArgumentParser(description='generic VAE training and saving')


# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')

# VAE dimensions
parser.add_argument('--encoderIOdims', type=list, default=[
                    1024, 600, 6], metavar='[int list]', help='IO dimensions of encoder net (default: [1024, 600, 6])')

parser.add_argument('--decoderIOdims', type=list, default=[
                    6, 600, 1024], metavar='[int list]', help='IO dimensions of decoder net (default: [6, 600, 1024])')

# VAE Non linear functions
parser.add_argument('--encoderNL', type=list, default=[
                    'relu'], metavar='[string list]', help='encoder nonlinear activations (default: [''relu''])')
parser.add_argument('--decoderNL', type=list, default=[
                    'relu'], metavar='[string list]', help='decoder nonlinear activations (default: [''relu''])')

# VAE type
parser.add_argument('--type', type=str, default='gaussian', metavar='bernoulli/gaussian',
                    help='chose type of vae: either gaussian or bernoulli (default: ''gaussian'')')

# load Dataset and save VAE state settings
parser.add_argument('--NPZ-dataset', type=str, default='../data/dummyDataset100Gaussian.npz',
                    metavar='path', help='datasetName.npz file path (default: ''./data/dummyDataset100Gaussian.npz'')')
parser.add_argument('--dataKey', type=str, default='Spectrums',
                    metavar='key', help='key for data in .npz dataset (default: ''Spectrums'')')
parser.add_argument('--labelKey', type=str, default='labels',
                    metavar='key', help='key for labels in .npz dataset (default: ''labels'')')
parser.add_argument('--save-path', type=str, default='../data/save/',
                    metavar='path', help='VAE save path after training (default: ''./data/saveTest''). CAREFUL !! define a save path for each VAE')

# training settings
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

args = parser.parse_args()

# copy parser args into variables
mb_size = args.batch_size
epoch_nb = args.epochs

X_dim = args.encoderIOdims[0]
Z_dim = args.decoderIOdims[0]
IOh_dims_Enc = args.encoderIOdims
IOh_dims_Dec = args.decoderIOdims
NL_types_Enc = args.encoderNL
NL_types_Dec = args.decoderNL
if args.type == 'bernoulli':
    vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
              IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=True, gaussian=False)
elif args.type == 'gaussian':
    vae = VAE(X_dim, Z_dim, IOh_dims_Enc,
              IOh_dims_Dec, NL_types_Enc, NL_types_Dec, mb_size, bernoulli=False, gaussian=True)
else:
    print("ERROR script: Chose VAE type -> either bernoulli or gaussian")

# prepare dataset
datasetName = args.NPZ_dataset.split("/")[-1]
datasetDir = args.NPZ_dataset.replace(datasetName, "")
saveDir = args.save_path
testDataset = NPZ_Dataset(datasetName,
                          datasetDir, args.dataKey, args.labelKey)
train_loader = torch.utils.data.DataLoader(
    testDataset, batch_size=mb_size, shuffle=True)

# train it for 10 epochs
vae.train(train_loader, epoch_nb)

# save it
savefile = vae.save(datasetName, saveDir)



