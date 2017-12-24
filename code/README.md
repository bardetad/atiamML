# Code Documentation
`src/` contains all Modules to create, train, save, load, visualize **Variational autoencoder** (VAE) with the wanted dataset.

   > mainScript.py is the script used either to train a VAE with a `.npz` dataset (see README.md in `../toy`) or load an already trained VAE and do some things like reconstruction analysis, latent space visualization, generation...\
Creation, training, saving and loading are handled by VAE.py and EncoderDecoder.py.\
ManageDataset.py in `dataset/` makes the link between the training dataset and the VAE.
Once the VAE is trained, `Visualize/` folder gives tools and functions to visualize and analyse it.

`data/` is the default folder where trained VAEs are stored\
`unitTest/` contains helpful simple tests on VAE classes, datasets and analysis to make sure  everything is ok when changes are made in source code

## **Warnings**
##### Before each commit **run unit tests** in *`./unitTest/`* folder
##### Name of datasets **should NOT** contain '_' characters
##### **mb-size** needs to be a divider of total dataset length.

## **Use of mainScript.py**

### **Help**
```{r, engine='bash', count_lines}
python mainScript.py --help
```

### 1. Training
 To train a **Gaussian VAE** on *dataset.npz* file of  44800 data , data dim of 1024, z dim of 10, 1-'relu'-layer NN for Encoder and Decoder and save it in *../data/savedVAE/* folder, the total command should be:
```{r, engine='bash', count_lines}
python mainScript.py -mode "train" -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -type "gaussian" -dataset-path "../data/dataset.npz" -dataKey "images" -save-path "../data/savedVAE/" -mb-size 10 -epochs 10  
```
The command can be reduced as it has default values:

```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -dataset-path "../data/dataset.npz" -save-path "../data/savedVAE/" 
```
 For a **Bernoulli VAE**, the equivalent command will be : 

```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" "sigmoid" -type "bernoulli" -dataset-path "../data/dataset.npz" -save-path "../data/savedVAE/" 
```

An **immediate test** is to train a VAE  on a dummy dataset of filepath *../data/dummyDataset98.npz* . It's composed of 100 spectra of length 1024. 
The command:
```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -mb-size 49 -dataKey "Spectrums"
```
will by default load this dataset. At the end of the training, the VAE is saved into the default save path *../data/dummySave*
Bernoulli equivalent :
```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" "sigmoid" -type "bernoulli" -mb-size 49 -dataKey "Spectrums"
```
**More flags**
- Warm-up (in number of epochs)
```{r, engine='bash', count_lines}
-Nwu 100
```
- Noise input during training (gain value of a random gaussian noise)
```{r, engine='bash', count_lines}
-noise 1.5
```


### 2. Loading VAE, visualisation & sampling latent space
Instead of using the default mode **"train"** in command, use the mode **"load"** with flag:
```{r, engine='bash', count_lines}
-mode "load"
```
```{r, engine='bash', count_lines}
python mainScript.py -mode "load" -vae-path "../unitTest/dummySaveTest/dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5"
```
The above command load the VAE saved from the previous training on the dummy dataset. 
For now it only samples from the first dimension of latent space and create an image of spectra through the linear evolution of z[0].
**TODO** - add tools of visualization.

**NB** - the save name of the VAE after training is very heavy.
Example : *../unitTest/dummySaveTest/dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5*
But very useful as it contains all the info on VAE structure and state at the end of its training.

# **Unit Testing**
**TODO**

---








