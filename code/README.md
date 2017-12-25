# Code Documentation

## Files & folders
`src/` contains all Modules to create, train, save, load, visualize **Variational autoencoder** (VAE) with the wanted dataset.

   > * **mainScript.py** is the script used either to train a VAE with a `.npz` dataset (see **README.md** in `../toy`) or load an already trained VAE and do some things like reconstruction analysis, latent space visualization, generation...\
>* Creation, training, saving and loading are handled by **VAE.py** and **EncoderDecoder.py**.\
>* **ManageDataset.py** in `dataset/` makes the link between the training dataset and the VAE.
Once the VAE is trained, `Visualize/` folder gives tools and functions to visualize and analyse it.

`data/` is the default folder where trained VAEs are stored\

`unitTest/` contains helpful simple tests on VAE classes, datasets and analysis to make sure  everything is ok when source code is changed
>* **EncoderDecoderTest.py** tests Encoder/Decoder structure
>* **DatasetTest.py** tests dataset handling
>* **VAETest.py** tests VAE train, save, load, analyse, visualize

## Warnings
Before each commit **run unit tests** in `./unitTest/` folder

Name of datasets **should NOT** contain any '_' characters

**mb-size** needs to be a divider of total dataset length.

## Use of **mainScript.py**

### Help
```{r, engine='bash', count_lines}
python mainScript.py --help
```

### 1. Training

#### Immediate test
To train a VAE  on a dummy dataset of filepath `../data/dummyDataset98.npz`. It's composed of 100 spectra of length 1024. 
The command:
```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" -mb-size 49 -dataKey "Spectrums"
```
will by default load this dataset. At the end of the training, the VAE is saved into the default save path `../data/dummySave`
Bernoulli equivalent :
```{r, engine='bash', count_lines}
python mainScript.py -encoderIOdims 1024 600 10 -decoderIOdims 10 600 1024 -encoderNL "relu6" -decoderNL "relu6" "sigmoid" -type "bernoulli" -mb-size 49 -dataKey "Spectrums"
```

#### Example
To train a **Gaussian VAE** on "dataset.npz" dataset of 44800 data of dimension 1024, whose latent dimensions is 10, 1-'relu'-layer NN for Encoder and Decoder and save it in `../data/savedVAE/` folder, the total command should be:
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
**More flags**
- Warm-up (in number of epochs)
```{r, engine='bash', count_lines}
-Nwu 100
```
- Noise input during training (gain value of a random gaussian noise)
```{r, engine='bash', count_lines}
-noise 1.5
```


### 2. Loading VAE, visualization & sampling latent space

Instead of using the default mode **"train"** in command, use the mode **"load"** with flag. This mode enables to use trained VAE and do some stuffs with it (e.g. PCA, t-sne, generation ...).
```{r, engine='bash', count_lines}
-mode "load"
```

**Generation**\
Here for example it generates 200 images. Each image is 200x1024.
It corresponds to 200 stacked spectra (of size 1024). Hence, x axis corresponds to frequency while y axis is a latent dimension varying from -10 to 10. When changing image it varies another latent dimension also from -10 to 10.

![alt text](https://github.com/bardetad/atiamML/blob/master/code/data/images/generationExample/z0valOffset100_z1rangeCentered100.png?raw=true "Spectra")
> One of the generated images of a VAE trained for 250 epochs with a 10 latent space dimensions toy dataset (toy-spectral-richness.npz). 2nd latent space dimension is varying from -10 (top) to 10 (bottom) while others are fixed. 

**Visualization**\
Plots are not in **mainScript.py** but they are easy to add. For more details see class "TestVAEVisualize" in **VAETests.py** in `src/unitTest/`

![alt text](https://github.com/bardetad/atiamML/blob/master/code/data/images/PCAExample/PCA_beta1_WU100.png?raw=true "PCA_warmup")
> A PCA from the same VAE. It represents labeled inputs a latent space plane (the more expressive). Blue points correspond to low harmonics richness spectra while yellow correspond to high. 



```{r, engine='bash', count_lines}
python mainScript.py -mode "load" -vae-path "../unitTest/dummySaveTest/dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5"
```
The above command load the VAE saved from the previous training on the dummy dataset. 



**NB** - the save name of the VAE after training is very heavy: `../unitTest/dummySaveTest/dummyDataset98_NPZ_E<1024-relu6-600-muSig-10>_D<10-relu6-600-muSig-1024>_beta1_mb49_lr0dot001_ep5`.\
But very useful as it contains all the info on VAE structure and state at the end of its training.



### 3. Unit Testing
```{r, engine='bash', count_lines}
cd unitTest/
```

* Tests on dataset handler
```{r, engine='bash', count_lines}
python DatasetTest.py
```
* Tests on encoder/decoder structure
```{r, engine='bash', count_lines}
python EncoderDecoderTest.py
```
* Various tests on VAE (learning, saving, loading, visualize)
```{r, engine='bash', count_lines}
python VAETest.py
```









