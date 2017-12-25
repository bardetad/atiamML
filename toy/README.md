# Datasets and associated scripts:

Complete **datasets and trained VAEs** are **[here]**.

## 1-factor dataset
ToyRichnessGenerator.py is the script generating the simple 1-factor of variation dataset.	

**The data represents harmonical richness.**
The value of richness increases smoothly with the number of harmonics. The spectral slope is fixed to 1/n with n the partial number.
The first signal is a pure sinwave at 110 Hz (can be changed with the fond_f variable), the second signal is the first one added to a pure sinwave at 2*fond_f Hz and a small amplitude. Then the second harmonic's ampltidue increases until it reaches 1/n (1/2). And the same process is repeated for the third partial, and all others until the maximum frequency is reached (fmax in the code). A desired number of vectors is chosen (dataset_size = 10000 in our case) and the number of gains per harmonic is calculated so that the number of vectors adds up to dataset_size

**Magnitude** spectra are then computed on these temporal signals.
**Phases** can also be concatenated with the PHASE boolean.
**Wav audio** files can also be written using the WRITE_WAV boolean, thanks to the soundfile python library.

**OUTPUTS :**
* ['Spectrums'] : 1024 x N values containing N spectra
or 2048 x N values containing the associated phases
1024 : number of frequency bins
N : dataset size
* ['Labels'] :    2 x N Parameter values associated to the spectra and phases   
[1,:] : Number of harmonics
[2,:] : Gain of the last harmonic (compared to 1/n)

The output is an .npz file which contains a numpy dictionary with two np arrays : ['Spectrums'] and ['labels']. 
The labels are used for visualisation purposes and not for training as our model is an unsupervised VAE.

## 4- and 5-factor dataset
Two scripts are included to create the 4- and 5- factor dataset and should be run subsequently. One should choose the appropriate dataset and associated representation hardcoded IN the scripts:

* **CreateSnds.py**    : Script to create original .wav files using the [pyo](http://ajaxsoundstudio.com/software/pyo/) library
* **CreateImgs.py**    : Script to perform transforms and create VAE input data (spectra, spectra + phase, MDCT), uses the **MDCT.py** file

Datasets are created using a fixed pitch:

**DATASET 1 (f0 = 110) Additive synthesis with BiQuad filter (22000 items)**:
1. Harmonicity = sum(s3.bn/in((2n+j)*f*i), j = [0,1,0.1]
2. Nb. of Partials k
3. Filter (Q)
4. Filter (f)

**DATASET 2 (f0 = 1500) FM-synthesisa + Moog filter (48000 items)**:
1. Carrier frequency j (3 carriers)
2. Ratio k 
3. Index r
4. Filter resonance Q
5. Filter frequency f

Output is a .npz file containing all spectrums of dataset. It contains a dictionary
with a 2D numpy matrix and associated labels.
Keys:
* ['images']: the 2D matrix with spectrums (ex. for dataset 1, spectra: 1025x22000; spectra + phase = 2050x22000, MDCT = 1024x22000)
* ['labels']: an array containing the labels associated to every spectrum (for dataset 1: 4x22000, for dataset 2: 5x44800)


[here]: https://drive.google.com/drive/folders/1Yg91uB7FRAVPM_WLwVRjBLCPesdP3K2n
