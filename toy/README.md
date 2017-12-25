# Datasets and associated scripts:

Complete **datasets and trained VAEs** are **[here]**.

## 1-factor dataset
**ToyRichnessGenerator.py** is the script generating the simple 1-factor of variation dataset.
Precise description about the way how to use it and the inner parameters is given inside the script in comments.

The output of both the toy dataset scripts is an .npz file which is a numpy dictionary containing 
two variables. The data and the labels. Python classes are provide to load and save npz files.

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
