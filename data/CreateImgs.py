# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:53:22 2017

Script to create .npz file containing all spectrums of dataset. It creates a dictionary
with a 2D numpy matrix and associated labels. One can choose to create spectra, spectra + phase or MDCT input.
Dataset 1 (additive) = 22000 items
Dataset 2 (FM) =  44800 items
Keys:
- ['images']: the 2D matrix with spectrums (ex. for dataset 1, spectra: 1025x22000; spectra + phase = 2050x22000, MDCT = 1024x22000)
- ['labels']: an array containing the labels associated to every spectrum (for dataset 1: 4x22000, for dataset 2: 5x44800)

@author: bavo
"""
import os
import numpy as np
import librosa
import MDCT

# Sound input parameters
folder_path = '/media/bavo/1E84369D843676FD/Documents and Settings/bavov/Documents/ATIAM/4_Informatique/MachineLearning_Project/Datasets/'
dataset = 'toy_dataset_2/'

# Spectrogram parameters
fft_size = 4096
hop_length = fft_size * 3 / 4

imgs_stack = np.zeros([2050, 1])
labels_stack = np.zeros([4, 20000])
i = 0

# Loop over sound files and create/store spectrogram
for filename in os.listdir(folder_path + dataset):
    if filename.endswith(".wav"):
        # Parse filename
        sound, _ = filename.split('.')
        label = sound.split('_')
        labels_stack[0, i] = int(label[1])
        labels_stack[1, i] = int(label[2])
        labels_stack[2, i] = int(label[3])
        labels_stack[3, i] = int(label[4])
#        labels_stack[4,i] = int(label[5])
        i = i + 1

        y, sr = librosa.load(folder_path + dataset + filename, sr=None)
        if dataset == 'Spectrumphase':
            # Spectrum and phase
            Spec = librosa.stft(y, fft_size)
            D = np.abs(Spec)
            Ang = np.angle(Spec)
    #        log_D = 20*np.log10(np.abs(D)**2)
            temp = np.append(Ang[0:1025, 5], D[0:1025, 5], axis=0)
        elif dataset == 'Spectrum':
            Spec = librosa.stft(y, fft_size)
            D = np.abs(Spec)
            temp = D
        elif dataset == 'MDCT':
            y_DCT = MDCT.mdct4(y[:2048])
            y_IDCT = MDCT.imdct4(y_DCT)
            temp = y_DCT

        # Store in numpy array
        imgs_stack = np.append(imgs_stack, temp[np.newaxis].T, axis=1)
        if i % 2000 == 0:
            print(i)

imgs_stack = np.delete(imgs_stack, 0, 1)
toy_dataset_dict = {'images': imgs_stack, 'labels': labels_stack}
np.savez(folder_path + 'dataset2SpecPhaseNormBVK.npz', **toy_dataset_dict)

# Example to load data
#test1 = np.load(folder_path + 'dataset3SpecphaseBVK.npz')
#imgs_stack = test1['images']
#labels_stack = test1['labels']
