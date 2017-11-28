# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:53:22 2017

Script to create .npz file containing all spectrograms of toydataset. It creates a dictionary
with a 3D numpy matrix (513x12xlength(dataset)) and associated labels. (TODO: label-values)
Keys:
- ['images']: the 3D matrix with spectrograms
- ['labels']: an array containing the labels associated to every spectrogram

@author: bavo
"""
import os
import librosa
from scipy import signal
import numpy as np

# Sound input parameters
folder_path = "/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/training/"
#sound = 'test_9.wav'

# Spectrogram parameters
fft = 1024
hop_length = fft * 3/4
window_hann = signal.hann(fft)

imgs_stack = np.zeros([513,12])
labels_stack = np.zeros([20,1])
i = - 1
# Loop over sound files and create/store spectrogram
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        i = i + 1
        # Parse filename
        sound , _ = filename.split('.')
        _ , label = sound.split('_')
        labels_stack[i] = label
        
        # Compute spectrogram
        y, sr = librosa.load(folder_path + filename, sr=None)
        D = librosa.stft(y, fft, hop_length, window = 'hanning', center=True)
        log_D = librosa.power_to_db(D, ref=np.max)
        
        # Store in numpy array
        imgs_stack = np.dstack((imgs_stack, log_D))

imgs_stack = np.delete(imgs_stack,0,2)
toy_dataset_dict = {'images': imgs_stack, 'labels': labels_stack}
np.savez(folder_path + 'toy_dataset_1.npz', **toy_dataset_dict)
# Example to load data
#test1 = np.load(folder_path + 'toy_dataset_1.npz')
#test2 = test1['images']
 
        
