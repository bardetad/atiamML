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
import numpy as np
from scipy import signal
import librosa


# Sound input parameters
folder_path = "/home/bavo/Documents/ATIAM/4_Informatique/MachineLearning_Project/2_VAE_dataset/training_2/"
#sound = 'test_9.wav'

# Spectrogram parameters
fft_size = 2048
hop_length = fft_size * 3/4
window_hann = signal.hann(fft_size)

imgs_stack = np.zeros([1025,1])
labels_stack = np.chararray([188889,5])
i = 0

# Loop over sound files and create/store spectrogram
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        # Parse filename
        sound , _ = filename.split('.')
        label = sound.split('_')
        labels_stack[i,0] = str(label[1])
        labels_stack[i,1] = str(label[2])
        labels_stack[i,2] = str(label[3])
        labels_stack[i,3] = str(label[4])
        labels_stack[i,4] = str(label[5])
        i = i + 1
        
        # Compute spectrogram
        y, sr = librosa.load(folder_path + filename, sr=None)
        D = np.fft.fft(y,fft_size)[0:fft_size/2+1]
        log_D = 20*np.log10(np.abs(D)**2)
#        D = librosa.stft(y, fft, hop_length, window = 'hanning', center=True)
#        log_D = librosa.power_to_db(D, ref=np.max)
        
        # Store in numpy array
#        imgs_stack = np.dstack((imgs_stack, log_D[:,0][np.newaxis].T))
        imgs_stack = np.dstack((imgs_stack, log_D[:][np.newaxis].T))
        if i%1000 == 0:
            print(i)

imgs_stack = np.delete(imgs_stack,0,2)
toy_dataset_dict = {'images': imgs_stack, 'labels': labels_stack}
np.savez(folder_path + 'toy_dataset_1.npz', **toy_dataset_dict)

# Example to load data
#test1 = np.load(folder_path + 'toy_dataset_1.npz')
#test2 = test1['images']
 
        
