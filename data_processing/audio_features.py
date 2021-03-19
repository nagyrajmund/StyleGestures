import numpy as np
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import sys
import os
from os.path import join
import math

def extract_melspec(audio_dir, files, destpath, fps, audio_format = "'.wav"):
    for f in files:
        file = join(audio_dir, f + audio_format)
        outfile = join(destpath, f + '.npy')
        
        print('{}\t->\t{}'.format(file, outfile))
        fs, X = wav.read(file)

        X = X.astype(float)/math.pow(2,15)

        assert fs%fps == 0
        
        hop_len=int(fs/fps)
        
        n_fft=int(fs*0.13)
        C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
        C += 1e-10
        C = np.log(C)
        
        
        print("fs: " + str(fs))
        print("hop_len: " + str(hop_len))
        print("n_fft: " + str(n_fft))
        print(C.shape)
        print(np.min(C),np.max(C))
        np.save(outfile,np.transpose(C))
