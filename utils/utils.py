import librosa
import numpy as np
import pandas as pd
import random

def frequency_masking(mel_spectrogram, frequency_masking_para=13, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape
    '''
    Apply frequency masking on mel spectrogram
    :param mel_spectrogram: mel spectrogram
    :param frequency_masking_para: parameter for frequency masking
    :param frequency_masking_num: number of masks
    :param fbank_size: size of the filterbank
    :return mel_spectrogram: frequency masked spectrogram
    '''
    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram
   
def time_masking(mel_spectrogram, time_masking_para=40, time_mask_num=1):
    fbank_size = mel_spectrogram.shape
    '''
    Apply time masking on mel spectrogram
    :param mel_spectrogram: mel spectrogram
    :param time_masking_para: parameter for time masking
    :param time_masking_num: number of masks
    :param fbank_size: size of the filterbank
    :return mel_spectrogram: time masked spectrogram
    '''
    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram

def cmvn(data):
    '''
    Capstral mean and variance normalization
    :param data: data to normalize as numpy array
    :return data: normalized data
    '''
    shape = data.shape
    eps = 2**-30
    for i in range(shape[0]):
        utt = data[i].squeeze().T
        mean = np.mean(utt, axis=0)
        utt = utt - mean
        std = np.std(utt, axis=0)
        utt = utt / (std + eps)
        utt = utt.T
        data[i] = utt.reshape((utt.shape[0], utt.shape[1], 1))
    return data

def deltas(X_in):
    '''
    Calculate deltas of spectrogram
    :param X_in: input data (mel spectrogram)
    :param X_out: delta of input data
    '''
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out

def deltas_single(X_in):
    '''
    Calculate deltas of spectrogram
    :param X_in: input data (mel spectrogram)
    :param X_out: delta of input data
    '''
    X_out = (X_in[:,2:,:]-X_in[:,:-2,:])/10.0
    X_out = X_out[:,1:-1,:]+(X_in[:,4:,:]-X_in[:,:-4,:])/5.0
    return X_out


def pitch_shift(stereo, dist_param_lower=-4, dist_param_upper=4, sr=48000):
    '''
    Apply time shift on time series data
    :param stereo: input time series data of all channels 
    :param dist_param_upper: upper parameter for generating uniform distribution
    :param distribution_param_lower: lower parameter for generating uniform distribution
    :param sr: sampling rate
    :return stereo_shifted: time shift applied time series data 
    '''
    n_step = np.random.uniform(dist_param_lower, dist_param_upper)
    stereo_shifted = []
    for mono in stereo:
        y_pitched = librosa.effects.pitch_shift(np.array(mono), sr, n_steps=n_step)
        stereo_shifted.append(y_pitched.round())
    return stereo_shifted    

def time_shift(stereo, uniform_low=0.5, uniform_high=1.5, sr=48000):
    '''
    Apply time shift on time series data
    :param stereo: input time series data of all channel 
    :param unfiform_high: upper parameter for generating uniform distribution
    :param uniform_low: lower parameter for generating uniform distribution
    :param sr: sampling rate
    :return stereo_shifted: time shift applied time series data 
    '''
    time_factor = np.random.uniform(uniform_low, uniform_high)
    stereo_shifted = []
    for mono in stereo:
        length = len(mono)
        y_stretch = librosa.effects.time_stretch(np.array(mono), time_factor)
        if len(y_stretch) < length:
            y_stretch = np.concatenate((y_stretch, y_stretch))
            y_stretch = y_stretch[0:length]
        else:
            y_stretch = y_stretch[0:length]
        stereo_shifted.append(y_stretch.round())
    return stereo_shifted

def add_random_noise(stereo):
    '''
    Apply random noise to time series data
    :param stereo: input time series data of all channel 
    :return stereo: added noise to time series data 
    '''
    noise = np.random.normal(0,1,len(stereo[0]))
    stereo_augmented = []
    for mono in stereo:
        mono = np.asanyarray(mono)
        try:
            augmented_data = np.where(mono != 0.0, mono.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
        except:
            print('len mono: {}, len noise {}'.format(len(mono), len(noise)))
            pass
        stereo_augmented.append(augmented_data)
    return stereo

