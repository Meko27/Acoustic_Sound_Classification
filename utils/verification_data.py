'''
Functions to preprocess data by removing unapprorpiate samples and provide same signal length.
'''
import glob
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import os
import shutil
import sys
from tqdm import trange

from config import *
from utils.dataio import *
from utils.utils import *

def _signal_to_noise(signal):
    '''
    Calculate signal to noise ration.
    :param signal: input signal
    :return: signal to noise ratio
    '''
    singal = np.asanyarray(signal)
    m = signal.mean(axis=0)
    std = signal.std(axis=0, ddof=0)
    return np.where(std == 0, 0, m/std)

def _signal_range(signal):
    '''
    Calculate signal range (min to max value)
    :param signal: input signal
    :return: range
    '''
    return max(signal) - min(signal)

def _cut_signal(signal_stereo, target_len):
    '''
    Cut signal from front to obtain same signal length for each mono signal.
    :param signal_stereo: stereo signal
    :param target_len: target signal length
    :return signal_stereo_cut: signal stereo with same signal length
    '''
    signal_stereo_cut = []
    for mono in signal_stereo:
        signal_stereo_cut.append(mono[len(mono)-target_len:])
    return signal_stereo_cut

def _cut_data(data, target_len):
    '''
    Cut signal from front to obtain same signal length for each mono signal.
    :param data: data as dict read directly from json
    :param target_len: target signal length
    :return signal_stereo_cut: signal stereo with same signal length
    '''
    signal_stereo_cut = []
    for node in data['sensor_data'].keys():
        n = len(data['sensor_data'][node])
        if n < target_len:
            print('--------------------------------')
            print('Length too small: ', n)
            print('Data ID: ', data['id'])
            print('--------------------------------')
        data['sensor_data'][node] = data['sensor_data'][node][n-target_len:]
    return data

def check_signal_length(signal_stereo, min_len):
    '''
    Filter out too short signals.
    :param signal_stereo: stereo signal containing 20 channels
    :param min_len: minimnum length to be considered as ok
    :return bool which decides wether signal should be sorted out
    '''
    for mono in signal_stereo:
        if len(mono) < min_len:
            return True
        else:
            continue
    return False

def check_missing_channels(signal_stereo):
    '''
    Filter out signals with missing channels.
    :param signal_stereo: stereo signal containing 20 channels
    :return bool which decides wether signal should be sorted out
    '''
    for mono in signal_stereo:
        if mono.size == 0:
            return True
        else:
            continue
    return False

def check_noise(signal_stereo, thres):
    '''
    Filter out noise.
    :param signal_stereo: stereo signal containing 20 channels
    :param thres: minimnum range to be considered as ok
    :return bool which decides wether signal should be sorted out
    '''
    for signal in signal_stereo:
        range = _signal_range(signal)
        if range > thres:
            return False
        else:
            continue    
    return True

def check_peaks(signal_stereo, thres=10):
    '''
    Filter out too short signals.
    :param signal_stereo: stereo signal containing 20 channels
    :param min_len: minimnum length to be considered as ok
    :return bool which decides wether signal should be sorted out
    '''
    for signal in signal_stereo:
        if max(signal) > thres*abs(min(signal)):
            return True
        else: 
            continue
    return False

def equalize_length(signal_stereo, target_length):
    '''
    Equalize length of mono signals to same targe_length.
    :param signal_stereo: stereo signal
    :param target_length: target length of signal
    :return signal_stereo_out: processed signal
    '''
    
def test_data_verification_functions():
    '''
    Test verificaton functions with exemples
    '''
    data_config = DataConfig('config/DataConfig.json')
    signal_empty = data_config.source_dir_local + '1615113905909_dcac8638-4e48-440f-b379-aba08b6ccca6.json'
    signal_noisy = data_config.source_dir_local + '1615112626381_1a54c228-22ba-42ab-911e-0982f0a0e351.json'
    signal_peaks = data_config.source_dir_local + '1615111246815_4ecd3d80-7b44-45a8-92d9-7eaafe7f1e29.json'
    singal_normal0 = data_config.source_dir_local + '1615111831849_d077d81d-59ff-43ee-b3b7-faca20e63119.json'
    singal_normal1 = data_config.source_dir_local + '1615112643589_69b86d9a-3e37-4e84-90b4-15fa56d6e6e6.json'
    addrs = [signal_empty, signal_noisy, signal_peaks, singal_normal0, singal_normal1]

    for addr in addrs:
        signal, label = load_raw_from_json(addr)
        res = check_missing_channels(signal)
        print('-----------------')
        plot_sequence(signal, label, show=True)

def plot_time_series_in_dir(path):
    '''
    Plots time series of data in given folder.
    :param path: data path with raw data
    '''
    addrs = glob.glob(path + '*.json')
    path_out = os.path.join(path, 'plots')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    for i in trange(len(addrs), desc='Plotting data', leave=True):
        signal, label = load_raw_from_json(addrs[i])
        plot_sequence(signal, label, fname=os.path.join(path_out, ntpath.basename(addrs[i])[:-5]))

def preprocess_verified_data_single(path, path_out, signal_length=6000):
    '''
    Plots time series of data in given folder.
    :param path: data path with raw data
    '''
    with open(path) as json_file:
        data = json.load(json_file) # json file as dict
    data_cut = _cut_data(data, target_len=signal_length)
    out_filename = str(data['time']) + '_' + str((data['surfaceLocation']['equatorial'],data['surfaceLocation']['height'])) + '_' + data['classification']['name'] + '.json'
    out_filename = os.path.join(path_out, out_filename)
    with open(out_filename, 'w') as outfile:
        json.dump(data_cut, outfile)

def preprocess_verified_data(path):
    '''
    Plots time series of data in given folder.
    :param path: data path with raw data
    '''
    SIGNAL_LEN = 6000
    addrs = glob.glob(path + '*.json')
    path_out = os.path.join(path, '..', 'all_processed')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    for i in trange(len(addrs), desc='Plotting data', leave=True):
        with open(addrs[i]) as json_file:
            data = json.load(json_file) # json file as dict
        #signal = decode_data(data, out_type='list')
        data_cut = _cut_data(data, target_len=SIGNAL_LEN)
        out_filename = str(data['time']) + '_' + str((data['surfaceLocation']['equatorial'],data['surfaceLocation']['height'])) + '_' + data['classification']['name'] + '.json'
        out_filename = os.path.join(path_out, out_filename)
        with open(out_filename, 'w') as outfile:
           json.dump(data_cut, outfile)

def verification(path=None, path_out=None):
    '''
    Apply data verification to all data.
    '''
    data_config = DataConfig('config/DataConfig.json')
    if path == None:
        addrs = glob.glob(os.path.join(data_config.source_dir_local, '*.json'))
    else:
        addrs = glob.glob(os.path.join(path, '*.json'))
    if path_out == None:
        path_out = os.path.join(data_config.source_dir_local, '..', 'collect05_filtered')

    MIN_LEN = 6000
    THRES_RANGE = 150
    THRES_PEAKS = 10

    for i in trange(len(addrs), desc='Verfying data', leave=True):
        signal, label = load_raw_from_json(addrs[i])
        if check_missing_channels(signal):
            continue
        elif check_signal_length(signal, min_len=MIN_LEN):
            continue
        elif check_noise(signal, thres=THRES_RANGE):
            continue
        elif check_peaks(signal, thres=THRES_PEAKS):
            continue
        else:
            preprocess_verified_data_single(addrs[i], path_out=path_out, signal_length=MIN_LEN)
            #shutil.copy(addrs[i], path_out)
