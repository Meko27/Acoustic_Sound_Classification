import glob
import itertools
import json
import numpy as np
import ntpath
import os
import pandas as pd
import ray
import sys
from tqdm import trange

from utils.dataio import *
from utils.utils import *

from config import *
data_config = DataConfig('config/DataConfig.json')

def _chunked_iterable(iterable, size):
    '''
    Auxiliary function to iterate over chunks
    '''
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def _augment_data(data, path_out, filename, type_out='npy', augment_callbacks=None):
    '''
    Applies data augmentation methods as defined in augment_callbacks and saves in files.
    :param data: raw data to be augmented as json dict
    :param path_out: path to save the augmented data
    :param filename: filename to save augmented data
    :param type_out: type to save ['json' or 'npy']
    :param augment_callbacks: callbacks determine which augmentation methods to apply
    '''
    
    if type_out == 'json':
        data_decoded = decode_data(data, out_type='list')
        for callback in augment_callbacks:
            data_augmented = {}
            
            data_aug = callback(data_decoded)
            #data_augmented[key] = data_aug
            if not os.path.exists(os.path.join(path_out, callback.__name__)):
                os.makedirs(os.path.join(path_out, callback.__name__))
            data_encoded = encode_data(data_aug)
            data = add_data_to_json(data, data_encoded)
            
            #data.to_json(path_out + callback.__name__ + '/' + filename)
            out_filename = os.path.join(path_out, callback.__name__, filename)
            with open(out_filename, 'w') as outfile:
                json.dump(data, outfile)
    elif type_out == 'npy':
        for callback in augment_callbacks:
            num_time_bin = int(np.ceil(data_config.duration*data_config.sampling_rate/ data_config.hop_length))
            data_aug = callback(data)
            logmel = calc_logmels_as_np(data_aug)[:,:num_time_bin]
            logmel = np.float32(logmel)
            logmel_deltas = np.concatenate((logmel[:,4:-4,:],deltas_single(logmel)[:,2:-2,:],deltas_single(deltas_single(logmel))),axis=-1)
            if not os.path.exists(os.path.join(path_out, callback.__name__)):
                os.makedirs(os.path.join(path_out, callback.__name__))
            out_filename = os.path.join(path_out, callback.__name__, filename)
            np.save(out_filename, logmel_deltas, allow_pickle=False)
    else: 
        sys.exit('No approrpiate output type given.')

def rename_files(addrs):
    for addr in addrs:
        os.rename(addr, addr[:-9] + addr[-4:])

@ray.remote
def augment_data(addr, path_out, callbacks):
    data, label = load_raw_from_json(addr)
    filename = ntpath.basename(addr)[:-5]
    _augment_data(data, path_out=path_out, filename=filename, augment_callbacks=callbacks, type_out='npy')

def augmentation(path_in, path_out, percentage_augmented_data=0.1):
    addrs = glob.glob(os.path.join(path_in, '*.json'))
    callbacks = [time_shift, pitch_shift, add_random_noise]
    for num_batch, batch in enumerate(_chunked_iterable(range(len(addrs[:int(len(addrs)*0.1)])), size=10)):
        for i in batch:
            calc_handle = augment_data.remote(addrs[i], path_out, callbacks=callbacks)
        ray.get(calc_handle)

