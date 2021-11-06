#from utils.utils import frequency_masking, time_masking
#from utils.utils import pitch_shift
import glob
import itertools
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import ntpath
import os
import pandas as pd
import ray
import re
import scipy.io
import shutil
import sys
from tqdm import trange

from utils.utils import *
from config import *

data_config = DataConfig('config/DataConfig.json')
train_config = TrainConfig('config/TrainConfig.json')

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
    
def _transform_mono_to_numpy(signal_stereo):
    '''
    Transforms each signal in list to numpy array
    :param list signal_stereo: list of mono signals for each channel list([[], [], ..., []])
    :return list data_stereo: list of mono signals saved as np array list([np.array(), ...])
    '''    
    data_stereo = []
    for record in signal_stereo:
        data_stereo.append(np.array(record))
    return data_stereo

def _transform_to_matrix(signal_stereo):
    '''
    Transforms signal into data matrix represented as numpy array
    :param list signal_stereo: list of mono signals for each channel list([[], [], ..., []])
    :return np.ndarray data_matrix: matrix of mono signals saved as np array np.array() shape: num_channelxdim
    '''    
    data_matrix = np.zeros(len(signal_stereo), len(signal_stereo[0])) #num_channel x dim
    for i,record in enumerate(signal_stereo):
        data_matrix[i,:] = np.array(record)
    return data_matrix

def _normalize(signal):
    '''
    Normalize by mean.
    '''
    signal_norm = signal - np.mean(signal)
    return signal_norm

def _get_label_from_filename(filename, labeltype='localization'):
    '''
    Extracts label from filename based on labeltype parameter.
    :param filename: basename of file 
    :param labeltype: defining the type of label ['localization' or 'class']
    :return label: label as string. E.g. '(10, 3)' for localization or 'bump' for classification
    '''
    if labeltype == 'location':
        # label = ntpath.basename(path)[14:-9]
        label = re.search(r"_(.*?)_", filename).group(1)
    elif labeltype == 'class':
        label = filename.split('_')[2][:-4]
    return label

def load_addrs(source_dir, valid_extensions): # TODO add crawl parameter
    '''
    Loads file paths of all imegs in source dir
    :param source_dir: source directory of data
    :return pahts: paths of files to load
    '''
    files = [file for file in os.listdir(source_dir) if any([file.endswith(extension) for extension in valid_extensions])]
    paths = [os.path.join(source_dir, file) for file in files]
    return paths

def load_raw_from_json(path: str, n_sensors: int=20, n_sensor_per_node=4, out_type: str='list', fmt='list4', sftp_client=None):
    '''
    Export raw signal data from json file.
    :param path: source directory to the json file
    :param n_sensors: number of sensors (channels).
    :param n_sensors: number of sensornodes (4 sensors per node).
    :param fmt: sepcifies format of samples saved in json file ['list', 'list4']
    :param sftp_client: sftp client object to open via ssh tunnel
    :return signal_stereo: list of stereo signals (each mono stereo signal is a list of numpy arrays)
    :return label: dict with time as timestamp, signal class represented as categoral string, location represented as tuple and event id
    '''
    #data = pd.read_json(path)
    if sftp_client != None:
        with sftp_client.open(path) as jsonfile:
            data = json.load(jsonfile)
    else:
        with open(path) as jsonfile:
            data = json.load(jsonfile)
    #signals_all = data.sensor_data
    signals_all = data['sensor_data']
    #sound_class = data.classification['name']
    sound_class = data['classification']['name']
    #location_surface = data.surfaceLocation
    location_surface = data['surfaceLocation']
    location = (location_surface['equatorial'], location_surface['height'])
    #event_id = data.id
    event_id = data['id']
    label = {'time': data['time'], 'class': sound_class, 'location': location, 'id': event_id}

    signal_stereo = []
    if fmt == 'list': # format of values ["timestamp", "value"]
        for i in range(n_sensors):
            signal_mono = []
            signal = signals_all[str(i)]
            for i,value in signal[:-1]:
                signal_mono.append(float(value[1])) #-INT16_RANGE
            signal_stereo.append(_normalize(np.array(signal_mono)))
    elif fmt == 'list4': # format of values ["timestamp", ["value0", "value1", "value2", "value3"]]
        for i in range(n_sensors//n_sensor_per_node):
            sensor_node = {'0':[],'1':[],'2':[],'3':[]} # one sensor node with 4 channels
            node_signals = signals_all[str(i)]
            if not node_signals == False: # check for empty channels
                for value in node_signals:
                    sensor_node['0'].append(float(value[1][0])) # [timestamp, [v0,..,v3]]
                    sensor_node['1'].append(float(value[1][1]))
                    sensor_node['2'].append(float(value[1][2]))
                    sensor_node['3'].append(float(value[1][3]))
                signal_stereo.append(_normalize(np.array(sensor_node['0'])))
                signal_stereo.append(_normalize(np.array(sensor_node['1'])))
                signal_stereo.append(_normalize(np.array(sensor_node['2'])))
                signal_stereo.append(_normalize(np.array(sensor_node['3'])))
            else: 
                continue
    else:
        for i in range(n_sensors):
            signal_mono = []
            signal = signals_all[str(i)]
            signal = signal.split(' ')
            for split_string in signal[1:]:
                signal_mono.append(float(split_string.split('\n')[0])) #-INT16_RANGE
            signal_stereo.append(_normalize(np.array(signal_mono)))
    
    valid_out_type = {'list', 'list-numpy', 'datamatrix'}
    if out_type not in valid_out_type:
        raise ValueError("out_type must be of type %r." % valid_out_type)
    if out_type == 'list':
        signal_out = signal_stereo
    elif out_type == 'list-numpy':
        signal_out = _transform_mono_to_numpy(signal_stereo)
    else:
        signal_out = _transform_to_matrix(signal_stereo)
    return signal_out, label

def calc_logmel(mono_signal,
                sr=48000,
                duration=data_config.duration,
                num_freq_bin=data_config.num_freq_bin,
                num_fft=data_config.num_fft,
                hop_length=data_config.hop_length,
                num_channel=20,
                fmin=0.0,
                fmax=48000/2,
                normalize=data_config.normalize):
    '''
    Calcualte mel histogram of mono signal.
    :param np.array mono signal: audio time-series data 
    :param int sr: sample rate
    :param int duration: duration of one sample 
    :param int num_freq_bin: number of Mel bands to generate
    :param int num_fft: number of fft components
    :param int num_channel: number of channels
    :param int fmin: minimum frequency of signal
    :param int fmax: maximum frequency of signal
    :param bool normalize: decides wether to clip feature to 0,1
    :return np.ndarray feat_matrix: Mel spectrogram
    '''
    #calculate parameters
    #hop_length = int(num_fft / 2)
    num_time_bin = int(np.ceil(duration * sr / hop_length))

    logmel_histogram = librosa.feature.melspectrogram(mono_signal, 
                                                     sr=sr, 
                                                     n_fft=num_fft, 
                                                     hop_length=hop_length,
                                                     n_mels=num_freq_bin, 
                                                     fmin=fmin, 
                                                     fmax=fmax, 
                                                     htk=True, 
                                                     norm=None)
    if normalize:
        logmel_histogram = np.log(logmel_histogram+1e-8)
        logmel_histogram = (logmel_histogram - np.min(logmel_histogram)) / (np.max(logmel_histogram) - np.min(logmel_histogram))
    return logmel_histogram

def calc_logmels(raw_data: list,
                 sr=data_config.sampling_rate,
                 duration=data_config.duration,
                 hop_length=data_config.hop_length):
    '''
    Calculate mel spectrograms of a each channel and return feature numpy matrix.
    :param list raw_data: audio time-series data 
    :param int sr: sample rate
    :param int duration: duration of one sample 
    :param int num_freq_bin: number of Mel bands to generate
    :param int num_fft: number of fft components
    :param int num_channel: number of channels
    :return feat_list: logmel histograms as list
    '''
    num_time_bin = int(np.ceil(duration * sr / hop_length))
    feat_list = []
    for i,mono in enumerate(raw_data):
        if len(mono) < duration*sr:
            mono = np.pad(np.array(mono), pad_width=(0,int(duration*sr-len(mono))), mode='constant', constant_values=(0,0))
        else:
            mono = np.array(mono)
        feat_mat = calc_logmel(mono)[:,:num_time_bin]
        feat_list.append(feat_mat)
    return feat_list

def calc_logmels_as_np(raw_data: list,
                       sr=data_config.sampling_rate,
                       duration=data_config.duration,
                       num_freq_bin=data_config.num_freq_bin,
                       hop_length=data_config.hop_length, 
                       num_channel=data_config.num_audio_channel):
    '''
    Calculate mel spectrograms of a each channel and return feature numpy matrix.
    :param list raw_data: audio time-series data 
    :param int sr: sample rate
    :param int duration: duration of one sample 
    :param int num_freq_bin: number of Mel bands to generate
    :param int num_fft: number of fft components
    :param int num_channel: number of channels
    :return feat_matrix: logmel histograms as numpy array with shape (num_freq_bin, num_time_bin, num_channel)
    '''
    num_time_bin = int(np.ceil(duration * sr / hop_length))
    feat_matrix = np.zeros((num_freq_bin, num_time_bin, num_channel))
    for i,mono in enumerate(raw_data):
        if len(mono) < duration*sr:
            mono = np.pad(np.array(mono), pad_width=(0,int(duration*sr-len(mono))), mode='constant', constant_values=(0,0))
        else:
            mono = np.array(mono)
        feat_mat = calc_logmel(mono, 
                               sr=sr, 
                               hop_length=hop_length,
                               num_freq_bin=num_freq_bin, 
                               fmin=0.0, 
                               fmax=sr/2)[:,:num_time_bin]
        feat_matrix[:feat_mat.shape[0],:feat_mat.shape[1],i] = feat_mat
    return feat_matrix
        
def load_data_feat_as_np(paths: str, labeltype=train_config.labeltype, outtype='ndarray'):
    '''
    Load feature data from json file.
    :param paths: paths to files to load
    :param num_feat_bins: number of frequency bins
    :param labeltype: type of label to output ['class', 'localization', 'raw']
    :param outtype: outputtype ['ndarray', 'list']
    :return logmels: Mel spectrograms as numpy matrix with shape (batch_size, num_freq_bin, num_time_bin, num_channel)
    :return labels: labels of specified type ('class', 'location' or 'raw')
    '''
    logmels = []
    labels = []
    if outtype == 'ndarray':
        for i,path in enumerate(paths):
            signal, label = load_raw_from_json(path)
            logmel = calc_logmels_as_np(signal)
            logmels.append(logmel)
            if labeltype is not 'raw':
                labels.append(label[labeltype])
            else: 
                labels.append(label)
        return np.asanyarray(logmels), labels
    else:
        for i,path in enumerate(paths):
            signal, label = load_raw_from_json(path)
            logmel = calc_logmels(signal)
            logmels.append(logmel)
            if labeltype is not 'raw':
                labels.append(label[labeltype])
            else: 
                labels.append(label)
        return logmels, labels

def load_data_feat_from_npy(paths: str, labeltype=train_config.labeltype, outtype='ndarray'):
    '''
    Load feature data from npy file.
    :param paths: paths to files to load
    :param labeltype: type of label to output ['class', 'localization', 'raw'] 
    :param outtype: outputtype ['ndarray', 'list']
    :return logmels: Mel spectrograms as numpy matrix with shape (batch_size, num_freq_bin, num_time_bin, num_channel)
    :return labels: labels of specified type ('class', 'location' or 'raw')
    '''
    logmels = []
    labels = []
    if outtype == 'ndarray':
        for i,path in enumerate(paths):
            label = _get_label_from_filename(os.path.basename(path), labeltype=labeltype)
            logmel = np.load(path)
            logmels.append(logmel)
            labels.append(label)
        return np.asanyarray(logmels), labels
    else:
        for i,path in enumerate(paths):
            label = _get_label_from_filename(os.path.basename(path), labeltype=labeltype)
            logmel = np.load(path)
            logmels.append(logmel)
            if labeltype is not 'raw':
                labels.append(label) #TODO see above
            else: 
                labels.append(label)
        return logmels, labels

def load_data_feat_as_np_single(path: str, labeltype=train_config.labeltype, outtype='ndarray'):
    '''
    Load feature data from json file.
    :param path: path to file to load
    :param num_feat_bins: number of frequency bins
    :param labeltype: type of label to output ['class', 'localization', 'raw']
    :param outtype: outputtype ['ndarray', 'list']
    :return logmel: Mel spectrograms as numpy matrix with shape (num_freq_bin, num_time_bin, num_channel)
    :return label: labels of specified type ('class', 'location' or 'raw')
    '''
    if outtype == 'ndarray':
        signal, label = load_raw_from_json(path)
        logmel = calc_logmels_as_np(signal)
        if labeltype is not 'raw':
            label = label[labeltype]
        return np.asanyarray(logmel), label
    else:
        signal, label = load_raw_from_json(path)
        logmel = calc_logmels(signal)
        if labeltype is not 'raw':
            label = label[labeltype]
        return logmel, label

def load_data_feat_from_npy_sinlge(path: str, labeltype: str='location'):
    '''
    Load feature data from npy file.
    :param path: path to files to load
    :param labeltype: type of label to output ['class', 'localization', 'raw'] 
    :return logmel: Mel spectrogram as numpy matrix with shape (num_freq_bin, num_time_bin, num_channel)
    :return label: label of specified type ('class', 'location' or 'raw') 
    '''
    if labeltype == 'location':
        # label = ntpath.basename(path)[14:-9]
        label = re.search(r"_(.*?)_", ntpath.basename(path)).group(1)
    elif labeltype == 'class':
        label = ntpath.basename(path).split('_')[2][:-4]
    logmel = np.load(path)
    return np.asanyarray(logmel), label

def play_sound(audio, num_channels=1, bytes=2, frequency=48000):
    '''
    Play sound of raw numpy array
    :param np.ndarray audio: n x num_channels time series
    :param int num_channels: number of channels
    :param int bytes: bytes for int to use
    :param int frequency: sampling frequency of sound signal
    '''
    play_obj = sa.play_buffer(audio.astype(np.int16), 1, 2, frequency)
    play_obj.wait_done() #wait for playback to finish befor exiting

def plot_sequence_single(mono_signal, show=False, fname=None, label=None, channel=0):
    '''
    Plots the raw signal over time as timeseries.
    :param mono_signal: mono raw acoustic signal
    :param show: Flag for showing plot
    :return fig: figure 
    '''
    fig_specs = {'num_rows': 1,
                 'num_cols': 1,
                 'fig_size': (16, 4)}
    fig, axs = plt.subplots(fig_specs['num_rows'], 
                            fig_specs['num_cols'], 
                            figsize=fig_specs['fig_size'],
                            sharex=True,
                            sharey=True)
    axs.plot(mono_signal)
    axs.set_xticks([])
    axs.set_yticks([])
    if fname is not None:
        if label is not None:
            plt.savefig(fname + label['id'] + '_' + label['class'] + '_' + str(label['location']) + '_' + str(channel) + '.jpg')
        else:
            plt.savefig(fname + '.jpg')
    if show:
        plt.show()
    plt.close()
    return fig

def plot_sequence(raw_data, label, show=False, fname=None, num_samples_to_plot=None):
    '''
    Plots the raw signal over time as timeseries.
    :param raw_data: list of signals for each channel
    :param show: Flag for showing plot
    :param num_samples_to_plot: number of samples to plot signale[:num_samples_to_plot]
    :return fig: figure 
    '''
    fig_specs = {'num_rows': 20,
                 'num_cols': 1,
                 'fig_size': (20, 20)}
    fig, axs = plt.subplots(fig_specs['num_rows'], 
                            fig_specs['num_cols'], 
                            #gridspec_kw={'wspace':1, 'hspace':0},
                            figsize=fig_specs['fig_size'],
                            sharex=True
                            #sharey=True
                            )
    # set limits for plot
    y_max_limit = 0
    y_min_limit = 0
    for data in raw_data:
        try:
            if max(data) > y_max_limit:
                y_max_limit = max(data)
            if min(data) < y_min_limit:
                y_min_limit = min(data)
        except:
            continue
    if num_samples_to_plot == None:
        for i,data in enumerate(raw_data):
            #if data.shape[0] < num_samples_to_plot:
            #    continue
            try:
                axs[i].plot(data)
                axs[i].set_xlabel('time', fontsize=20)
                axs[i].set_ylabel('{:02d}'.format(i), fontsize=20)
                #axs[i].set(title='Channel {}'.format(i))
                axs[i].label_outer()
                fig.suptitle('Class: ' + label['class'] + ', Location: ' + str(label['location']), fontsize=30)
                #y_min_limit = -25000
                #y_max_limit = 25000
                axs[i].set_ylim((y_min_limit, y_max_limit))
                #axs[i].set_yticks([])
                x_min_limit = 0
                #axs[i].set_xlim((x_min_limit, x_max_limit))
                #axs[i].set_xticks([])
            except:
                continue
    else:
        for i,data in enumerate(raw_data):
            #if data.shape[0] < num_samples_to_plot:
            #    continue
            try:
                axs[i].plot(data[:num_samples_to_plot])
                axs[i].set_xlabel('time', fontsize=20)
                axs[i].set_ylabel('{:02d}'.format(i), fontsize=20)
                #axs[i].set(title='Channel {}'.format(i))
                axs[i].label_outer()
                fig.suptitle('Class: ' + label['class'] + ', Location: ' + str(label['location']), fontsize=30)
                #y_min_limit = -25000
                #y_max_limit = 25000
                axs[i].set_ylim((y_min_limit, y_max_limit))
                #axs[i].set_yticks([])
                x_min_limit = 0
                x_max_limit = num_samples_to_plot
                axs[i].set_xlim((x_min_limit, x_max_limit))
                #axs[i].set_xticks([])
            except:
                continue
    if fname is not None:
        if label is not None:
            plt.savefig(fname + str(label['time']) + '_' + str(label['location']) + '_' + label['class'] +'.jpg') #label['id']['id']
        else:
            plt.savefig(fname + '.jpg')
    if show:
        plt.show()
    plt.close()
    return fig

def show_spectrogram_single(logmel, fname=None, label=None, show=False, channel=0):
    '''
    Save a single mel spectrogram
    :param logmel: logmel spectrogram
    :param fname: name of file to save
    :param label: label of signal as dict with keys id, class, location
    :param show: flag which determines to show plot
    :return fig: figure
    '''
    fig_specs = {'num_rows': 1,
                 'num_cols': 1,
                 'fig_size': (6, 4)}
    fig, axs = plt.subplots(fig_specs['num_rows'], 
                            fig_specs['num_cols'], 
                            #gridspec_kw={'wspace':1, 'hspace':0},
                            figsize=fig_specs['fig_size'],
                            sharex=True,
                            sharey=True)
    img = librosa.display.specshow(librosa.power_to_db(logmel, ref=np.max), #since logmel is in mel scale (power scale)
                                   y_axis='mel', 
                                   fmax=24000,
                                   x_axis='time',
                                   ax = axs)
    fig.colorbar(img, ax= axs,format='%+2.0f dB')
    axs.label_outer()
    if fname is not None:
        if label is not None:
            plt.savefig(fname + label['id'] + '_' + label['class'] + '_' + str(label['location']) + '_' + str(channel) + '.jpg')
        else:
            plt.savefig(fname + '.jpg')
    if show:
        plt.show()
    plt.close()
    return fig
    
def show_spectrogram(logmels, fname=None, label=None, show=False, counter=None, ref=1):
    '''
    Plots and saves mel-spectrogram of signal
    :param logmels: logmels in a list
    :param fname: name of file to save
    :param label: label of signal
    :param show: flag which determines to show plot    
    :return fig: figure
    '''
    fig_specs = {'num_rows': 4,
                'num_cols': 5,
                'fig_size': (12, 9)}
    fig, axs = plt.subplots(fig_specs['num_rows'], 
                            fig_specs['num_cols'], 
                            #gridspec_kw={'wspace':1, 'hspace':0},
                            figsize=fig_specs['fig_size'],
                            sharex=True,
                            sharey=True)
    for i,logmel in enumerate(logmels):
        idx = np.unravel_index(i, (fig_specs['num_rows'], fig_specs['num_cols']))
        img = librosa.display.specshow(librosa.power_to_db(logmel, ref=ref), #since s is in mel scale (power scale)
                                       y_axis='mel', 
                                       fmax=24000,
                                       x_axis='time',
                                       ax = axs[idx])
        if idx[1] == fig_specs['num_cols']-1: #colorbar only at the right spectrograms
            #fig.colorbar(img, ax= axs[idx],format='%+2.0f dB')  
            axs[idx].set_xlabel('')              
            if idx[0] == fig_specs['num_rows']-1: #x-label at last row
                axs[idx].set_xlabel('time')
        axs[idx].set(title='Channel {}'.format(i))
        axs[idx].label_outer()
        fig.suptitle('Class: ' + label['class'] + ', Location: ' + str(label['location']))
    fig.colorbar(img, ax= axs,format='%+2.0f dB')
    if fname is not None:
        if label is not None:
            plt.savefig(fname + str(label['time']) + '_' + str(label['location']) + '_' + label['class'] + '_' + counter + '.jpg') #label['id']['id'] 
        else:
            plt.savefig(fname + '.jpg')
    if show:
        plt.show()
    plt.close()
    return fig

def save_data(path, data, label):
    '''
    Saves a record of data as jpg files, each channel into one jpeg.
    :param path: path to target data folder
    :param data: list of mel features (one entry for each channel)
    :param label: dict with meta data
    '''
    filename_base = path
    for i,channel in enumerate(data):
        cv2.imwrite(filename_base + label['id'] + '_{:02d}.jpg'.format(i), channel)

def read_label_from_json(addr):
    '''
    Reads label from a json file
    :param addr: addr of json file containing raw signal data
    :return label: single label as dict with keys: ['time', 'class', 'location', 'id']
    '''
    with open(addr) as json_file:
        data = json.load(json_file)
    timestamp = data['time']
    sound_class = data['classification']['name']
    location_surface = data['surfaceLocation']
    location = (location_surface['equatorial'], location_surface['height'])
    event_id = data['id']
    label = {'time': timestamp, 
             'class': sound_class, 
             'location': location, 
             'id': event_id}
    return label

def calc_data_stats(addrs, path_out=None):
    '''
    Saves locations of training data.
    :param addrs: addrs of json files containing raw signals
    :param path_out: file path to save csv file
    :return labels: returns labels of addrs as pandas DataFrame
    '''
    labels = pd.DataFrame(columns=['time', 'class', 'location', 'id'])
    for i in trange(len(addrs), desc='Extracting labels', leave=True):
        label = read_label_from_json(addrs[i])
        label['time'] = pd.to_datetime(label['time'], unit='ms')
        labels = labels.append(label, ignore_index=True)
    if path_out is not None:
        labels.to_csv(path_out, sep=',')
    return labels

def decode_data(data, out_type='list', n_sensors=20, n_sensor_per_node=4):
    '''
    Decodes data from json to targes format
    :param data: encoded data in json file
    :return data_decoded: decoded data in dict for each channel
    '''
    data_decoded = {'00':[],'01':[],'02':[],'03':[], 
                    '10':[],'11':[],'12':[],'13':[],
                    '20':[],'21':[],'22':[],'23':[],
                    '30':[],'31':[],'32':[],'33':[],
                    '40':[],'41':[],'42':[],'43':[]} # one sensor node with 4 channels
    for i in range(n_sensors//n_sensor_per_node):
            node_signals = data['sensor_data'][str(i)]
            for j,value in enumerate(node_signals):
                data_decoded[str(i)+'0'].append(float(value[1][0])) # [timestamp, [v0,..,v3]]
                data_decoded[str(i)+'1'].append(float(value[1][1]))
                data_decoded[str(i)+'2'].append(float(value[1][2]))
                data_decoded[str(i)+'3'].append(float(value[1][3]))
    if out_type == 'list':
        data_list = []
        for key in data_decoded.keys():
            data_list.append(data_decoded[key])
        data_decoded = data_list
    return data_decoded

def encode_data(data, n_sensor_per_node=4):
    '''
    Encodes data into json format.
    :param data: decoded data
    :param data_encoded: encoded data
    '''
    data_encoded = {'0':[], '1':[], '2':[], '3':[], '4':[]}
    for i,key in enumerate(data_encoded.keys()):
        data_channel = []
        #for j in range(len(data[key]+'0')):
        #    channels = [data[key + '0'][j],
        #                data[key + '1'][j],
        #                data[key + '2'][j],
        #                data[key + '3'][j]]
        for j in range(len(data[i*4])):
            channels = [data[i*4+0][j],
                        data[i*4+1][j],
                        data[i*4+2][j],
                        data[i*4+3][j]]
            data_channel.append(channels)
        data_encoded[key] = data_channel
    return data_encoded

def add_data_to_json(data, data_encoded):
    '''
    Adds the encoded data into the json file by replacing the sensor data but keeping the time stamp.
    :param data: json file read as dict
    :param data_encoded: encoded data into desired format [[c0,c1,c2,c3], [c0,c1,c2,c3], ...]
    '''
    for i,key in enumerate(data['sensor_data'].keys()):
        for j in range(len(data['sensor_data'][key])):
            data['sensor_data'][key][j][1] = data_encoded[key][j]
    return data

@ray.remote
def calc_and_save_features(addr, path_out):
    '''
    Calculates mel spectrograms and saves to disk.
    '''
    logmel, label = load_data_feat_as_np_single(addr)
    logmel_deltas = np.concatenate((logmel[:,4:-4,:],deltas_single(logmel)[:,2:-2,:],deltas_single(deltas_single(logmel))),axis=-1)
    np.save(os.path.join(path_out, ntpath.basename(addr)[:-5]), logmel_deltas)

### Scripting ###
def export_data_stats():
    path_in = '/media/pandadgx/184e5765-265c-4898-8188-780df8e9608a/MeKo/projects/audi/data/localization/all_processed/'  
    path_out = '/media/pandadgx/184e5765-265c-4898-8188-780df8e9608a/MeKo/projects/audi/data/localization/all_processed.csv'  
    addrs = glob.glob(path_in + '*.json')
    addrs.sort()
    calc_data_stats(addrs, path_out=path_out)

def export_features(path_in=None, path_out=None):
    if path_in == None:
        path_in = data_config.source_dir_local
    addrs = glob.glob(os.path.join(path_in, '*.json'))
    addrs.sort()
    for num_batch, batch in enumerate(_chunked_iterable(range(len(addrs)), size=20)):
        for i in batch:
            calc_handle = calc_and_save_features.remote(addrs[i], path_out)
        ray.get(calc_handle)

def filter_train_data():
    import shutil
    path_in = "/media/pandadgx/184e5765-265c-4898-8188-780df8e9608a/MeKo/projects/audi/data/localization/all_raw/"
    path_out = "/media/pandadgx/184e5765-265c-4898-8188-780df8e9608a/MeKo/projects/audi/data/localization/all_filtered/"
    addrs = glob.glob(os.path.join(path_in, '*.npy'))
    addrs.sort()
    for i in range(0,len(addrs),10):
        filename = ntpath.basename(addrs[i])
        shutil.copyfile(addrs[i], os.path.join(path_out, filename))

def filter_test_data(path_in, path_out):
    addrs = glob.glob(os.path.join(path_in, '*.npy'))
    addrs.sort()
    for i in range(0,len(addrs),10):
        filename = ntpath.basename(addrs[i])
        shutil.move(addrs[i], os.path.join(path_out, filename))

def filter_other_types():
    import shutil
    path_in = "/media/pandadgx/184e5765-265c-4898-8188-780df8e9608a/MeKo/projects/audi/data/features_collect05/"
    path_out = "/media/pandadgx/184e5765-265c-4898-8188-780df8e9608a/MeKo/projects/audi/data/features_collect05_othertypes/"
    addrs = glob.glob(os.path.join(path_in, '*.npy'))
    for addr in addrs:
        if 'bump' not in addr:
            filename = ntpath.basename(addr)
            shutil.move(addr, os.path.join(path_out, filename))
