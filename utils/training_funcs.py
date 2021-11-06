import numpy as np
import ntpath
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import threading
import sys

from utils.process_labels import map_location_coord_to_class_vect
from utils.utils import *
from utils.dataio import *

from config import *
data_config = DataConfig('config/DataConfig.json')
train_config = TrainConfig('config/TrainConfig.json')
model_config = ModelConfig('config/ModelConfig.json')

class LR_WarmRestart(tf.keras.callbacks.Callback):
    '''
    A learning rate warm restart scheduler for training the neural network in a more effective way. 
    '''
    def __init__(self, nbatch, initial_lr, min_lr, epochs_restart, Tmult):
        '''
        Initialzize learning reate scheduler
        :param int nbatch: number of batches in the training prozess
        :param int initial_lr: initial learning rate to start with
        :param int min_lr: minimum learning rate during training
        :param int epochs_restart: number of epochs, for restarting the learning rate (e.g. [10,50,90])
        :param int Tmult: factor to increase lr at every restart
        '''
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs_restart = epochs_restart
        self.nbatch = nbatch
        self.currentEP=0
        self.startEP=0
        self.Tmult=Tmult
        
    def on_epoch_begin(self, epoch, logs=None):
        '''
        Sets the params for lr schedule at beginn of epoch.
        '''
        if epoch+1<self.epochs_restart[0]:
            self.currentEP = epoch
        else:
            self.currentEP = epoch+1
            
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.Tmult=2*self.Tmult
        
    def on_epoch_end(self, epochs, logs=None):
        '''
        Sets the params for lr scheudle at end of epoch.
        '''
        lr = K.get_value(self.model.optimizer.lr)
        print ('\nLearningRate:{:.6f}'.format(lr))
    
    def on_batch_begin(self, batch, logs=None):
        '''
        Sets the params for lr scheudle at beginn of batch.
        '''
        pts = self.currentEP + batch/self.nbatch - self.startEP
        decay = 1+np.cos(pts/self.Tmult*np.pi)
        lr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,lr)

class Dataset():
    '''
    A tf.dataset for loading and manipulating the data during training in an efficient way.
    '''
    def __init__(self,
                 paths, 
                 feat_dim, 
                 hop_length=128,
                 duration=0.125, 
                 sr=48000, 
                 batch_size=32, 
                 crop_percentage=0.1,
                 alpha=0.2, 
                 shuffle=True, 
                 num_classes=1, 
                 do_mixup=False,
                 normalize=False,
                 labeltype='location'): 
        '''
        Initialize dataset class.
        :param list paths: list
        :param list feat_dim:
        :param int hop_length:
        :param float duration:
        :param int sr
        :param int batch_size
        :param float crop_percentage
        :param float alpha
        :param bool shuffle
        :param int um_classes
        :param bool do_mixup
        :param bool normalize
        :param str labeltype
        '''
        self.paths = paths
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.do_mixup = do_mixup
        self.normalize = normalize
        self.NewLength = int(np.ceil(duration * sr / hop_length)*(1-crop_percentage)) #crop_length
        self.labeltype = labeltype
        #dir_augmented = [os.path.join(data_config.source_dir_augmented, 'time_shift', '*.npy')]
        #dir_augmented.append(os.path.join(data_config.source_dir_augmented, 'pitch_shift', '*.npy'))
        #dir_augmented.append(os.path.join(data_config.source_dir_augmented, 'add_random_noise', '*.npy'))
        #self.list_ds = tf.data.Dataset.list_files([paths + '*.npy'] + dir_augmented + [data_config.source_dir_further_data + '*.npy'])
        self.list_ds = tf.data.Dataset.list_files(paths)

    def _parse_logmel(self, filenames_ds):
        '''
        Load the data and label from a tensorflow op representing the filepath
        :param tf.dataset filenames_ds: tensorflow dataset of listed filenames 
        :return tf.tenor logmel: tensor with features
        :return tf.tensor label: tensor with labels
        '''
        if self.labeltype == 'location':
            label = str(ntpath.basename(filenames_ds.numpy())).split('_')[1]
        elif self.labeltype == 'class':
            label = str(ntpath.basename(filenames_ds.numpy())).split('_')[2][:-5]
        else:
             raise ValueError('labeltype must be either <location> or <class>')
        logmel = np.load(filenames_ds.numpy(), allow_pickle=True)
        logmel = np.float64(logmel)
        if self.normalize:
            logmel = np.log(logmel+1e-8)
            logmel = (logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel))
        return tf.convert_to_tensor(logmel), label

    def _augment_dataset(self, batch_ds, label):
        '''
        Apply functions to augment dataset
        :param tf.dataset batch_ds: tensor with batched data 
        :param tf.tensor label: tesnor with labels of batches
        :return tf.tensor dataset: augmented tensor of data
        :return np.ndarray label: corresponding labels as numpy arrays
        '''
        X1 = batch_ds.numpy()
        #red_indexes_with_deltas = data_config.red_channels + [red+20 for red in data_config.red_channels] + [red+40 for red in data_config.red_channels]
        #X1 = X1[:,:,:,red_indexes_with_deltas]
        for j in range(X1.shape[0]):
            # spectrum augment
            for c in range(X1.shape[3]):
                X1[j, :, :, c] = frequency_masking(X1[j, :, :, c])
                X1[j, :, :, c] = time_masking(X1[j, :, :, c])
            # random cropping
            StartLoc1 = np.random.randint(0,X1.shape[2]-self.NewLength)
            X1[j,:,0:self.NewLength,:] = X1[j,:,StartLoc1:StartLoc1+self.NewLength,:]
        X1 = X1[:,:,0:self.NewLength,:]
        X1_shape = X1.shape
        if self.labeltype == 'location':
            label = map_location_coord_to_class_vect(label.numpy())
        else:
            label = map_class_type_to_class_vect(label.numpy())
        label_shape = label.shape
        label = tf.convert_to_tensor(label)
        label.set_shape(list(label_shape))
        X1 = tf.convert_to_tensor(X1)
        X1.set_shape(list(X1_shape))
        return X1, label

    def _set_shape(self, batch_ds, label):
        '''
        Set shape of tensors.
        :param tf.dataset batch_ds: batched dataset
        :param np.ndarray label: corresponding labels as numpy arrays
        :return tf.dataset batch_ds: batched dataset with right shape
        :return np.ndarray label: labels as numpy array
        '''
        batch_ds.set_shape([None, 128, 169, 60]) # TODO hard coded --> integrate to config
        label.set_shape([None, self.num_classes])
        tf.ensure_shape(batch_ds, [None, 128, 169, 60]) # TODO hard coded --> integrate to config
        tf.ensure_shape(label, [None, self.num_classes])
        return batch_ds, label
        
    def create_dataset(self):
        '''
        Create augmented dataset by mapping functions over the dataset
        :return tf.dataset logmel_augmented_ds: tf.dataset, which is batched and augmented by functions denoted by _augment_dataset
        '''
        logmel_ds = self.list_ds.map(lambda x: tf.py_function(self._parse_logmel, [x], [tf.float64, tf.string]))
        #len_dataset = tf.data.experimental.cardinality(logmel_ds)
        if self.shuffle:
            logmel_ds = logmel_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(self.batch_size)
        else:
            logmel_ds = logmel_ds.batch(self.batch_size)
        logmel_augmented_ds = logmel_ds.map(lambda x,y: tf.py_function(self._augment_dataset, [x,y], [tf.float64, tf.float32]))
        logmel_augmented_ds = logmel_augmented_ds.map(self._set_shape)
        return logmel_augmented_ds
