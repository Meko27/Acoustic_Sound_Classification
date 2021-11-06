import glob
import json
import numpy as np
import os
from sys import path
from tqdm import trange

from config import *
from utils.dataio import read_label_from_json

train_config = TrainConfig('config/TrainConfig.json')

def sort_list(labels_unsorted):
    '''
    Sort labels while keeping only unique entries.
    '''
    labels = list(set(labels_unsorted))
    labels.sort()
    return labels

def save_labels_as_json(labels, path_out):
    '''
    Saves sorted labels as json file as reference for mapping. 
    '''
    labels_dict = {'labels': labels}
    with open(path_out, 'w') as outfile:
            json.dump(labels, outfile)

def count_classes(label_path):
    '''
    Count number of classes provided by training data. 
    '''
    with open(label_path) as jsonfile:
        labels = json.load(jsonfile)
    num_classes = len(labels)
    return num_classes

def map_location_coord_to_class_vect(location):
    '''
    Maps location coordinate to class vector
    :param location: location coordinate as tuple (e.g. (10,2)), list of tuples or numpy array of tuples
    :return class_vect: numeric class vector as numpy array (e.g. np.array([0,0,0,1,0])) or list of arrays
    '''
    ref_labels_path = train_config.ref_label_path_location
    with open(ref_labels_path) as json_file:
        classes_ref = json.load(json_file)
    if type(location) == list:
        class_vect = np.zeros((len(location), len(classes_ref)))
        for i,loc in enumerate(location):
            if type(loc) == str:
                idx = classes_ref.index(list(eval(loc)))   
            else: 
                idx = classes_ref.index(list(loc))
            class_vect[i,idx] = 1
    elif type(location) == tuple:
        idx = classes_ref.index(location)
        class_vect = np.zeros(len(classes_ref))
        class_vect[idx] = 1
    else:
        location = list(location)
        class_vect = np.zeros((len(location), len(classes_ref)))
        for i,loc in enumerate(location):
            idx = classes_ref.index(list(eval(loc)))
            class_vect[i,idx] = 1
    return class_vect.astype(np.float32)

def map_class_type_to_class_vect(class_label):
    '''
    Maps class label to class vector
    :param class_label: class as string (e.g. stich)
    :return class_vect: numeric class vector as numpy array (e.g. np.array([0,0,0,1,0])) or list of arrays
    '''
    ref_labels_path = train_config.ref_label_path_class
    with open(ref_labels_path) as json_file:
        classes_ref = json.load(json_file)
    if type(class_label) == list or type(class_label) == np.ndarray:
        if type(class_label) == np.ndarray:
            class_label = list(class_label)
        class_vect = np.zeros((len(class_label), len(classes_ref)))
        for i,label in enumerate(class_label):
            idx = classes_ref.index(label.decode('utf-8'))   
            class_vect[i,idx] = 1
    else:
        class_vect = np.zeros(len(classes_ref))
        idx = classes_ref.index(class_label.decode('utf-8'))   
        class_vect[idx] = 1
    return class_vect.astype(np.float32)

def map_class_vect_to_location(class_vect):
    '''
    Maps class vector to location coordinate
    :param class_vect: numeric class vector as numpy array (e.g. np.array([0,0,0,1,0])) or list of arrays
    :return location: location coordinate as tuple (e.g. (10,2)) or list of tuples
    '''
    ref_labels_path = train_config.ref_label_path_location 
    with open(ref_labels_path) as json_file:
        classes_ref = json.load(json_file)
    if class_vect.shape[0] > 1:
        location = []
        for i,vect in enumerate(class_vect):
            idx = np.argmax(vect)
            location.append(classes_ref[idx])
    else:
        idx = np.argmax(class_vect)
        location = classes_ref[idx]
    return location

def map_class_vect_to_class(class_vect):
    '''
    Maps class vector to class 
    :param class_vect: numeric class vector as numpy array (e.g. np.array([0,0,0,1,0])) or list of arrays
    :return class_label: class label as string (e.g. "stich" or list of strings
    '''
    ref_labels_path = train_config.ref_label_path_class
    with open(ref_labels_path) as json_file:
        classes_ref = json.load(json_file)
    if class_vect.shape[0] > 1:
        class_label = []
        for i,vect in enumerate(class_vect):
            idx = np.argmax(vect)
            class_label.append(classes_ref[idx])
    else:
        idx = np.argmax(class_vect)
        class_label = classes_ref[idx]
    return class_label

def load_and_save_labels(path_in, path_out, labeltype):
    '''
    Loads labels from training data and saves reference json file with labels
    '''
    addrs = glob.glob(os.path.join(path_in, '*.json')) 
    labels = []
    if labeltype == 'location':
        for i in trange(len(addrs), desc='Loading Lables', leave=True):
            label_dict = read_label_from_json(addrs[i])
            label = label_dict['location']
            labels.append(label)
    else:
        for i in trange(len(addrs), desc='Loading Lables', leave=True):
            label_dict = read_label_from_json(addrs[i])
            label = label_dict['class']
            labels.append(label)
    labels_sorted = sort_list(labels)
    save_labels_as_json(labels_sorted, path_out=path_out)

