'''
Preprocessing pipeline to extract training data
'''
import glob
import ntpath
import numpy as np
import os
import ray
import shutil
import sys
from tqdm import trange

from config import *
from utils.augmentation import *
from utils.dataio import *
from utils.utils import *
from utils.verification_data import *

#Configurations
data_config = DataConfig('config/DataConfig.json')
model_config = ModelConfig('config/ModelConfig.json')
train_config = TrainConfig('config/TrainConfig.json')
test_config = TestConfig('config/TestConfig.json')

#Initialize ray for multiprocessing
ray.init()

#Verify & preprocess data
data_to_process = data_config.source_dir_local
data_processed = 'data/processed'
if os.path.isdir(data_processed) == False:
    os.mkdir(data_processed)
verification(path=data_to_process, path_out=data_processed)

#Extract features
export_features(path_in=data_processed, path_out='data/features')

#Augment datat
augmentation(path_in='data/processed', path_out='data/augmented')

#Split test data
filter_test_data(path_in='data/features', path_out='data/test')
