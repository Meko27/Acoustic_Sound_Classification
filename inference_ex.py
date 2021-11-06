import numpy as np
import os
import tensorflow as tf
import time

from config import *
from model.fcnn_att import model_fcnn, create_model
from preprocess_labels import map_class_vect_to_location
from utils.dataio import load_data_feat_as_np_single
from utils.utils import deltas_single

data_config = DataConfig('config/DataConfig.json')
model_config = ModelConfig('config/ModelConfig.json')
train_config = TrainConfig('config/TrainConfig.json')
test_config = TestConfig('config/TestConfig.json')

# Create model and load weights
model = create_model(model_config, data_config, train_config)
model.load_weights(test_config.path_model)

# Load testfile
FILEPATH = 'path_to_testfile' #json raw data
t0 = time.time()
x_test, y_test = load_data_feat_as_np_single(FILEPATH)
x_test = np.concatenate((x_test[:,4:-4,:],deltas_single(x_test)[:,2:-2,:],deltas_single(deltas_single(x_test))),axis=-1)
t1 = time.time()

# Inference
t2 = time.time()
y_pred = model(x_test[None, ...], training=False)
y_pred_loc = map_class_vect_to_location(y_pred)
t3 = time.time()
print('Ground Truth: {}\nPredicted: {}'.format(y_test, y_pred_loc))
print('Calculation time loading data {:.4f}\nInference: {:.4f}'.format(t1-t0, t3-t2))