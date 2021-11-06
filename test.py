import numpy as np
import sys
import tensorflow as tf
import os
import pandas as pd

from config import *
from models.network import model_fcnn
from utils.training_funcs import *
from utils.process_labels import *
from utils.dataio import *
from utils.utils import *

data_config = DataConfig('config/DataConfig.json')
model_config = ModelConfig('config/ModelConfig.json')
train_config = TrainConfig('config/TrainConfig.json')
test_config = TestConfig('config/TestConfig.json')

def create_model(model_config, data_config, train_config):
    model = model_fcnn(model_config.num_classes, 
                       input_shape=[data_config.num_freq_bin, None, 3*data_config.num_audio_channel], 
                       num_filters=model_config.num_filters, 
                       wd=model_config.wd) 
    model.compile(loss='categorical_crossentropy',
                #   optimizer=tf.keras.optimizers.SGD(lr=train_config.max_lr, decay=train_config.decay, momentum=train_config.momentum, nesterov=False),
                  optimizer =tf.keras.optimizers.Adam(learning_rate=train_config.max_lr, decay=train_config.decay),
                  metrics=['accuracy'])
    return model

file_list_test = os.listdir(test_config.path_test_data) # path to test data
file_list_training = os.listdir(data_config.source_dir_training) # path to train data
folder_list = [test_config.path_test_data]

if test_config.use_training_data:
    n = len(file_list_test) + len(file_list_training)
    folder_list.append(data_config.source_dir_training)
    file_list = file_list_test + file_list_training
else:
    n = len(file_list_test)
    file_list = file_list_test

rows_for_dataframe = []

# Load the trained model
model = create_model(model_config, data_config, train_config)
model.load_weights(test_config.path_model)
#model = tf.saved_model.load(test_config.path_model)
#model = tf.keras.models.load_model(test_config.path_model)

print('Loaded Model, testing on {} files from {}'.format(n, folder_list))

### Testing on test files
for i, file in enumerate(file_list_test):
    # Load and prepare data to be tested
    x_test, y_test = load_data_feat_from_npy_sinlge(os.path.join(test_config.path_test_data, file))
    x_test = np.float32(x_test)
    y_test = list(map(int, y_test[1:-1].split(', ')))

    # Pass the data through the model
    y_pred = model(x_test[None, ...], training=False)
    y_pred_loc = map_class_vect_to_location(y_pred)

    distance = np.linalg.norm(np.array(y_test) - np.array(y_pred_loc))
    print('{}/{} | Ground Truth: {}, Prediction: {}, Confidence: {:.4f}, Euclidean distance: {:.2f}'.format(i+1, n, y_test, y_pred_loc, np.max(y_pred), distance))

    rows_for_dataframe.append([y_test, y_pred_loc, np.max(y_pred), distance])

### Testing on training files
if test_config.use_training_data:
    for j, file in enumerate(file_list_training):
        # Load and prepare data to be tested
        x_test, y_test = load_data_feat_from_npy_sinlge(os.path.join(test_config.path_training_data, file))
        y_test = list(map(int, y_test[1:-1].split(', ')))

        # Pass the data through the model
        y_pred = model(x_test[None, ...], training=False)
        y_pred_loc = map_class_vect_to_location(y_pred)

        distance = np.linalg.norm(np.array(y_test) - np.array(y_pred_loc))
        print('{}/{} | Ground Truth: {}, Prediction: {}, Confidence: {:.4f}, Euclidean distance: {:.2f}'.format(len(file_list_test)+j+1, n, y_test, y_pred_loc, np.max(y_pred), distance))

        rows_for_dataframe.append([y_test, y_pred_loc, np.max(y_pred), distance])

df = pd.DataFrame(rows_for_dataframe, file_list, ['groundtruth', 'prediction', 'confidence', 'distance'])
if not os.path.exists(test_config.savepath_dataframe):
    os.mkdir(test_config.savepath_dataframe)

modelname = os.path.basename(os.path.dirname(test_config.path_model))
epoch = str(os.path.basename(test_config.path_model)).split('-')[1]

if test_config.use_training_data:
    filename = modelname + '_ep' + epoch + '_with_training_data'+'.csv'
else:
    filename = modelname + '_ep' + epoch + '.csv'
    #filename = os.path.dirname(test_config.path_model) + '.csv'
df.to_csv(os.path.join(test_config.savepath_dataframe, filename))
