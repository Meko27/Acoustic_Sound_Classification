import numpy as np
import sys
# import tensorflow as tf
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from config import *
from models.network import model_fcnn
from utils.training_funcs import *
from utils.process_labels import *
from utils.dataio import *
from utils.utils import *

test_config = TestConfig('config/TestConfig.json')

def calc_variation_in_model(path_to_csv: str, parts: bool, save=True) -> pd.DataFrame:
    '''
    Calculates for each location the variation in the model outputs for samples of this location, measured by
    the percentage of inputs that were mapped to the most common output.
    :param path_to_csv: Path to the .csv file that contains the results from the testing of the model
    :param parts: If True, the Model outputs are classified to the car parts
    :return: pandas DataFrame that contains the statistics for the variation
    '''
    path_partsmap = 'data/partsmap/partsmap.csv'
    path_partnames = 'data/partsmap/partnames.csv'

    df = pd.read_csv(path_to_csv, header=0, index_col=0)

    n = len(df)
    df = df[df['confidence']>=test_config.confidence_threshold]
    localized = len(df)

    if parts:
        df_map = pd.read_csv(path_partsmap, sep=';', header=0, index_col=[0, 1])
        df_names = pd.read_csv(path_partnames, sep=';', header=0, index_col=0)

    # Reform table to have multiindex [groundtruth, filename]
    df = df.reindex(pd.Index(df.index.to_list(), name='filename'))
    df.reset_index(inplace=True)
    df = df.set_index(['groundtruth']).sort_index().set_index('filename', append=True)

    if parts:
        df['groundtruth_index'] = df.apply(lambda row: df_map.loc[(list(map(int, row.name[0][1:-1].split(', ')))[0], list(map(int, row.name[0][1:-1].split(', ')))[1]), 'partindex'], axis=1)
        df['groundtruth_name'] = df.apply(lambda row: df_names.loc[row['groundtruth_index'], 'name'], axis=1)
        df['prediction_index'] = df.apply(lambda row: df_map.loc[(list(map(int, row['prediction'][1:-1].split(', ')))[0], list(map(int, row['prediction'][1:-1].split(', ')))[1]), 'partindex'], axis=1)
        df['prediction_name'] = df.apply(lambda row: df_names.loc[row['prediction_index'], 'name'], axis=1)

    # List of groundtruth locations
    ground_locs = df.index.get_level_values(0).drop_duplicates().to_list()

    new_rows = []

    for ground_loc in ground_locs:
        if parts:
            predictions = df.loc[ground_loc, ['groundtruth_index', 'groundtruth_name', 'prediction_index', 'prediction_name']].value_counts()
        else:
            predictions = df.loc[ground_loc, 'prediction'].value_counts()
        
        # if parts, this produces a tuple of ('groundtruth_index', 'groundtruth_name', 'prediction_index', 'prediction_name')
        max_prediction = predictions.idxmax()
        num_max_prediction = predictions.loc[max_prediction]
        freq_max_prediction = num_max_prediction / predictions.sum()
        num_predicted_classes = predictions.count()
        num_samples_of_ground_loc = predictions.sum()
        if parts:
            new_rows.append([max_prediction[0], max_prediction[1], max_prediction[2], max_prediction[3], freq_max_prediction, num_predicted_classes, num_samples_of_ground_loc])
        else:
            new_rows.append([max_prediction, freq_max_prediction, num_predicted_classes, num_samples_of_ground_loc])
    
    if parts:
        df_var = pd.DataFrame(new_rows, df.index.get_level_values(0).drop_duplicates(), ['truepartindex', 'truepartname', 'maxpredictionpartindex', 'maxpredictionpartname', 'freqmaxprediction', 'numpredclasses', 'numsamples'])
    else:
        df_var = pd.DataFrame(new_rows, df.index.get_level_values(0).drop_duplicates(), ['maxprediction', 'freqmaxprediction', 'numpredclasses', 'numsamples'])

    if save:
        if parts:
            filename = path_to_csv[:-4]+'_conf>='+str(test_config.confidence_threshold)+'_loc={:.4f}'.format(localized/n)+'_variations_by_parts.csv'
        else:
            filename = path_to_csv[:-4]+'_conf>='+str(test_config.confidence_threshold)+'_loc={:.4f}'.format(localized/n)+'_variations.csv'
        df_var.to_csv(filename)

    return df_var

def _load_column(path_in, col_name):
    '''
    Load column from csv as list
    :param path_in: path to csv file
    :parm col_name: column name to load
    :return y_out: column of csv file
    '''
    df = pd.read_csv(path_in)
    return df[col_name]
        
def _calc_acc_by_parts(y_true, y_pred):
    '''
    Calculates accuracy for every class.
    :param y_true: list containing the ground true labels (id of parts)
    :param y_pred: list containing the predictions (id of parts)
    :return results: dict with results in formalt {'classA': acc, 'classB': acc, ..., 'total': acc}
    '''
    labels=list(np.linspace(0,15,16,dtype='int'))
    acc_total = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    accuracies = cm.diagonal()
    accuracies_dict = {}
    for i,acc in enumerate(list(accuracies)):
        accuracies_dict[str(labels[i])] = acc
    accuracies_dict['total'] = acc_total
    return accuracies_dict

def calc_acc_by_parts(path_in, path_out=None, col_true='truepartindex', col_pred='maxpredictionpartindex'):
    '''
    Calculates acc of every class and saves as json
    :param path_in: path to csv with results
    :param path_out: output path to save json file
    :param col_true: column name denoting ground truth
    :param col_pred: column name denoting predictions
    '''
    if path_out == None:
        path_out = os.path.splitext(path_in)[0] + '_acc.json'
    y_true = _load_column(path_in, col_true)
    y_pred = _load_column(path_in, col_pred)
    acc = _calc_acc_by_parts(y_true, y_pred)
    with open(path_out, 'w') as outfile:
        json.dump(acc, outfile)

def calc_all_acc_without_threshold():
    files = list(set(glob.glob('eval/*.csv')) ^ set(glob.glob('eval/*_by_parts.csv')))
    for file in files:
        test_config.confidence_threshold = 0.0
        test_results = calc_variation_in_model(file, parts=True, save=False)
        y_true = test_results['truepartindex']
        y_pred = test_results['maxpredictionpartindex']
        acc = _calc_acc_by_parts(y_true, y_pred)
        with open('eval/without_threshold/' + os.path.splitext(os.path.basename(file))[0]+'variations_by_parts_acc.json', 'w') as outfile:
            json.dump(acc, outfile)


if __name__ == '__main__':
    #calc_variation_in_model('eval/adam_multi_nonormalize_arch00_lr01_schedule01_6channels_ep99.csv', parts=True)
    calc_acc_by_parts(path_in='path_to_test_csv')
    #calc_all_acc_without_threshold()