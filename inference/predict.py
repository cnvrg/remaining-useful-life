import argparse
import os
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import pickle

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix, classification_report
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *

from sklearn.preprocessing import StandardScaler
import pathlib
import requests
import base64
import json
from io import BytesIO


def download_model_files(FILES):
    """
    Downloads the model files if they are not already present or
    pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(
                current_dir + f'/{f}') and not os.path.exists('/input/cnn/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(BASE_FOLDER_URL + f)
            f1 = os.path.join(current_dir, f)
            with open(f1, "wb") as fb:
                fb.write(response.content)

def download_test_file(URL):
    """
    Downloads the model files if they are not already present or
    pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    if not os.path.exists(
            current_dir + f'/{URL}') and not os.path.exists('/input/cnn/' + URL):
        print(f'Downloading file: {URL}')
        response = requests.get(URL)
        f = URL.split("/")[-1]
        f1 = os.path.join(current_dir, f)
        with open(f1, "wb") as fb:
            fb.write(response.content)

def data_preprocessing(x_test_data, num_col, seq_len):
    '''
    Parameters
    ----------
    x_test_data : test data
    num_col : numeric columns

    Yields
    ------
    num_col : prerpocessed data
    '''
    test_df = x_test_data
    test_df.head(3)
    sequence_length = seq_len

    def gen_sequence(id_df, seq_length, seq_cols):

        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,142),(50,192)
        # 0 50 (start stop) -> from row 0 to row 50
        # 1 51 (start stop) -> from row 1 to row 51
        # 2 52 (start stop) -> from row 2 to row 52
        # ...
        # 141 191 (start stop) -> from row 141 to 191
        for start, stop in zip(range(0, num_elements -
                                     seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    ### GENERATE X TRAIN TEST ###
    x_test_data = []
    for engine_id in test_df['id'].unique():
        for sequence in gen_sequence(
                test_df[test_df.id == engine_id], sequence_length, num_col):
            x_test_data.append(sequence)

    x_test_data = np.asarray(x_test_data)

    return x_test_data


def rec_plot(s, eps=0.10, steps=10):
    '''
    generate reccurence plots
    '''
    d = pdist(s[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


def scaling_data(raw_test_data, num_feat, meta_cols):
    '''
    Parameters
    ----------
    raw_test_data : raw_test_data
    num_feat : numeric features
    meta_cols : meta columns

    Returns
    -------
    raw_test_data : scaled data
    seq_cols : columns to be fed to the model

    '''
    scaler = StandardScaler()
    raw_test_data[num_feat] = scaler.fit_transform(
        raw_test_data[num_feat])
    raw_test_data = raw_test_data.loc[:,
                                      raw_test_data.apply(pd.Series.nunique) != 1]
    seq_cols = list(set(raw_test_data.columns.tolist()) - set(meta_cols))

    return raw_test_data, seq_cols
########## Path changing (whether the input comes from training or non-training ##########

def predict(data):
### SCALE TEST DATA ###
    print('Running Stand Alone Endpoint')
    FILES = ['cnn_model.h5', 'shape_data.csv']
    BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/remaining_useful_life_data/"
    script_dir = pathlib.Path(__file__).parent.resolve()
    download_model_files(FILES)
    test_file = str(data['file'])
    download_test_file(test_file)

    #x_test = pd.read_csv(os.path.join(script_dir,'endpoint_test_data.csv'))
    f = test_file.split("/")[-1]
    x_test = pd.read_csv(os.path.join(script_dir,f))
    cnn = os.path.join(script_dir,'cnn_model.h5')
    cols = pd.read_csv(os.path.join(script_dir,'shape_data.csv'))
    numeric_features = cols['cols'].tolist()
    meta_columns = cols['meta_cols'].tolist()
    seq_len = int(cols.loc[0,'sequence_length'])
    upper_limit = int(cols.loc[0, 'cycles'])
    lower_limit = int(cols.loc[1, 'cycles'])
    id_col = x_test.id.unique()
    
    #predict(x_test, numeric_features, meta_columns, seq_length)
    test_data, sequence_cols = scaling_data(
        x_test, numeric_features, meta_columns)
    result_df = pd.DataFrame()
    print(test_data.columns)
    main_dic = []
    dic = {}
    for i in id_col:
        # for each id in test data
        #'dict_'+vars()[i] = {}
        try:
            if len(id_col)<=1:
                test_data['id'] = id_col[0]
            x_test = test_data[test_data['id'] == i]
            x_test.reset_index(inplace=True)

            data = data_preprocessing(x_test, sequence_cols, seq_len)
            print(data.shape)
            x_test_img = np.apply_along_axis(
                rec_plot, 1, data).astype('float16')

            # load model
            model = load_model(cnn)

            # predict
            model.predict(x_test_img)
            cnn_predicted = model.predict(x_test_img)
            cnn_predicted_df = pd.DataFrame(cnn_predicted)            
            cnn_predicted_df.rename(columns = {0:'>{} Cycles'.format(upper_limit), 1:'{}-{} Cycles'.format(lower_limit,upper_limit), 2:'<{} Cycles'.format(lower_limit)}, inplace = True)

            cnn_predicted_df[cnn_predicted_df.columns] = round(cnn_predicted_df[cnn_predicted_df.columns]*100,2)
            result = cnn_predicted_df.iloc[-1, :].idxmax()
            if result == '>{} Cycles'.format(upper_limit):
                j = 'More than {} cycles left'.format(upper_limit)
            elif result == '{}-{} Cycles'.format(lower_limit,upper_limit):
                j = '{} to {} cycles left'.format(lower_limit,upper_limit)
            elif result == '<{} Cycles'.format(lower_limit):
                j = 'Less than {} cycles left'.format(lower_limit)
            dic['id -> {}'.format(str(i))] = 'prediction -> {}'.format(j)
            #print(dic)

        except BaseException as e:
            dic['id -> {}'.format(i)] = 'prediction -> machine has less rows than the defined sequence length'
    return dic

#text = {'file' : 'https://libhub-readme.s3.us-west-2.amazonaws.com/remaining_useful_life_data/endpoint_test_data.csv'}
#print(predict(text))