'''Data Preprocessing'''
import warnings
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
warnings.filterwarnings(action='ignore')
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

def gen_sequence(id_df, seq_length, seq_cols):
    ''' Generate sequence as per user defination'''
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


def gen_labels(id_df, seq_length, label_col):
    ''' Generate Labels'''
    data_matrix = id_df[label_col].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label
    # as target.
    return data_matrix[seq_length:num_elements, :]

def reshape_save_array(arr, file_name):
    '''Reshape saved array'''
    # reshaping the array from 3D matrice to 2D matrice.
    arr_reshaped = arr.reshape(arr.shape[0], -1)
    # saving reshaped array to file.
    filename = cnvrg_workdir + '/' + file_name
    np.savetxt(filename, arr_reshaped)


def save_array(arr, file_name):
    '''save array'''
    # saving reshaped array to file.
    filename = cnvrg_workdir + '/' + file_name
    np.savetxt(filename, arr)
        
if __name__ == '__main__':

    #########parser arguments#############
    parser = argparse.ArgumentParser(description="""Preprocessor""")
    parser.add_argument(
        "--raw_train_data",
        action="store",
        dest="raw_train_data",
        default="/input/s3_connector/remaining_useful_life_data/raw_train_data.csv",
        required=True,
        help="""raw train data""",
    )
    parser.add_argument(
        "--common_letter_numeric",
        action="store",
        dest="common_letter_numeric",
        default="",
        required=False,
        help="""common letter in numeric features""",
    )
    parser.add_argument(
        "--numeric_features",
        action="store",
        dest="numeric_features",
        default="",
        required=True,
        help="""numeric features""",
    )
    parser.add_argument(
        "--meta_columns",
        action="store",
        dest="meta_columns",
        default="",
        required=True,
        help="""meta columns""",
    )
    parser.add_argument(
        "--sequence_length",
        action="store",
        dest="sequence_length",
        default="50",
        required=True,
        help="""sequence_length""",
    )
    parser.add_argument(
        "--upper_limit",
        action="store",
        dest="upper_limit",
        default="45",
        required=True,
        help="""upper limit of cycle""",
    )
    parser.add_argument(
        "--lower_limit",
        action="store",
        dest="lower_limit",
        default="15",
        required=True,
        help="""lower limit of cycle""",
    )
    args = parser.parse_args()
    raw_train_data = args.raw_train_data
    common_letter_numeric = args.common_letter_numeric
    numeric_features = args.numeric_features.split(',')
    meta_columns = args.meta_columns.split(',')
    UPPER_LIMIT = int(args.upper_limit)
    LOWER_LIMIT = int(args.lower_limit)
    if numeric_features[0] == 'garbage999':
        numeric_features = []
    sequence_length = int(args.sequence_length)
    #########parser arguments#############

    ### LOAD TRAIN ###
    train_df = pd.read_csv(raw_train_data)

    print(train_df.shape)

    print("medium working time:", train_df.id.value_counts().mean())
    print("max working time:", train_df.id.value_counts().max())
    print("min working time:", train_df.id.value_counts().min())

    ### CALCULATE RUL TRAIN ###
    train_df['RUL'] = train_df.groupby(
        ['id'])['cycle'].transform(max) - train_df['cycle']
    # train_df.to_csv('test.csv')

    ### ADD NEW LABEL TRAIN ###
    train_df['label1'] = np.where(train_df['RUL'] <= UPPER_LIMIT, 1, 0)
    train_df['label2'] = train_df['label1']
    train_df.loc[train_df['RUL'] <= LOWER_LIMIT, 'label2'] = 2

    ### SCALE TRAIN DATA ###
    for col in train_df.columns:
        if col[0] == common_letter_numeric:
            print(col)
            numeric_features.append(col)

    scaler = StandardScaler()
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    train_df = train_df.loc[:, train_df.apply(pd.Series.nunique) != 1]
    train_df.head()
    print(train_df.columns)
    numeric_cols = train_df.columns

    ### SEQUENCE COL: COLUMNS TO CONSIDER ###
    sequence_cols = []
    for col in train_df.columns:
        if col[0] == 's':
            sequence_cols.append(col)
    print(sequence_cols)

    ### GENERATE X TRAIN TEST ###
    x_train = []
    for engine_id in train_df['id'].unique():
        for sequence in gen_sequence(
                train_df[train_df.id == engine_id], sequence_length, sequence_cols):
            x_train.append(sequence)

    x_train = np.asarray(x_train)
    x_shape = x_train.shape[2]
    shape_data = pd.DataFrame(columns=['shape', 'cols'])
    shape_data['cols'] = numeric_features
    shape_data.loc[0, 'shape'] = x_shape
    shape_data.loc[0:1, 'meta_cols'] = meta_columns
    shape_data.loc[0, 'sequence_length'] = sequence_length
    shape_data.loc[0, 'cycles'] = UPPER_LIMIT
    shape_data.loc[1, 'cycles'] = LOWER_LIMIT

    shape_data.to_csv(cnvrg_workdir + "/shape_data.csv", index=False)

    print("X_Train shape:", x_train.shape)

    TRAIN_FILE = 'x_train'
    reshape_save_array(x_train, TRAIN_FILE)

    ### GENERATE Y TRAIN TEST ###
    y_train = []
    for engine_id in train_df['id'].unique():
        for label in gen_labels(
                train_df[train_df.id == engine_id], sequence_length, ['label2']):
            y_train.append(label)

    y_train = np.asarray(y_train).reshape(-1, 1)

    print("y_train shape:", y_train.shape)

    ### ENCODE LABEL ###
    y_train = tf.keras.utils.to_categorical(y_train)
    print(y_train.shape)

    TRAIN_FILE = 'y_train'
    save_array(y_train, TRAIN_FILE)
