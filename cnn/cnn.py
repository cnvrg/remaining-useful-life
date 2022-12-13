'''2-D Convolutional Neural Network'''
import warnings
import argparse
import os
import random
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import numpy as np
import pandas as pd
warnings.filterwarnings(action='ignore')
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

def reshape_load_data(arr):
        '''
        Parameters
        ----------
        arr : array
        Returns
        -------
        loaded_original : original array from data preprocessing
        '''
        # retrieving data from file.
        loaded_arr = np.loadtxt(arr)
        print(loaded_arr.shape)
        # This loaded_arr is a 2D array, therefore we need to convert it to the original array shape.
        # reshaping to get original matrice with original shape.
        loaded_original = loaded_arr.reshape(
            loaded_arr.shape[0],
            loaded_arr.shape[1] // x_shape,
            x_shape)

        return loaded_original

def rec_plot(s, eps=0.10, steps=10):
        '''
        recurrence plots
        '''
        d = pdist(s[:, None])
        d = np.floor(d / eps)
        d[d > steps] = steps
        Z = squareform(d)
        return Z

def load_data(file_name):
        '''
        Parameters
        ----------
        file_name : name of the array

        Returns
        -------
        loaded_arr : loaded array
        '''
        # retrieving data from file.
        loaded_arr = np.loadtxt(file_name)

        return loaded_arr  

def set_seed(seed):
        '''
        set seed
        '''
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        
if __name__ == '__main__':

    #########parser arguments#############
    parser = argparse.ArgumentParser(description="""Preprocessor""")
    parser.add_argument(
        "--x_train",
        action="store",
        dest="x_train",
        default="/input/data_preprocessing/x_train",
        required=True,
        help="""x_train""",
    )
    parser.add_argument(
        "--y_train",
        action="store",
        dest="y_train",
        default="/input/data_preprocessing/y_train",
        required=True,
        help="""y_train""",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        dest="batch_size",
        default="512",
        required=True,
        help="""cnn model fit arguments""",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        dest="epochs",
        default="25",
        required=True,
        help="""cnn model fit arguments""",
    )
    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        default="58",
        required=True,
        help="""set seed""",
    )    
    parser.add_argument(
        "--shape_data",
        action="store",
        dest="shape_data",
        default="/input/data_preprocessing/shape_data.csv",
        required=True,
        help="""shape_data""",
    )

    args = parser.parse_args()
    x_train = args.x_train
    y_train = args.y_train
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    shape_data = args.shape_data
    shape_data = pd.read_csv(shape_data)
    x_shape = int(shape_data.loc[0, 'shape'])
    sequence_length = int(shape_data.loc[0, 'sequence_length'])
    seed = int(args.seed)
    print(x_shape)
    #########parser arguments#############
    
    ### Load Arrays ##
    x_train = reshape_load_data(x_train)
    print(x_train.shape)
    y_train = load_data(y_train)
    
    ### TRANSFORM X TRAIN TEST IN IMAGES ###
    x_train_img = np.apply_along_axis(rec_plot, 1, x_train).astype('float16')
    print(x_train_img.shape)
    
    ####################MODEL####################
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
        sequence_length, sequence_length, x_shape)))  # ?????????
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    print(model.summary())

    set_seed(seed)

    es = EarlyStopping(
        monitor='val_accuracy',
        mode='auto',
        restore_best_weights=True,
        verbose=1,
        patience=6)

    model.fit(
        x_train_img,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=2)

    # save the model to disk
    filename = cnvrg_workdir + '/cnn_model.h5'
    model.save(filename)
    ####################MODEL####################
