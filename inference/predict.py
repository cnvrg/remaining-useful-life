'''Endpoint'''
# pylint: disable=E0611
# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=C0411
# pylint: disable=C0415
# pylint: disable=C0301

#import base64
import requests
import pathlib
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings(action='ignore')

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


FILES = ['endpoint_test_data.csv', 'cnn_model_1.h5']

BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/cnvrg/remaining_useful_life_data/"


def download_model_files():
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


download_model_files()


def data_preprocessing(x_test_data):
    '''

    Parameters
    ----------
    x_test_data : test data

    Returns
    -------
    x_test_data : preprocessed data

    '''

    test_df = x_test_data
    print(test_df.shape)
    test_df.head(3)
    print(test_df['s6'])

    id_col = test_df.loc[0, 'id']
    ### SCALE TEST DATA ###
    numeric_features = []
    for col in test_df.columns:
        if col[0] == 's':
            numeric_features.append(col)
    scaler = StandardScaler()
    test_df[numeric_features] = scaler.fit_transform(test_df[numeric_features])
    #sensor_6 = test_df[['s6','s10']]
    test_df = test_df[[i for i in test_df if len(set(test_df[i])) > 1]]
    #test_df[['s6','s10']] = sensor_6
#    test_df = test_df.loc[:,test_df.apply(pd.Series.nunique) != 1]
    print(test_df.head())
    print('===================================')
    print(test_df.columns)
    sequence_length = 50
    test_df['id'] = id_col

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

    ### SEQUENCE COL: COLUMNS TO CONSIDER ###
    sequence_cols = []
    for col in test_df.columns:
        if col[0] == 's':
            sequence_cols.append(col)
    print(sequence_cols)

    ### GENERATE X TRAIN TEST ###
    test_data = []
    for engine_id in test_df['id'].unique():
        for sequence in gen_sequence(
                test_df[test_df.id == engine_id], sequence_length, sequence_cols):
            test_data.append(sequence)

    test_data = np.asarray(test_data)

    return test_data


########## Path changing (whether the input comes from training or non-tra
if os.path.exists("/input/cnn/cnn_model.h5"):
    print('Running Training Inference')
    cnn = "/input/cnn/cnn_model.h5"
    shape_data = pd.read_csv("/input/data_preprocessing/shape_data.csv")

else:
    print('Running Stand Alone Endpoint')
    script_dir = pathlib.Path(__file__).parent.resolve()
    #x_test = pd.read_csv(os.path.join(script_dir,'raw_test_data.csv'))
    cnn = os.path.join(script_dir, 'cnn_model.h5')


def predict(data):
    '''
    Parameters
    ----------
    data : raw data from user

    Returns
    -------
    JSON file with predictions for remaining life of the machine

    '''
    print(data)
    x_test = data
    #df = base64.b64decode(data)
    #df = df.decode('utf-8').splitlines()
    #df = json.loads(json.dumps(data))
    #x_test = pd.read_csv(BytesIO(df))
    x_test = data_preprocessing(x_test)
    print(x_test.shape)

    def rec_plot(s, eps=0.10, steps=10):
        d = pdist(s[:, None])
        d = np.floor(d / eps)
        d[d > steps] = steps
        Z = squareform(d)
        return Z

    x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')

    print(x_test_img.shape)

    from tensorflow.keras.models import load_model
    # load model
    model = load_model(cnn)

    model.predict(x_test_img)
    print(model.summary())
    cnn_predicted = model.predict(x_test_img)
    cnn_predicted_df = pd.DataFrame(cnn_predicted)
    result = cnn_predicted_df.iloc[-1, :]
    #response = cnn_predicted_df.to_json(orient='columns')
    response = result.to_json(orient='columns')
    print(response)
    return response
    # cnn_predicted_df.to_csv(cnvrg_workdir+"/cnn_predicted_data.csv",
    # index=False) = "x_test"
