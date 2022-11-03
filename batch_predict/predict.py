'''Batch predict'''
import warnings
import argparse
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
warnings.filterwarnings(action='ignore')
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

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


if __name__ == '__main__':
    
    #########parser arguments#############
    parser = argparse.ArgumentParser(description="""Preprocessor""")
    parser.add_argument(
        "--x_test",
        action="store",
        dest="x_test",
        default="/input/s3_connector/remaining_useful_life_data/raw_test_data.csv",
        required=True,
        help="""x_test""",
    )
    parser.add_argument(
        "--cnn",
        action="store",
        dest="cnn",
        default="/input/cnn/cnn_model.h5",
        required=True,
        help="""cnn""",
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
    test_data = pd.read_csv(args.x_test)
    cnn = args.cnn
    cols = pd.read_csv(args.shape_data)
    numeric_features = cols['cols'].tolist()
    meta_columns = cols['meta_cols'].tolist()
    upper_limit = int(cols.loc[0, 'cycles'])
    lower_limit = int(cols.loc[1, 'cycles'])
    seq_length = int(cols.loc[0,'sequence_length'])
    #########parser arguments#############

    ### SCALE TEST DATA ###
    test_data, sequence_cols = scaling_data(
        test_data, numeric_features, meta_columns)
    result_df = pd.DataFrame()
    for i in test_data.id.unique():
        # for each id in test data
        try:
            x_test = test_data[test_data['id'] == i]
            x_test.reset_index(inplace=True)

            data = data_preprocessing(x_test, sequence_cols, seq_length)
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
            temp_df = cnn_predicted_df.tail(1)
            if result == '>{} Cycles'.format(upper_limit):
                j = 'More than {} cycles left'.format(upper_limit)
            elif result == '{}-{} Cycles'.format(lower_limit,upper_limit):
                j = '{} to {} cycles left'.format(lower_limit,upper_limit)
            elif result == '<{} Cycles'.format(lower_limit):
                j = 'Less than {} cycles left'.format(lower_limit)
            #dic = {'id': i, 'prediction': j}
            temp_df['ID'] = i
            temp_df['Prediction'] = j
            print(temp_df)
        except BaseException:
            print('id-{} has less rows than the defined sequence length'.format(i))
            temp_df = pd.DataFrame()
            temp_df.loc[0,'>{} Cycles'.format(upper_limit)] = np.nan
            temp_df.loc[0,'{}-{} Cycles'.format(lower_limit,upper_limit)] = np.nan
            temp_df.loc[0,'<{} Cycles'.format(lower_limit)] = np.nan
            temp_df.loc[0,'ID'] = i
            temp_df.loc[0,'Prediction'] = 'id-{} has less rows than the defined sequence length'.format(i)
            print(temp_df)
        result_df = result_df.append(temp_df)
    
    result_df = result_df[['ID','Prediction','>{} Cycles'.format(upper_limit),'{}-{} Cycles'.format(lower_limit,upper_limit),'<{} Cycles'.format(lower_limit)]]
    result_df.to_csv(cnvrg_workdir + '/predicted_data.csv', index=False)
