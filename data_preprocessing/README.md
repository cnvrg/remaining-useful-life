# Remaining Useful Life  (Data Preprocessing)
This library performs data preprocessing on tabular data and makes it ready to be used by the classification algorithms. Roughly, the steps include missing value treatment, encoding, evaluation of the structure//type of data and scaling. Below are the relevant features of the library

### Features
- perform missing value treatment for each type of column in the data
- Perform encoding of character values
- add labels for each cycle limit
- scale data
- generate arrays according to the sequence length
- reshape 3-D array to 2-D array

# Input Arguments
- `--x_train` raw training data provided by the user
- `--common_letter_numeric` Default=(s) -- common letter if all the sensors or settings start from the same letter
    (eg settings1,setting2,sensor1,sendor2,sensor3...)
- `--numeric_features` Default=(garbage999) -- list of numeric features in the data to be scaled or labelled
- `--meta_columns` Default=(id,cycle) columns other than numeric columns such as ID and cycle
- `--sequence_length` Default=(50) In order to have at our disposal the maximum number of data for the train, we split the series with a fixed window and a sliding of 1 step. For example, engine1 have 192 cycles in train, with a window length equal to 50 we extract 142 time series with length 50
- `--upper_limit` Default=(25) -- upper limit to number of cycles
- `--lower_limit` Default=(25) -- lower limit to number of cycles
..

# Model Artifacts
- `--x_train` 2-D array for x-train data
- `--y_train`  2-D array for y-train data
- `--shape_data` refers to the name of the file which contains the the custom-trained cnn model


