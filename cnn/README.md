# Remaining Useful Life  (Convolutional Neural Network)
We’ve adopted a classical 2D CNN architecture
### Features
- Takes in pre-processed training data from the data-preprocessing block and applies a 2-D CNN model by converting train data into images and fitting the same to the model 
- User can choose from multiple arguments to train the model as per their liking such as epochs and batch-size

# Input Arguments
- `--x_train` pre-processed training data in form of 2-D array
- `--y_train` pre-processed test data in form of 2-D array
- `--batch_size` Default=(512) -- batch-size to be used in the cnn model
- `--epochs` Default=(25) -- number of epochs to be used in the cnn model
- `--shape_data` dataframe containing mutiple variables to be used in batch predict from data preprocessing block
..

# Model Artifacts
- `--cnn_model.h5` refers to the name of the file which contains mutiple variables to be used in batch predict

