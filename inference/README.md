# Remaining Useful Life  (Batch Predict)
Predict rare events is becoming an important topic of research and development in a lot of Artificial Intelligent solutions. Given these scenarios, we can imagine a rare event as a particular state that occurs under specific conditions, divergent from normal behaviors, but which plays a key role in terms of economic interest.
We’ve developed a Machine Learning solution to predict the Remaining Useful Life (RUL) of a particular engine component. This kind of problem plays a key role in the field of Predictive Maintenance, where the purpose is to say ‘How much time is left before the next fault?’. To achieve this target I developed a Convolutional NN in Keras that deals with time series in the form of images.
### Features
- Upload dataset containing data of a machine's settings and sensors
- User can choose from multiple arguments to train the model as per their liking

# Input Arguments
- `--x_test` raw test dataset from the user
- `--cnn` pre-trained 2-D CNN model
- `--shape_data` dataframe containing mutiple variables to be used in batch predict from data preprocessing block
..

# Model Artifacts
- JSON Response
```
{"24":"less than 15 cycles left"}

