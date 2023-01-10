Use this blueprint to tailor-train a convolutional neural network (CNN) model with your customized dataset to predict when a company asset is likely to fail within given cycles.

To clean and validate the data on which to further train the model, provide one folder in the S3 Connector containing your raw training data. Inputs to further train the CNN model are placed in a remaining_useful_life folder containing the training data `raw_train_data.csv` file and the test data `raw_test_data.csv` file on which to make predictions. This blueprint also establishes an endpoint that can be used to predict RUL cycles based on the newly trained model.

Complete the following steps to train a RUL-predictor model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` - Value: enter the data bucket name
     - Key: `prefix` - Value: provide the main path to the CVS file folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Data Preprocessing** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `--raw_train_data` – Value: provide the path to the S3 location with the training data in the following format: `/input/s3_connector/remaining_useful_life_data/raw_train_data.csv`
     * Key: `--common_letter_numeric` – Value: provide the name of the ID column
     * Key: `--numeric_features` – Value: list the numeric features in the data to be scaled or labeled
     * Key: `--meta_columns` – Value: identify columns other than the numeric features such as ID and cycle
     * Key: `--sequence_length` – Value: enter the length of the sequence
     * Key: `--upper_limit` – Value: provide the upper limit of cycles
     * Key: `--lower_limit` – Value: provide the lower limit of cycles
     
     NOTE: You can use the prebuilt example data paths provided.

   * Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Return to the flow and click the **CNN (Train)** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `--x_train` – Value: provide the path to the preprocessed training data in the following format: `/input/data_preprocessing/x_train`
     * Key: `--y_train` – Value: provide the path to preprocessed test data in the following format: `/input/data_preprocessing/y_train`
     * Key: `--batch_size` – Value: provide the batch size to train the CNN model
     * Key: -`-epochs` – Value: provide the path the number of epochs to train the CNN model
     * Key: `--seed` – Value: provide the value to initialize the random number generator in the CNN model
     * Key: `--shape_data` – Value: provide the dataframe containing multiple variables to batch predict from data preprocessing block.
    
    NOTE: You can use the prebuilt example data paths provided. 

   * Click the **Advanced** tab to change resources to run the blueprint, as required.
 5. Return to the flow and click the **Batch Predict** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `--x_test` – Value: provide the path to the test dataset in the following format: /input/s3_connector/remaining_useful_life_data/raw_test_data.csv
     * Key: `--cnn` – Value: provide the path to trained model in the following format: /input/cnn/cnn_model.h5
     * Key: `--shape_data` – Value: provide the shape dataframe path in the following format: /input/data_preprocessing/shape_data.csv
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
 5.	Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained RUL-predictor model and deploying it as a new API endpoint.
6. Track the blueprint's real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
7. Click the **Serving** tab in the project and locate your endpoint.
8. Complete one or both of the following options:
   * Use the Try it Live section with any data to check the model's predictions.
   * Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

A custom model and API endpoint, which can predict the number of cycles until an asset fails, have now been trained and deployed. If also using the Batch Predict task, a trained RUL-predictor model that makes batch predictions has now been deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/remaining-useful-life).
