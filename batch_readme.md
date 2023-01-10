Use this blueprint to run a pretrained convolutional neural network (CNN) model with your customized dataset to predict when a company asset is likely to fail within given cycles. To clean and validate the data on which to further train the model, provide one folder in the S3 Connector containing your raw training data.

Complete the following steps to run this RUL-predictor blueprint in batch mode:
1. Click **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` − Value: provide the data bucket name
     - Key: `prefix` − Value: provide the main path to the images folders
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Click the **Batch-Predict** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `--x_test` – Value: provide the path to the test dataset in the following format: `/input/s3_connector/remaining_useful_life_data/raw_test_data.csv`
     - Key: `--cnn` – Value: provide the path to trained model in the following format: `/input/ s3_connector/remaining_useful_life_data/cnn_model.h5`
     - Key: `--shape_data` – Value: provide the shape dataframe path in the following format: `/input/ s3_connector/remaining_useful_life_data/shape_data.csv`

     NOTE: You can use prebuilt data example paths provided.
     
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software deploys a RUL-predictor model that predicts the number of cycles until an asset fails.
5. Select **Batch Predict > Experiments > Artifacts** and locate the output CSV file.
6. Select the **predicted_data.csv** File Name, click the right Menu icon, and click **Open File** to view the output CSV file.

A custom model that can predict the number of cycles until an asset fails has now been deployed in batch mode. To learn how this blueprint was created, click [here](https://github.com/cnvrg/remaining-useful-life).
