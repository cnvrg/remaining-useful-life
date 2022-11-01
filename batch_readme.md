You can use this blueprint to clean and validate data in order to further train multiple models to predict results for if a customer is likely to churn or not using your own customized dataset. In order to clean the data you will be needed to provide: --raw_train_data raw data uploaded by the user on the platform --common_letter_numeric get the name of the id column by the user --label_encoding_cols list of columns to be label encoded by the user --scaler the common letter in all numeric variables --numeric_features list of numeric features --meta_columns columns other than the numeric features --sequence_length length of the sequence --upper_limit upper limit of cycles --lower_limit lower limit of cycles

You would need to provide 1 folder in s3 where you can keep your training data

remaining_useful_life: Folder containing the training data "raw_train_data.csv" and the test data file on which to predict, "raw_test_data.csv"
Directions for use:

Click on Use Blueprint button

You will be redirected to your blueprint flow page

In the flow, edit the following tasks to provide your data:

In the S3 Connector task:

Under the bucketname parameter provide the bucket name of the data
Under the prefix parameter provide the main path to where the input file is located
In the Data-Preprocessing task:

Under the raw_train_data parameter provide the path to the input folder including the prefix you provided in the S3 Connector, it should look like: /input/s3_connector/<prefix>/raw_train_data.csv
NOTE: You can use prebuilt data examples paths that are already provided

Click on the 'Run Flow' button
In a few minutes you will train a new rul model and deploy as a new API endpoint
Go to the 'Serving' tab in the project and look for your endpoint
You can use the Try it Live section with a data point similar to your input data (in terms of variables and data types) to check your model
You can also integrate your API with your code using the integration panel at the bottom of the page
Congrats! You have trained and deployed a custom model that detects the number of cycles under which the machine is going to fail