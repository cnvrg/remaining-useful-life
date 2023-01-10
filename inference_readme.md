Use this blueprint to immediately predict when a machine is likely to fail within given cycles using your customized dataset. To use this pretrained RUL-predictor model, create a ready-to-use API-endpoint that can be quickly integrated with your data and application.

This inference blueprint’s model was trained using the [NASA Turbofan Jet Engine Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps). To use custom data according to your specific business, run this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/rul-train), which trains the model and establishes an endpoint based on the newly trained model.

Complete the following steps to deploy an RUL-predictor API endpoint:
1. Click the **Use Blueprint** button.
2. In the dialog, select the relevant compute to deploy the API endpoint and click the **Start** button.
3. The cnvrg software redirects to your endpoint. Complete one or both of the following options:
   - Use the Try it Live section with any data to check your model's predictions.
   - Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

An API endpoint that can predict the number of cycles until a machine or equipment fails has now been deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/remaining-useful-life).
