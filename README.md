# Welcome to End to End Loan Prediction Project



Here is the welcome page of the Loan Prediction project:

<img src="https://raw.githubusercontent.com/pooja-singh702/End-to-End-LoanPrediction/main/welcome.svg" width="100%" />





## Loan Eligibility Form

You can fill out the form here: [Loan Eligibility Form](http://127.0.0.1:5002/Predictdata)











## Overview: 
```
Developed an End-to-End loan prediction application using a simple but challenging dataset. The goal was not just to build pipeline or predict
loan approvals but also to handle the intricacies of imbalanced data while focusing
on reducing false positives— a critical aspect in real-world financial decision-making,
where defaults should not be incorrectly classified as non-defaults integrating machine learning,
data preprocessing, and a user-friendly interface focusing on improving data quality and leveraging statistical analysis like Hypothesis Testing for insights. 

```

## Challenge
```
The dataset was chosen to specifically tackle the challenge of working with an imbalanced dataset, where the focus was more on reducing false positives then on accuracy. The goal was not only to address the imbalance but also to implement a full end-to-end pipeline and deploy the model via a Flask-based web application. The challenge of false positives was especially critical in ensuring the financial risk to lending institutions was minimized, while simultaneously offering a user-friendly application for predictions.

```
## Dataset
Overview
```
The dataset contains information about loan applicants and their respective loan status. It includes both demographic(Gender, Marital Status ) and financial features(Credit History, LoanAmount), as well as the target variable(Loan Status) for prediction. The dataset is imbalanced, with more "Y" (approved loans) than "N" (rejected loans), which can lead to a Classification model predicting "Y" more frequently. This results in false positives, where rejected loans are incorrectly classified as approved.
```


## Key Features
 
1. [End-to-End-Pipeline](#End-to-End-Pipeline) 
       -[Creating-Conda-Enviroment](#Creating-Conda-Enviroment)
       -[Set-up-requirememnts.txt-Dependencies](#Set-up-requirememnts.txt)
       -[Install-setuptools-setup.py](#Install-setuptools-setup.py)
       -[Create-Custom-Logger](#Create-Custom-Logger)
       -[Common-methods-Script:utils.py](#Common-methods-Script:utils.py)
       -[Project-Structure-Flowchart](#Project-Structure-Flowchart)
       -[The-Data-Ingestion-Pipeline](#The-Data-Ingestion-Pipeline)
       -[The-Data-Transformation-Pipeline](#[The-Data-Transformation-Pipeline)
       -[Model-Trainer](#Model-Trainer)
       -[prediction-pipeline](#prediction-pipeline)
       -[Web-Deployment-using-Flask](#Flask-Based-Deployment)
2. [Understanding-the-Data](#Understanding-the-Data)
3. [Checked-Multicollinaerity](#Checked-ulticollinaerity)
4. [Feature-Engineering-&-Hypothesis-Testing](#[Feature-Engineering-&-Hypothesis-Testing)
5. [Handling-Outlier](#Handling-Outlier)
6. [Handling-Imbalance](#Handling-Imbalance)
7. [Handling-Imbalanced-Data](#Handling-Imbalanced-Data)
8. [Machine-Learning-Models](#Machine-Learning-Models)
9. [Focus-on-Minimizing-False-Positives](#Focus-on-Minimizing-False-Positives)
10. [User-Friendly-Web-Application](#User-Friendly-Web-Application)
11. [Flask-Based-Deployment](#Flask-Based-Deployment)
         -[Run-the-Flask](#Run-the-Flask)
         -[Problems-Faced-on-Running](#Problems-Faced-on-Running)


## End-to-End-Pipeline
Implementation of a comprehensive data pipeline, covering everything from data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation, to ensure the model's readiness for deployment.

## Project-Structure-Flowchart:
   
For a visual representation of the project's folder structure, check out the following [flowchart](Flowchart2.html):Clink the link: http://127.0.0.1:5501/flowchart2.html.

## Creating-Conda-Enviroment:
```
Set up a Conda Environment
Used Conda to manage the  dependencies.
First, create a new Conda environment with the required Python version ( Python 3.8):
conda create env python=3.8 -y
```
## Set-up-requirememnts.txt-Dependencies:
```
  
Install the required dependencies listed in the requirements.txt file by running.
This will install all the necessary libraries for the project, such as Flask, scikit-learn, pandas, and others.
pip install -r requirements.txt
```

## Install setuptools set.py
```
To ensure the proper setup of the application, make sure setuptools is installed.This ensures that anysetup-related tasks, such as packaging or installation of the application, will work as expected.
pip install setuptools.
```   

## Create-Custom-Logger:
```
    The application uses a custom logging setup to capture all critical events and errors. Logs are saved in the logs/ folder, and each time the application runs, a new log file is created with a timestamped filename (e.g., logfile_2024-12-09.log).
    Log Format:
    [YYYY-MM-DD HH:MM:SS,SSS] LineNumber loggerName - LogLevel - Message
    Log Folder and File Creation
    The logs/ folder is automatically created if it does not exist.
    Each time the application is run, a new log file is generated to ensure that logs from different runs are kept separate.
    Errors and warnings are logged using the appropriate log levels, such as ERROR and WARNING.
```
## Common-methods-Script:utils.py:
```
This script contains commonly used methods like save_obj() and evaluate_models()
```

## Structure/Flow  of Pipelines are as follows:

## The Data Ingestion Pipeline:  
```
Overview:
This pipeline prepares the data for the next stages of transformation and model training.
The Data Ingestion component handles the raw dataset, creating paths for storing data and 
performing data splitting.

It is structured into two main classes:

1.  DataIngestionConfig Class:
      -This class Defines paths for storing datasets in the artifacts folder.
      -Three paths are configured here:
           -Train Data Path: Location for training data.
           -Test Data Path: Location for test data.
           -Raw Data Path: Location for the initial raw data.
2.  DataIngestion Class:
      - Configuration Method: Uses DataIngestionConfig to set the storage paths and create the necessary directories.
      - This class uses GetdataIngestion() method to :
            -1. Creates directories and stores the raw data as raw_data.csv.
            -2. Reads the raw data into a DataFrame.
            -3. Splits the data into training and test datasets.
            -4. Saves the train and test datasets as train.csv and test.csv in the artifacts folder.
3.  Result:
      - Raw Data: Stored as raw_data.csv.
      - Train Data: Stored as train.csv.
      - Test Data: Stored as test.csv.

```

## The Data Transformation Pipeline: 
Overview:
```
Methods and class used:
    Class:
      1. Class DataTransformationConfig:
          -This class Defines paths for storing procesed dataframe in the artifacts folder.
          
      2. Class DataTransformation:
        - Configuration Method: Uses DataIngestionConfig to set the storage paths and create the necessary        directories.
        - Class uses its get_transformation() method  defined to prepare pipelines seperately for numerical and catgorical columns and then returns combined piipeline as preprocessing obj. 
        - Getclassmethod() defines Function transformer for dropping unecesary columns and also for outliers. It
          defines outlier method to remove them and then uses it as fumction transformer. 
        - This method then defines pipeline seperately for numerical and categorical columns
        - Makes the list of numerical col and catgorical col seperately.
        - Finally return combined pipeline as preprocessing object used by other method initiate_data_transformation(train_path, test_path) for tranforming train data.

        - initiate_data_transformation(train_path, test_path): DataTransformation class uses initiate_data_transformation(train_path, test_path) to initiate the  process  of transformation. It calls get_transformation() method to use its preprocessing obj on train data.
        - Before applying transformation, it splits train and test array data set into 4 parts: feature train/test data and target train/test data. 
        -Transformation applied only to feature train and test data 
        -Combines each with their respective target columns resulting as test arr and train arr
        - saves the preprocessor obj and returns train arr and test arr
    Overall Flow Diag:

    [Function called from data ingetion] → DataTransformation() → initiate_data_transformation()
                                                               Reads train test data
                                                                       ↓
                                                              Splits train/ test into features and target train test data frames
                                                                      ↓
                                                              [get_transformation()]
                                                                      ↓
                                                             [ColumnTransformer(preprocessor obj)] → Combined Pipeline
                                                                     ↓
                                                             Acts on feature train df(fit/transform) and feature test df(transform)
                                                                     ↓
                                                            results them into array
                                                                     ↓
                                                            Combines train faeture array with its train target col as array
                                                                     ↓
                                                             Combines test faeture array with its test target column as array
                                                                     ↓
                                                             Returns combined array as transformed train arr and test arr
                                                                     ↓
                                                            Saves preprocessor obj at artifacts

```
## Model Trainer
Overview
```
  List of classes and methods:
  Classes
    1. Class: ModelTrainerConfig trained_model_file_path: Defines the file path to save the trained model (model.pkl).
    2. Class: ModelTrainer: 
       1. Initializes the configuration object (ModelTrainerConfig).
       2. Uses its method Initiate_model_trainer(train_array, test_array):
          This is the  method that splits the processed data(train array and test array), contains list of classisfication models, uses evaluate_models() method defined at utils.py to train and evaluate the models.
       3. Function: evaluate_models(X_train, y_train, X_test, y_test, models, params) (defined @ src.utils):
          This function evaluates different models using GridSearchCV for hyperparameter tuning and cross-validation, calculates performance metrics, and returns a report along with the best model.
       4. Function: save_object(file_path, obj) defined in src.utils): This function is used to save the trained    model to the specified file path using pickle.
```
## prediction-Pipeline:
```
The prediction pipeline is a crucial component used by the Flask app.py to make predictions with the trained classification model. This pipeline connects the front-end web application to the back-end logic, allowing users to input data and receive real-time predictions.

Components of the Prediction Pipeline:

- Classes:
  - CustomData Class: Initializes the input data received from the user through the HTML form.
  - PredictionPipeline Class: Handles loading the trained model and preprocessor, transforming the input data, and making predictions.
Process Flow:

- Input Data Initialization:
The input data from the HTML form is retrieved using request.form.get() and initialized using the CustomData class.
The initialized data is then converted into a pandas DataFrame.
- Data Transformation and Prediction:
The PredictionPipeline class is called, which uses its predict() method.
It first loads the trained model.pkl and the preprocessing object (pipeline).
The input data is transformed using the same preprocessing pipeline applied during training.
The model (model.pkl) is then used to make predictions based on the transformed data.
- Important Notes:
Ensure the data types and column sequence of the new user input match those of the training data.
The same transformation pipeline (for numerical and categorical columns) is applied to ensure consistent processing and accurate predictions.

```

## Understanding-the-Data
Overview:
```
The first step in the analysis was to thoroughly understand the raw data, which involved investigating various aspects such as missing values, numerical variables, and distribution patterns. Visualizations were used to explore data distributions, relationships between features, and the target variable. Different types of charts helped uncover patterns and identify areas for improvement in data preprocessing.
  - 1. Exploratory Data Analysis (EDA): Reviewed the raw data for missing values and basic statistics using methods  like .describe() and .isnull(). The data was then split into numerical and categorical features to enable focused analysis.
  -2. For numerical data:
      I examined the distribution of variables using histograms and summary statistics with .describe().
      A correlation heatmap (Pearson & Spearman) helped identify relationships between numerical features.
      Outliers were detected using box plots and Z-scores to ensure clean data for modeling.
  -3. For categorical data:
      Conducted univariate analysis using bar charts and count plots to understand feature distributions.
      Bivariate analysis was done through cross-tabulation and stacked bar plots to explore feature interactions.
      Hypothesis testing (Chi-Square tests) was applied to analyze the relationships between categorical features and the target variable.
  -4. Feature-Target Relationship:
      I used hypothesis testing to investigate how each feature relates to the target variable (both categorical and numerical). This process provided critical insights and prepared the data for modeling.


```

## Checked-Multicollinaerity:
Overview:
```
The Spearman correlation showed a moderate non-linear correlation between LoanAmount and ApplicantIncome. Spearman’s rank correlation measures the monotonic relationship between variables, which means it captures non-linear patter. This potential multicollinearity could indicate that both features are providing overlapping information. Keeping both features in the model could lead to redundancy, which may degrade model performance by introducing instability or overfitting.

```

## Handling Imbalanced Data:
Overview:
```
Techniques to overcome Imbalance**
1. Hypertuning Methods: Adjusting class weights, estimators, during model training.
2. Stratified K-Fold Cross-Validation: It ensures that the proportion of each class is maintained in every fold, which helps provide a more balanced evaluation.This technique prevents overfitting to the majority class by ensuring the model performs well on unseen data.
3. Robust Models: Using algorithms that are robust to imbalance (like Random Forest or XGBoost).
4. Adjusting-Classification-Thresold-ROCCurve

```
## User Friendly Web Application:
Overview:
```
Development of an intuitive Flask-based web interface, where users can easily input their loan details and receive predictions on loan approval status. The application is designed for simplicity and ease of use, providing users with an accessible way to interact with the model. 
```
## [Handling-Outlier]
```
Logarithms and Robust scaling Combined Method is used for handling outliers. For right skewed data, Logarithms woorks well and then robust scaling method is applied and as a result got the nearly normal distribution on applying to some columns.
```
## [Handling-Missing values]

```
Used impute method for both categorical and numerical.
```


```
## Focus-on-Minimizing-False-Positives:
Overview
```
Emphasis on minimizing false positives (incorrectly classifying non-defaulters as defaulters), which can be critical for financial institutions. Techniques such as adjusting the ROC curve threshold and optimizing model evaluation metrics for minimizing false positives were employed.


## Feature-Engineering-&-Hypothesis-Testing:
Overview:
```
Use of feature engineering methods to improve the model's performance by transforming and selecting important features. Statistical tests like hypothesis testing were used to explore the relationships between features, ensuring a more refined model. 

```
## Machine-Learning-Models:
Overview:
```
In this section, multiple machine learning models were tested using different parameters, and stratified cross-validation was applied to handle class imbalance. The goal was to build a model that can predict loan approval with minimal false positives.
Differerent Models with paremeters used with straitfied cross validation.
1. logistic Regression
2. Random Forest
3. SVC
4. Gradient Boosting


```  
 
## Flask-Based-Deployment(Front-end)

The trained model was deployed as a fully functional web application using Flask. The front-end was developed with Flask, HTML, and CSS, creating a simple and user-friendly interface for interactions. Flask powers the backend, processing user input, making predictions, and returning real-time results. The front-end design ensures a smooth and responsive experience, making the application accessible even to non-technical users. The model is seamlessly integrated into the Flask backend, where user inputs are processed, predictions are made, and results are displayed effortlessly.

Main idea here is Flask Integration for User Input and Prediction:
The main idea behind this implementation is to gather user inputs through the web page using a Flask route. Here's how it works:

   - User Input through Web Page:Users input data via a web form (HTML page). Flask routes handle the  submission and use the POST method to collect the data.

   - Data Handling:The CustomData function is used to initialize and process the input data. The received data is then converted into a pandas DataFrame, ensuring it is in the same structure expected by the model.
   
   - Data Transformation and Prediction:The DataFrame is passed to the predict() method, which uses the preprocessed object (pipeline) to transform the new data. The transformation applies the same preprocessing steps (e.g., handling numerical and categorical features) as during model training.
  
   - Prediction:The transformed data is then fed into the saved model to generate predictions.
    The output is returned to the user through the web page. This process ensures that the new user input is handled consistently with the model's expected input format, and predictions are made accordingly

## Run-the-Flask:

 **Run the Flask Application**
  ```
   Once the dependencies are installed,  run the application using the following command:
   python app.py
   This will start the Flask development server.

  ```

6. **Access the Application**
  ```
  You can access the application by opening your web browser and navigating to:
  
    http://127.0.0.1:5002/Predictdata
  ```
7. **Test the Application**
  ```
  Once the application is running, open your browser and go to http://127.0.0.1:5002. This is index html file welcome page. By adding /Predictdata (http://127.0.0.1:5002/Predictdata) loan prediction form opens. You can input financial and demographic data into the form and click "Predict" to get a prediction on loan default.


## Problems-Faced-on-Running-Flask for User Predictions:
Important Points for Running and Deploying the Model with Flask
```
1. Consistency in Data Types and Sequence:
Ensure that the data types and sequence of features used during training are consistent when receiving new input data from the user.
If you used a particular pipeline for transforming the training data (like scaling numerical features or one-hot encoding categorical features separately), ensure the same transformations are applied to new input data using the same preprocessing pipeline categorisin data as per used list(build during transformation for model training) of num and catg columns. 

2. Handling Categorical and Numerical Features:
In the model, categorical and numerical columns should be handled separately, using appropriate transformations (e.g., one-hot encoding for categorical and scaling for numerical).
When new data is received via the Flask form, the categorical features should be encoded and numerical features should be scaled using the same transformation pipeline that was applied during training.

3. Feature Ordering:
The feature order in the new input data must match the order used during training. Any changes to the feature sequence might result in incorrect predictions.
Ensure the sequence of columns in the user inputs aligns exactly with how they were ordered during the training process. This means, predict_pipeline, app.py to get user input, html format should have same sequence of columns as transformed data used for model training.

4. Saving the Preprocessing Object:
The preprocessing object (which defines how data should be transformed) should be saved and loaded during both the training and prediction stages. This ensures the same transformations are applied at both ends (training and prediction).
This can be done using a function like save_object, where both the preprocessing pipeline and the model are saved to disk.

5. Data Input from the User (Form Handling in Flask):
The Flask app should collect data from the HTML form using request.form.get(). The user inputs should be initialized as a custom data structure (e.g., the CustomData class), then converted into a DataFrame.
The data frame should be transformed (using the same transformation pipeline) before passing it to the trained model for prediction.

6. Prediction Process:
The predict() method in the Flask backend should load the saved model and preprocessing pipeline.
The user input data is passed through the transformation pipeline to ensure consistency in how the model received the data during training.
The transformed data is then passed to the model to make predictions and return the results to the user.

7. Potential Issues and Considerations:
   - Missing values: If new input data contains missing values, the preprocessing pipeline should handle them appropriately (e.g., through imputation).
   - Incorrect data types: If a feature that was treated as categorical in the training data is passed as a numerical value (or vice versa), it may lead to incorrect predictions or errors during transformation.
   - Ensure that categorical values (like Credit_History) are handled properly. If they were represented as strings during training, the new data should be provided in the same format (either as strings or numeric values).
   - Testing:
    Before deploying the model, make sure to test with different types of input to ensure the app handles various edge cases correctly.
    Ensure that the model performs well with new, unseen data by validating the transformation pipeline and ensuring the model can handle all the transformations without errors.
-   User Input Validation:
    Always validate user input on both the front-end (in HTML forms) and the back-end (in Flask) to ensure data integrity.
    Provide clear error messages for invalid input data to help users understand what needs to be corrected.
  - Performance Considerations: If the model or transformation pipeline is large, loading them repeatedly on every prediction might slow down the app. It’s good practice to cache the loaded model and preprocessing object if possible.
  - Ensure that the Flask app can handle multiple simultaneous requests (e.g., by using production-ready servers like Gunicorn or uWSGI).
  ```
  

```




