# get raw data
# seperate catg and num
# encoding
# train test split on transformed
# return transformed data

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


from src.utils import save_object

import sys
from dataclasses import dataclass


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import OneHotEncoder


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# In data_transformation.py



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
  

class DataTransformation:
    def __init__(self):
        # path of preprocessed data
        self.data_transformation_config=DataTransformationConfig()
         # get the path of train test data
    def get_transformation(self):
        # this will help to transform data and pipeline will be built
        try:


        #step1: columns to remove

                columns_to_drop= ["Loan_ID","State", "LoanAmount", "Zip"]
                drop_columns = FunctionTransformer(lambda df: df.drop(columns= columns_to_drop, errors = 'ignore'))

        # #step2: mapping loan status from yes no to 1, 0 in oth test train
        # target_feature_train_df =  target_feature_train_df.replace({'Y': 1, 'N': 0})
        # target_feature_test_df =  target_feature_test_df.replace({'Y': 1, 'N': 0})

      

        # step 3: # Updated numerical and categorical columns
                numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']
                cat_col_list = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']


                def convert_columns_to_int(X, numerical_cols):
                    df = pd.DataFrame(X, columns=numerical_cols)  # Ensure we're working with the right columns
                    for col in numerical_cols:
                        df[col] = df[col].astype(int)  # Convert the column to integers
                        return df.to_numpy()  # Convert back to NumPy array
                convert_to_int = FunctionTransformer(lambda X: convert_columns_to_int(X, numerical_cols), validate=False) 
                # Define the numeric transformer pipeline
               


                def impute_outliers(X, numerical_cols ):
                    df = pd.DataFrame(X, columns=numerical_cols)
                    for col in numerical_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        median_value = df[col].median()

                        df[col] = np.where(df[col] < lower_bound, median_value, df[col])
                        df[col] = np.where(df[col] > upper_bound, median_value, df[col])

                    return df.to_numpy()
        
           
         # Create a FunctionTransformer for outlier imputation
                outlier_imputer = FunctionTransformer(lambda df: impute_outliers(df, numerical_cols), validate=False)
       

    # Preprocessing for numerical data
                numeric_transformer = Pipeline(steps=[("drop_columns",  drop_columns),('mean_imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
                                                ('convert_to_int',convert_to_int), ('outlier_imputer', outlier_imputer) ])
# Preprocessing for categorical data
                categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])   



    # Combine both transformers
                preprocessor = ColumnTransformer(
                transformers=[('num', numeric_transformer, numerical_cols),
               ('cat', categorical_transformer, cat_col_list)])

                return preprocessor
         
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path,test_path):
        try:

            logging.info("Reading train , test data")
        
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

    


            logging.info("spliting data into fetaures and target from train and test")
            target_column_name= "Loan_Status"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

               # Mapping target 
            target_feature_train_df =  target_feature_train_df.replace({'Y': 1, 'N': 0})
            target_feature_test_df =  target_feature_test_df.replace({'Y': 1, 'N': 0})



        ## CALLING GET TRANSFORMATION
            logging.info("Obtaining preprocessing object")

        

            numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']
            cat_col_list = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

            preprocessor_obj=self.get_transformation()
    #Fit and transform your training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

# Transform the test data using the fitted preprocessor
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

# Convert back to DataFrame for easier interpretation
            num_feature_names = numerical_cols
            cat_feature_names = preprocessor_obj.named_transformers_['cat']['onehot'].get_feature_names_out(cat_col_list)

# Create the new DataFrame for training data
            X_train_processed_df = pd.DataFrame(input_feature_train_arr, columns=np.concatenate([num_feature_names, cat_feature_names]))

# Create the new DataFrame for testing data
            X_test_processed_df = pd.DataFrame(input_feature_test_arr, columns=np.concatenate([num_feature_names, cat_feature_names]))

# Check the shape and first few rows
            print("Training Data Shape:", X_train_processed_df.shape)
            print("Training Data Shape:", X_train_processed_df.columns)
            print("Training Data Sample:\n", X_train_processed_df.head())
            print("Testing Data Shape:", X_test_processed_df.shape)
            print("Testing Data Sample:\n", X_test_processed_df.head())
## saving preprocessor obj and processed data as array
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )

        except Exception as e:
            raise CustomException(e,sys)
     










