##This is used to connct the web app developed to predict the features and at back, predict pipeline get daya from web page and its data get initialised
## then we load pickl files model and preprocessor.pkl to process an dpredict data at web page.

import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            print(data_scaled)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        ApplicantIncome: int,
        CoapplicantIncome: float,
        Loan_Amount_Term: float,
        Gender: str,
        Married: str,
        Dependents: str,
        Education: str,
        Self_Employed: str,
        Credit_History: float,
        Property_Area: str
        ):

        self.ApplicantIncome= ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
     
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Gender = Gender

        self.Married = Married

        self.Dependents = Dependents
        self.Education = Education

        self.Self_Employed = Self_Employed
        
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome], 
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area]
                
                
            }
            a = pd.DataFrame(custom_data_input_dict)
            print(a)

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)