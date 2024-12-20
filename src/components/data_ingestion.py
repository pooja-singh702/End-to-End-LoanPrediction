
import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv") # artifacts = parent dir
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv") # locate at artifacts/data.csv



class DataIngestion:
    def __init__(self):
        # Path of raw data initialised here inside class
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Create the directory for saving the raw data CSV
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Read the dataset
            df = pd.read_csv("loan_prediction.csv")
            logging.info("Read the dataset")
            


            ##  # Step 1: Convert Credit_History to Object before splitting
            # cannot use astype int as na values are not considered
            df['Credit_History'] = df['Credit_History'].astype('object')
           
            
            ## Saving raw dataframe as csv at artifacts/data.csv

            logging.info("saving raw data at artifacts")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Splitting the data")
           # Split the data in only two parts: train set and test set
            train_set, test_set = train_test_split(df, test_size=0.2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transform = DataTransformation()
    train_arr, test_arr,_ = data_transform.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    accuracy = modeltrainer.initiate_model_trainer(train_arr, test_arr)


    

    