import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.helper import save_object
from src.components.model_training import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path :str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        
        try:
            or_df = pd.read_csv("/config/workspace/emotions/data/data.csv")
            df = or_df.sample(n = 1000,random_state = 42)
            logging.info("Read data as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train test split is initiated")
            train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            logging.info("Data ingestion is compleated")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException( e, sys)


if __name__ == "__main__":
    #data ingestion
    ingestion = DataIngestion()
    train_data,test_data = ingestion.initiate_data_ingestion()
    #data transformation
    datatransformation = DataTransformation()
    train_arr,test_arr= datatransformation.initiate_data_transformation(train_data,test_data)
    #model training
    modeltrainer = ModelTrainer()
    print("Accuracy Score and best model :",modeltrainer.initiate_model_trainer(train_arr, test_arr))
    
     

            