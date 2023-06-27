import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.helper import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            text_feature = "text"
            target_feature = "emotions"

            text_pipeline = Pipeline(
                steps = [
                    ("clean_text",clean_text()),
                    ("tfidf",TfidfVectorizer())
                ]
                )
            target_pipeline = Pipeline(
                [("label_encoder",LabelEncoder())]
                )
            preprocessor = ColumnTransformer(
                [("text_pipeline",text_pipeline,text_feature),
                ("target_pipeline",target_pipeline,target_feature)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException( e, sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data compleated")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_obj()
            target_column_name = "emotions"
            text_feature_name = "text"

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            train_arr = np.array(input_feature_train_arr)
            test_arr = np.array(input_feature_test_arr)

            logging.info("Saved preprocessing object")

            save_object(self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

            