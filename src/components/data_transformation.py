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
        self.data_transformation_config = DataTransformation()
    
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
            