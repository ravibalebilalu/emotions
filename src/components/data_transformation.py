import os
import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
ps = SnowballStemmer(language = "english")
nltk.download("stopwords")
     

def clean_text( text):         
         
        result = re.sub("[^a-zA-Z]"," ", text)
         
        result = result.lower()
        result = result.split()
        result = [ps.stem(word) for word in result if word not in set(stopwords.words('english'))]
        result = " ".join(result)
         
        
        return result


@dataclass
class DataTransformationConfig:
    vectorizer_obj_file_path = os.path.join("artifacts","tfidfvectorizer.pkl")
    labelizer_obj_file_path = os.path.join("artifacts","labels.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test data compleated")

             
             
            target_column_name = "emotions"
            text_column_name = "text"

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]
              

            logging.info("Text cleaning initiated")
            input_feature_train_df["text"] = input_feature_train_df['text'].apply(clean_text)
            input_feature_test_df["text"] = input_feature_test_df['text'].apply(clean_text)
            logging.info("Text Cleaning compleated")

            logging.info("Text transformation initiated")
            tf = TfidfVectorizer(max_features = 500)
            tf.fit(input_feature_train_df["text"])
            vectorizer_obj = open(self.data_transformation_config.vectorizer_obj_file_path,"wb")
            pickle.dump(tf,vectorizer_obj)
            vectorizer_obj.close()

            

            train_transformed_arr = tf.transform(input_feature_train_df['text']).toarray()
            test_transformed_arr = tf.transform(input_feature_test_df['text']).toarray()
            logging.info("Text transformation compleated")
            
            logging.info("Label encoding for target column initialized")
            le = LabelEncoder()
            le.fit(target_feature_train_df)
            labelizer = open(self.data_transformation_config.labelizer_obj_file_path,"wb")
            pickle.dump(le, labelizer)
            labelizer.close()
            target_train_arr = le.transform(target_feature_train_df)
            target_test_arr = le.transform(target_feature_test_df)
            logging.info("Label encoding for target column is compleated")
            
            train_arr = np.c_[train_transformed_arr,target_train_arr]
            test_arr = np.c_[test_transformed_arr,target_test_arr]
            logging.info("train_arr and test_arr are ready")
                      
             
            return (
                train_arr,
                test_arr)
                 

        except Exception as e:
            raise CustomException(e,sys)
