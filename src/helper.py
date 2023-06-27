import os
import sys
import pandas as pd
import dill 
import pickle
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)
        with open(file_path,"wb") as file_object:
            pickle.dump(obj, file_object)

    except Exception as e:
        raise CustomException(e, sys)