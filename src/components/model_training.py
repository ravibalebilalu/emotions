import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score,f1_score,precision_score
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
from src.helper import evaluate_models,load_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting trainig and testing data initiated")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("Splitting trainig and testing data compleated")

            logging.info("Preparing to hyperparameter tunning")
            models = {
            "LogisticRegression" : LogisticRegression(),
            "RidgeClassifier" : RidgeClassifier(),
            "KNeighborsClassifier" : KNeighborsClassifier(),
            "DecisionTreeClassifier" : DecisionTreeClassifier(),
            "RandomForestClassifier" : RandomForestClassifier(),
            "AdaBoostClassifier" : AdaBoostClassifier(),
            "XgbClassifier" : XGBClassifier(),
            "CatBoostClassifier" : CatBoostClassifier()
            }
            params = {
               
             "LogisticRegression":{
            #'penalty': ['l1', 'l2'],
            #'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            #'max_iter': [100, 500, 1000],
            #'multi_class': ['auto', 'ovr', 'multinomial'],
            #'class_weight': [None, 'balanced']
            },

            "RidgeClassifier" :{
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'normalize': [True, False],
            #'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'max_iter': [100, 500, 1000],
            #'class_weight': [None, 'balanced']
            },
            "KNeighborsClassifier" :{
        
            'n_neighbors': [3, 5, 7, 9],
            #'weights': ['uniform', 'distance'],
            #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40],
            'p': [1, 2]
            },
            "DecisionTreeClassifier" :{
     
            #'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            #'max_features': [None, 'sqrt', 'log2']
            },

            "RandomForestClassifier":{
     
            'n_estimators': [100, 200, 500],
            #'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            #'max_features': ['auto', 'sqrt', 'log2']
            },
            "AdaBoostClassifier": {
     
            'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0],
            #'algorithm': ['SAMME', 'SAMME.R']
            },
            "XgbClassifier":{
       
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300],
            #'gamma': [0, 0.1, 0.2],
            #'subsample': [0.8, 1.0],
            #'colsample_bytree': [0.8, 1.0],
            #'reg_alpha': [0, 0.1, 0.5],
            #'reg_lambda': [0, 0.1, 0.5],
            'min_child_weight': [1, 5, 10]
            },

            "CatBoostClassifier" :{
     
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100],
            #'l2_leaf_reg': [0.1, 0.5, 1.0],
            #'border_count': [32, 64, 128],
            #'bagging_temperature': [0.0, 1.0],
            #'random_strength': [0.0, 1.0],
            #'scale_pos_weight': [1.0, 2.0, 5.0]
            }
            }

            model_report:dict = evaluate_models(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, models = models, param = params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found ")
            logging.info("Best model found for both training and testing")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model
            )
            predicted = best_model.predict(x_test)
            score_ = accuracy_score(y_test,predicted)
            return score_,best_model_name
        except Exception as e:
            raise CustomException( e, sys)
            

