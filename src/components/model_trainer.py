import os,sys
import numpy as np 
import pandas as pd
import mlflow
import dagshub

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from src.utils import evalute_model
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

dagshub.init(repo_owner='arpitkanani', repo_name='Cement-Strengthen-prediction-', mlflow=True)

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("model trainer is start.")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:-1],
                test_arr[:,:-1],
                test_arr[:-1]
            )

            models= {
                "Linear Regerssion":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "DecisionTree Regeressor":DecisionTreeRegressor(),
                "RandomForest Regressor":RandomForestRegressor(),
                "AdaBoost Rgeressor":AdaBoostRegressor(),
                "GradiantBoost Regressor":GradientBoostingRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor()
            }

            param = {

                "Linear Regression": {
                #"fit_intercept": [True, False]
                },

                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    "max_iter": [1000, 5000]
                },

                "Ridge": {
                    "alpha": [0.1, 1, 10, 50],
                    "solver": ["auto", "svd", "cholesky"]
                },

                "DecisionTree Regeressor": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5,3, 10, 20],
                   # "min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4]
                },

                "RandomForest Regressor": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },

                "GradiantBoost Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    #"subsample": [0.8, 1.0]
                },

                'XGB Regressor' : {
                    "n_estimators": [200, 400],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [4, 6, 8],
                    #"subsample": [0.7, 0.8, 1.0],
                    #"colsample_bytree": [0.7, 0.8, 1.0],
                    "reg_lambda": [1, 5, 10]    
                },
                'CatBoost Regressor':{
                    #"iterations": [300, 500],
                    #"learning_rate": [0.03, 0.05, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5, 7]
                }
                
            }

            model_report:dict=evalute_model(X_train,y_train,X_test,y_test,models,param)
            
            logging.info(f"model report :{model_report}")

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(f"best model is :{best_model_name},R2 score : {best_model_score}")
            print("="*12)

            logging.info(f"best Model Found, Model Name :{best_model_name}, R2_score : {best_model_score}")

            print("this is the best model")
            print(best_model_name)

            models_names=list(param.keys())
            
            actual_model=""

        except Exception as e:
            logging.info("error occured in model trainer method.")
            raise CustomException(e,sys) #type: ignore
