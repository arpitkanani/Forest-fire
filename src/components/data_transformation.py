import pandas as pd
import os,sys
import numpy as np

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:

            logging.info("data transformation initiated")
            df=pd.read_csv("artifacts/train.csv")
            
            target_col = 'FWI'

            num_columns = df.select_dtypes(exclude='object').drop(columns=[target_col]).columns
            cat_columns = df.select_dtypes(include='object').columns


            logging.info("pipeline initiated.")


            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ],
                remainder='drop',
                sparse_threshold=0
            )

            logging.info("pipeline completed.")

            
            return preprocessor
        except Exception as e:
            logging.info("error occured in preprocessor pipeline.")
            raise CustomException(e,sys) #type: ignore

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data is completed.")

            logging.info("obtaining preprocessor object.")

            preprocessor_obj=self.get_data_transformation()

            target_col='FWI'
            drop_columns=[target_col]
            
            input_feature_train_df=train_df.drop(columns=['FWI'])
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=['FWI'])
            target_feature_test_df=test_df[target_col]

            logging.info("Applying preprocessing on both dataset.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            target_feature_train_df = target_feature_train_df.reset_index(drop=True)
            target_feature_test_df = target_feature_test_df.reset_index(drop=True)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]



            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
                )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )


        except Exception as e:
            logging.info("error occured in initate data transformation  method.")
            raise CustomException(e,sys) #type: ignore

