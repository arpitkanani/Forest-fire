import pandas as pd
import numpy as np
import os,sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfog:
    train_data_path=os.path.join("artifacts",'train.csv')
    test_data_path=os.path.join("artifacts",'test.csv')
    raw_data_path=os.path.join("artifacts",'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfog()


    def initiate_data_ingestion(self):
        logging.info("data ingestion initiated.")

        try:
            df=pd.read_csv(r"E:\Data Analysis\regression-project\notebooks\data\Algerian_forest_fires_dataset_CLEANED.csv")
            df.drop(columns=['day','month','year','BUI','DC'],inplace=True)
            df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)


            logging.info(f"reading from dataset {df.columns}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)

            train_data,test_data=train_test_split(df,test_size=0.3,random_state=10)
            
            train_data.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,header=True,index=False)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )


        except Exception as e:
            logging.info("error occured in data ingestion")
            raise CustomException(e,sys) #type:ignore
