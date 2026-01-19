import pandas as pd
import sys,os

from src.utils import load_object
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts",'model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            predict=model.predict(data_scaled)

            return predict
        except Exception as e:
            logging.info("error occured in predict pipeline predict method.")
            raise CustomException(e,sys) #type: ignore

class CustomData:
    def __init__(self,
        Temperature:float,
        RH:float,
        Ws:float,
        Rain:float,
        FFMC:float,
        DMC:float,
        ISI:float,
        Classes:int, 
        Region:int
    ):
        self.Temperature=Temperature
        self.RH=RH
        self.Ws=Ws
        self.Rain=Rain
        self.FFMC=FFMC
        self.DMC=DMC
        self.ISI=ISI
        self.Classes=Classes
        self.Region=Region
         
    def get_data_as_dataframe(self):
        try:
            custom_data_input={
                'Temperature':[self.Temperature],
                'RH':[self.RH], 
                'Ws':[self.Ws],
                'Rain':[self.Rain], 
                'FFMC':[self.FFMC],
                'DMC':[self.DMC],  
                'ISI':[self.ISI],
                'Classes':[self.Classes], 
                'Region':[self.Region]
            }
            
            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            logging.info("error occured in get data as dataframe method in prediction pipeline")
            raise CustomException(e,sys) #type:ignore
        