import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTranformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import Model_Training_config
from src.components.model_trainer import ModelTrainer

@dataclass
class Data_Ingestion_Config:
    train_data_path: str = os.path.join("artifacts", "Train.csv")
    test_data_path: str = os.path.join("artifacts", "Test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.dataingestionconfig = Data_Ingestion_Config()
    
    def Initiate_Data_Ingestion(self):
        logging.info("Ingestion of data has been started")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("the data had been read as dataframe")
            
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_data_path),exist_ok=True)
            df.to_csv(self.dataingestionconfig.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)
            train_set.to_csv(self.dataingestionconfig.train_data_path,index = False, header = True)
            test_set.to_csv(self.dataingestionconfig.test_data_path,index = False, header = True)

            logging.info("the data ingestion was done")

            return(
                self.dataingestionconfig.train_data_path,
                self.dataingestionconfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.Initiate_Data_Ingestion()

    data_transformation = DataTranformation()
    train_arr, test_arr, _ = data_transformation.Initiate_Data_Transformation(train_data, test_data)

    model_training = ModelTrainer()
    print(model_training.Initate_Model_training(train_arr, test_arr))