import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj = os.path.join("artifacts","preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()
    
    def get_data_transformation(self):
        try:

            numerical_features = [
                'reading_score', 
                'writing_score'
                ]
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
                ]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("Scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("one_hot_encoding", OneHotEncoder())
                ]
            )

            perprocessor = ColumnTransformer(
                [
                    ("numerical", num_pipeline, numerical_features),
                    ("categorical", cat_pipeline, categorical_features)
                ]
            )

            return perprocessor

        except Exception as e:
            raise CustomException(e,sys)
    


    def Initiate_Data_Transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("the train and test data had been read")

            logging.info("obtaining preprocesing data")
            preprocessing_obj = self.get_data_transformation()

            target_column = "math_score"

            input_features_for_train_df = train_df.drop(columns = [target_column],axis=1)
            target_features_for_train_df = train_df[target_column]

            input_features_for_test_df = test_df.drop(columns=[target_column], axis=1)
            target_features_for_test_df = test_df[target_column]

            logging.info("applying preprocessing for the training features ")

            preprocessed_input_train_arr = preprocessing_obj.fit_transform(input_features_for_train_df)
            preprocessed_input_test_arr = preprocessing_obj.transform(input_features_for_test_df)

            target_with_processed_train_arr = np.c_[
                preprocessed_input_train_arr, np.array(target_features_for_train_df)
            ]

            target_with_processed_test_arr = np.c_[
                preprocessed_input_test_arr, np.array(target_features_for_test_df)
            ]

            save_object(
                file_path = self.datatransformationconfig.preprocessor_obj,
                obj = preprocessing_obj
            )

            return (
                target_with_processed_train_arr,
                target_with_processed_test_arr,
                self.datatransformationconfig.preprocessor_obj
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        