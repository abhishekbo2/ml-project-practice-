import os 
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.util import save_object
from src.util import evaluate_model


@dataclass
class Model_Training_config:
    trained_model_file_path = os.path.join("artifacts", "Model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modeltrainingconfig = Model_Training_config()

    def Initate_Model_training(self,train_data, test_data):
        try:
            logging.info("splitting of data into training and testing was initiated")

            X_train,X_test,Y_train,Y_test = (
                train_data[:,:-1],
                test_data[:,:-1],
                train_data[:,-1],
                test_data[:,-1]
            )

            models = {
                "LinearRegression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "SVR" : SVR()
            }

            model_report:dict = evaluate_model(x_train = X_train, x_test = X_test, y_train = Y_train, y_test = Y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best performing model found")
            logging.info(f"the best performing model was {best_model_name}")

            save_object(
                file_path = self.modeltrainingconfig.trained_model_file_path,
                obj = best_model
            )

            best_model_prediction = best_model.predict(X_test)
            r2_score_of_best_model = r2_score(Y_test, best_model_prediction)

            return r2_score_of_best_model

        except Exception as e:
            raise CustomException(e,sys)