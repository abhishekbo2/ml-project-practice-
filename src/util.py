import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dire_name = os.path.dirname(file_path)

        os.makedirs(dire_name, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(x_train,x_test,y_train,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj) 
        except Exception as e:
            raise CustomException(e,sys)
