import os
import sys
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

#import dill
import pickle

from utils.exceptions import CustomException

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        print(f"Directory path: {dir_path}")
        print(f"Filepath: {filepath}")

        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        i = 0
        for model in models:
            curr_model = models[model]

            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)
            error = mean_squared_error(y_pred, y_test)

            report[list(models.keys())[i]] = error
            i += 1
        return report
    
    except Exception as e:
        raise CustomException(e)
    

def load_object(filepath):
    try:
        with open(file=filepath, mode='rb') as obj:
            data = pickle.load(obj)
            return data
    except Exception as e:
        raise CustomException(e)

    

