import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utility_funs import load_object
from utils.exceptions import CustomException
from utils.logger import logging

class PredictPipeline():
   def __init__(self, features) -> None:
       self.features = features

   def predict(self):
        try:
            preprocessor_path = os.path.join("data", "proprocessor.pkl")
            model_path = os.path.join("data", "model_trainer.pkl")
            preprocessor_obj = load_object(preprocessor_path)
            model_trainer_obj = load_object(model_path)
            transformed_features = preprocessor_obj.transform(self.features)
            preds = model_trainer_obj.predict(transformed_features)
            return preds
        except Exception as e:
            raise CustomException(e)


class CustomData():
    def __init__(self, Unnamed_0, key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
        
        self.Unnamed_0 = Unnamed_0
        self.key = key
        self.fare_amount = fare_amount
        self.pickup_datetime = pickup_datetime
        self.pickup_longitude = pickup_longitude
        self.pickup_latitude = pickup_latitude
        self.dropoff_longitude = dropoff_longitude
        self.dropoff_latitude = dropoff_latitude
        self.passenger_count = passenger_count

    def convert_ip_to_df(self):
        try:
            ip_dict = {
                "Unnamed_0": [self.Unnamed_0],
                "key": [self.key],
                "fare_amount": [self.fare_amount],
                "pickup_datetime": [self.pickup_datetime],
                "pickup_longitude": [self.pickup_longitude],
                "pickup_latitude": [self.pickup_latitude],
                "dropoff_longitude": [self.dropoff_longitude],
                "dropoff_latitude": [self.dropoff_latitude],
                "passenger_count": [self.passenger_count]
            }
            
            input_df = pd.DataFrame(ip_dict)
            return input_df

        except Exception as e:
            raise CustomException(e)
        

