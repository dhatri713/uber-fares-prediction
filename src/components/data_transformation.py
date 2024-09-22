import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
import pandas as pd
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import logging
from utils.exceptions import CustomException
from utils.utility_funs import save_object

class RemoveNullValues(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data):
        return data
    
    def transform(self, data):
        data = data.dropna()
        return data

class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, data):
        return data
    
    def transform(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
        return data
class RemoveUnwantedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
    
class CalculateDistance(BaseEstimator, TransformerMixin):
    def __init__(self, lat1_col, lat2_col, long1_col, long2_col) -> None:
        super().__init__()
        self.lat1_col = lat1_col
        self.lat2_col = lat2_col
        self.long1_col = long1_col
        self.long2_col = long2_col
    
    def fit(self, data):
        return self
    
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        R = 6371.0
        distance = R * c
        
        return distance
    
    def transform(self, data):
        distances = []

        for i in range(len(data)):
            lat1 = data['pickup_latitude'][i]
            lat2 = data['dropoff_latitude'][i]
            lon1 = data['pickup_longitude'][i]
            lon2 = data['dropoff_longitude'][i]
            dist = self.haversine(lat1, lon1, lat2, lon2)
            distances.append(dist * 1000)

        data['distance_haversine'] = distances

        return data

class ExtractTemporalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datatime_column) -> None:
        super().__init__()
        self.datetime_column = datatime_column
    
    def fit(self, data):
        return data
    
    def transform(self, data):
        data['datetime'] = pd.to_datetime(data[self.datetime_column])
        data['month'] = data['datetime'].dt.month
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.weekday

        return data

@dataclass
class DataTransformationConfig():
    preprocessor_obj_filepath = os.path.join('data',"proprocessor.pkl") # this is where the pickl file for the preprocessor will be stored

class DataTransformation():
    def __init__(self):
        self.preprocessor_obj_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:

            # Define columns and drop lists
            columns = ['Unnamed: 0', 'key', 'pickup_datetime', 
                    'pickup_longitude', 'pickup_latitude', 
                    'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
            drop1 = ['Unnamed: 0', 'key']
            categorical_feats = ['passenger_count', 'month', 'hour', 'day_of_week']
            numerical_feats = ['distance_haversine']
            drop2 = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
                    'dropoff_longitude', 'dropoff_latitude']

            # Define the numerical preprocessing pipeline
            numerical_pipeline = Pipeline([
                ('remove_nulls', RemoveNullValues()),
                ('remove_outliers', RemoveOutliers()),
                ('haversine_distance', CalculateDistance(
                    lat1_col='pickup_latitude',
                    lon1_col='pickup_longitude',
                    lat2_col='dropoff_latitude',
                    lon2_col='dropoff_longitude')),
                ('extract_temporal', ExtractTemporalFeatures(datetime_column='pickup_datetime')),
                ('scaler', StandardScaler())
            ])

            # Define the categorical preprocessing pipeline
            categorical_pipeline = Pipeline([
                ('onehot_encoding', OneHotEncoder())
            ])

            # Create the ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numerical_pipeline, numerical_feats),
                    ('categorical', categorical_pipeline, categorical_feats)
                ],
                remainder='drop'  # This will drop any columns not specified in the transformers
            )

            # Final pipeline to preprocess the data
            final_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                # Add any final steps here, e.g., model fitting
            ])

            logging.info("Transformer made")

            return preprocessor
        except Exception as e:
            raise CustomException(e)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Train and test data read")
            logging.info("Obtaining preprocessor object")

            X_train = train_data.drop(["fare_amount"], axis=1)
            y_train = train_data["fare_amount"]
            X_test = test_data.drop(["fare_amount"], axis=1)
            y_test = test_data["fare_amount"]

            new_columns = ['Unnamed: 0', 'key', 'pickup_datetime', 
                    'pickup_longitude', 'pickup_latitude', 
                    'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

            X_train.columns = new_columns
            X_test.columns = new_columns

            preprocessing_obj = self.get_data_transformer_obj()

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[
                X_train_arr, np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_arr, np.array(y_test)
            ]

            logging.info("Test and train data trasnformed")

            save_object(
                filepath=self.preprocessor_obj_config.preprocessor_obj_filepath,
                preprocess_obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_config.preprocessor_obj_filepath
            )
        except Exception as e:
            raise CustomException(e)