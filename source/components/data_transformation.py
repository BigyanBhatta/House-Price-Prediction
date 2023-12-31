import sys 
from dataclasses import dataclass
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os 

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__ (self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_cols = ['bath', 'balcony', 'rectified_sqft']
            cat_cols = ['area_type', 'availabletomove']
            
            num_pipeline = Pipeline(
                steps = [
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('one_hot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean= False))
                ]
            )
            logging.info('standard scaling of numerical columns completed')
            logging.info('encoding of categorical columns completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read train and test data completed')


            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "price"
            numerical_columns = ['bath', 'balcony', 'rectified_sqft']

            input_feature_train_df = train_df.drop(columns= [target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_ [
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_ [input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )    
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)






