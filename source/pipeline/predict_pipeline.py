import sys
import pandas as pd
from source.exception import CustomException
from source.utils import load_object

class PredictPipeline:
    def __init__ (self):
        pass


    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            print('before loading')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            print('after processing')
            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, area_type:str, bath:int, balcony:int, availabletomove: str, rectified_sqft: float):
        self.area_type = area_type
        self.bath = bath
        self.balcony = balcony
        self.availabletomove = availabletomove
        self.rectified_sqft = rectified_sqft

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "area_type": [self.area_type],
                "bath" : [self.bath],
                "balcony": [self.balcony],
                "availabletomove": [self.availabletomove],
                "rectified_sqft": [self.rectified_sqft]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        


