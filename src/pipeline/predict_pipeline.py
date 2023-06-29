import sys
import os
import pandas as pd
from src.exception import CustomException
from src.helper import load_object
from src.components.data_transformation import clean_text
import warnings
warnings.filterwarnings("ignore")

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            label_path = os.path.join("artifacts","labels.pkl")
            vectorizer_path = os.path.join("artifacts","tfidfvectorizer.pkl")
            
            model = load_object(file_path = model_path)
            labelizer = load_object(file_path = label_path)
            vectorizer = load_object(file_path = vectorizer_path)

            #data = pd.DataFrame(features,columns = ["text"])
            data = [clean_text(features)]
            
            x = vectorizer.transform(data)
            pred = model.predict(x)
            pred = labelizer.inverse_transform([int(pred)])
            return pred

        except Exception as e:
            raise CustomException( e, sys)

        
class CustomData:
    def __init__(self,text:str):
        self.text = text
    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                "text" : [self.text]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException( e, sys)

if __name__ == "__main__":
    predict_pipeline = PredictPipeline()
    test_data =  input("enter text.....\n")
    result = predict_pipeline.predict(test_data)
    print(result[0])
    
    