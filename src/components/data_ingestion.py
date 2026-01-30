# generic imports
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import Custom_Exception
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import model_trianer_config
from src.components.model_trainer import model_training

from dataclasses import dataclass
import os
import sys

#merging with other pipeline

@dataclass
class DataIngestionconfig:
    
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class InitiateDataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
    
    def initiate_data_config(self):
        logging.info('starting the data ingestion')
        try:
            df = pd.read_csv('Notebooks/spam.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path ),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path,index =False,header = True)

            logging.info('starting saving train and test set')
            train_set , test_set = train_test_split(df,test_size = 0.25)


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False , header = True)

            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.raw_data_path
            )



        except Exception as e:
            raise Custom_Exception(e,sys)
        

if __name__ == "__main__":
    dataingestion_obj = InitiateDataIngestion()

    train_data,test_data,raw_data = dataingestion_obj.initiate_data_config()

    # data_transformation_obj = DataTransformation()
    raw_df = pd.read_csv('artifacts/data.csv')

    # training_corpus = data_transformation_obj.get_data_transformer_object(raw_df) # helps in traing word2vec from scratch


    transformer = DataTransformation()

    corpus = transformer.get_data_transformer_object(raw_df)

    w2v_model = transformer.vectorization_of_text(corpus)

    X = transformer.vector_transformation(corpus, w2v_model, raw_df)

    train_arr, test_arr = train_test_split(X, test_size=0.3, random_state=42)

    print(train_arr.shape,test_arr.shape)



    model_trainer_obj = model_training()

    print(model_trainer_obj.initiate_model_training(train_arr,test_arr))




            


