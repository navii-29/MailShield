import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.logger import logging
from src.exception import Custom_Exception
from dataclasses import dataclass
import os
import sys
from src.utils import evaluation,save_object


@dataclass
class model_trianer_config:
    trained_model_path = os.path.join('artifacts','trained_model.pkl')

class model_training:
    def __init__(self):
        self.trained_model = model_trianer_config()

    def initiate_model_training(self, train, test):
        try:
            logging.info('Train-test split started')

            X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
            X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

            model_rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            model_report: dict = evaluation(
                X_train, y_train, X_test, y_test, model_rf
            )

            logging.info(f"Model performance report: {model_report}")

            save_object(
                model_path=self.trained_model.trained_model_path,
                obj=model_rf
            )

            return model_report

        except Exception as e:
            raise Custom_Exception(e, sys)





