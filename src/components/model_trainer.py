import sys
import os
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier,
                              AdaBoostRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            #takout last column
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                }
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model= models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted  = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model: {r2_square}")
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
   