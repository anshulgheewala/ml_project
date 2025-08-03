import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor, GradientBoostingRegressor
from src.utils import save_object
from src.utils import evaluate_model
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train, x_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Gradient Boosting": GradientBoostingRegressor(),
            }

            params={
                "Linear Regression":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            print(f"x_train shape: {x_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}")
            print(f"y_test shape: {y_test.shape}")

            model_report: dict=evaluate_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, models=models,params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name =  list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException("No best Model Found")
            
            logging.info(f"Best model found on both training and testing datasets is {best_model}")
            logging.info(f"Best Params for the model is {best_model.get_params()}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
    