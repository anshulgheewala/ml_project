import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender','race/ethnicity','parental level of education',
                                   'lunch','test preparation course'
                                   ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline =Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_Pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df =  pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            inp_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            inp_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            # train_arr = np.c_[inp_feature_train_arr, np.array(target_feature_train_df)]
            train_arr = np.c_[inp_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)]


            # test_arr = np.c_[inp_feature_test_arr, np.array(target_feature_test_df)]
            test_arr = np.c_[inp_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)]

            logging.info(f"Saved preprocessing object")


            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e, sys)
 