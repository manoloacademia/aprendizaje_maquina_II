"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Initialize an instance of YourClassName.

        Parameters:
        - input_path (str): The file path of the input data.
        - output_path (str): The file path where the output will be saved.

        Returns:
        - None
        """
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Read data from a CSV file specified by the input_path and return it as a Pandas DataFrame.

        Returns:
        - pandas.DataFrame: The DataFrame containing the data from the CSV file.
        """
        pandas_df = pd.read_csv(self.input_path)
        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        # Defining numeric and categorical features of the dataset
        numeric_features = df.select_dtypes(include = ['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include = ['object']).columns
        target = ['SalePrice']
        
        # Combine all the preprocessing steps into a single pipeline for numeric features
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Step 1: Impute missing values
            ('scaler', StandardScaler()),  # Step 2: Scale the data
            ('capper', CappingTransformer(threshold=1.5))  # Step 3: Apply the CappingTransformer
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Step 1: Impute missing values
            ('onehot', OneHotEncoder())  # Step 2: One-hot encode categorical features
        ])

        # Create a ColumnTransformer to apply the numeric_pipeline to numeric features and the categorical_pipeline to categorical features
        feature_transformer = ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, numeric_features),
                ('categoric', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough'  # Pass through features not specified in transformers
        )

        # Combine all the steps into a single pipeline
        full_pipeline = Pipeline([
            ('feature_transform', feature_transformer)
        ])

        numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath',
            'YearRemodAdd', 'YearBuilt', 'Fireplaces', 'MasVnrArea', 'LotArea',
            'SalePrice']
        categorical_features = ['ExterQual', 'CentralAir']
        metadata_features  = ['Id']

        final_features = numeric_features + categorical_features + metadata_features

        full_pipeline.fit(X = data_cleaned[final_features])
        final_data = full_pipeline.transform(X = data_cleaned[final_features])
        final_data_df = pd.DataFrame(final_data, columns = full_pipeline.get_feature_names_out())
        df_transformed = final_data_df.rename(columns=lambda x: x.replace('numeric__', '').replace('categoric__', '').replace('remainder__', ''))
        
        return df_transformed

    def write_prepared_data(self, transformed_dataframe):
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = 'Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
                               output_path = 'Ruta/Donde/Voy/A/Escribir/Mi/Archivo').run()