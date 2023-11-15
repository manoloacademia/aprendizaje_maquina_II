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
from utils import CappingTransformer
from utils import calculate_null_percentage
from utils import calculate_outlier_percentage



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
        Perform data transformation on the input DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the dataset.

        Returns:
        - pd.DataFrame: Transformed DataFrame after preprocessing steps.

        Steps:
        1. Define numeric and categorical features of the dataset.
        2. Combine preprocessing steps into pipelines for numeric and categorical features.
        3. Create a ColumnTransformer to apply pipelines to respective feature types.
        4. Combine all steps into a single pipeline.
        5. Identify specific features for transformation.
        6. Perform data cleaning transformations:
            - Remove specified columns with high outlier percentages and missing values.
            - Drop rows with NA values.
        7. Fit the full pipeline on the cleaned data.
        8. Transform the cleaned data using the full pipeline.
        9. Rename columns in the transformed DataFrame.

        Example:
        ```
        transformer = FeatureEngineeringPipeline()
        transformed_data = transformer.data_transformation(input_dataframe)
        ```

        """
        # Define numeric and categorical features of the dataset
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

        # Specify the features in categories
        numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath',
            'YearRemodAdd', 'YearBuilt', 'Fireplaces', 'MasVnrArea', 'LotArea',
            'SalePrice']
        categorical_features = ['ExterQual', 'CentralAir']
        metadata_features  = ['Id']

        final_features = numeric_features + categorical_features + metadata_features

        # Data cleaned transformation
        # Removing columns
        data_cleaned = df.copy()

        # Calculate null and outlier percentages
        null_percentages = calculate_null_percentage(df)
        outlier_percentages = calculate_outlier_percentage(df[numeric_features])

        # Selecting the 2 features with highest number of outliers and the 6 with highest number of missings
        features_to_drop = outlier_percentages[:2]['Column'].tolist() + null_percentages[:6]['Column'].tolist()
        data_cleaned.drop(columns = features_to_drop, inplace = True)

        # Removing na
        data_cleaned.dropna(inplace = True)

        full_pipeline.fit(X = data_cleaned[final_features])
        final_data = full_pipeline.transform(X = data_cleaned[final_features])
        final_data_df = pd.DataFrame(final_data, columns = full_pipeline.get_feature_names_out())
        df_transformed = final_data_df.rename(columns=lambda x: x.replace('numeric__', '').replace('categoric__', '').replace('remainder__', ''))
        
        return df_transformed

    def write_prepared_data(self, transformed_dataframe) -> None:
        """
        Write the transformed DataFrame to a CSV file.

        Parameters:
        - transformed_dataframe (pd.DataFrame): The DataFrame containing the prepared and transformed data.

        Returns:
        - None

        This function takes a preprocessed DataFrame and saves it to a CSV file named 'transformed_df.csv'.
        The CSV file will be created in the current working directory.

        Example:
        ```
        transformer = YourTransformerClass()
        prepared_data = transformer.data_transformation(input_dataframe)
        transformer.write_prepared_data(prepared_data)
        ```

        """
        transformed_dataframe.to_csv('TP - house price\data\transformed_df.csv')
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = 'TP - house price\data\train.csv', # Chequear rutas con OS!!!!!!!!
                               output_path = 'TP - house price\data\transformed_df.csv').run()