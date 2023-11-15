import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def calculate_outlier_percentage(data, threshold = 1.5):

    outlier_percentages = []

    for column in data.columns:
        # Extract the column data as a NumPy array
        column_data = data[column].values

        # Calculate the IQR and the lower and upper bounds for potential outliers
        Q1 = np.percentile(column_data, 25)
        Q3 = np.percentile(column_data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify the outliers
        num_outliers = len([x for x in column_data if x <
                           lower_bound or x > upper_bound])

        # Calculate the percentage of outliers
        percentage = (num_outliers / len(column_data)) * 100

        outlier_percentages.append((column, percentage))

    # Create a new DataFrame with the outlier percentages
    result_df = pd.DataFrame(outlier_percentages,
                             columns=['Column', 'Outlier Percentage [%]'])

    # Sort the DataFrame in descending order by "Outlier Percentage"
    result_df = result_df.sort_values(by='Outlier Percentage [%]',
                                      ascending=False)

    return result_df

def calculate_null_percentage(data):

    # Calculate the percentage of null values for each column
    null_percentages = (data.isnull().mean() * 100).round(2)

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({'Column': null_percentages.index,
                              'Null Percentage [%]': null_percentages.values})

    # Sort the DataFrame in descending order by "Null Percentage"
    result_df = result_df.sort_values(
        by='Null Percentage [%]', ascending=False)

    return result_df


def outlier_diagnostic_plots(df, variable):
   
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    # Histogram
    sns.histplot(df[variable], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Histogram')

    # QQ plot
    stats.probplot(df[variable], dist="norm", plot=axes[1])
    axes[1].set_title('QQ Plot')

    # Box plot
    sns.boxplot(y=df[variable], ax=axes[2])
    axes[2].set_title('Box & Whiskers')

    fig.suptitle(variable, fontsize=16)
    plt.show()


def feature_target_correlation_df(df, target_column):

    if target_column not in df.columns:
        raise ValueError("Target column not found in the DataFrame.")

    feature_columns = [col for col in df.columns if col != target_column]
    correlations = df[feature_columns].corrwith(df[target_column])
    correlation_df = pd.DataFrame({'Correlation': correlations})
    correlation_df['Absolute Correlation'] = correlation_df['Correlation'].abs()

    # Sort the DataFrame in descending order of absolute correlations
    correlation_df = correlation_df.sort_values(
        by='Absolute Correlation', ascending=False)
    correlation_df.drop(columns=['Absolute Correlation'], inplace=True)

    return correlation_df


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y, palette="husl")
    plt.xticks(rotation=90)


def load_and_split_data(data_frame, target_col):
    
    # Split the dataset into features (X) and the target variable (y)
    X = data_frame.drop(target_col, axis=1)
    y = data_frame[target_col]

    # Split the data into a training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


def train_regressor(model, X_train, y_train, param_grid):

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CappingTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for capping (winsorizing) numerical data using the Interquartile Range (IQR) and a threshold.

    Parameters:
    -----------
    threshold : float, optional (default=1.5)
        The threshold multiplier for the IQR. Determines how far beyond the IQR the limits should extend.

    Attributes:
    -----------
    lower_limit : float
        The lower limit for capping, calculated during fitting.
    upper_limit : float
        The upper limit for capping, calculated during fitting.

    Methods:
    --------
    fit(X, y=None):
        Calculate the lower and upper limits based on the data distribution during training.

    transform(X, y=None):
        Apply capping to the input data using the precomputed lower and upper limits.

    Example Usage:
    -------------
    # Create a capping transformer with a threshold of 1.5
    capper = CappingTransformer(threshold=1.5)

    # Fit the transformer on data
    capper.fit(data)

    # Transform the data using the calculated limits
    capped_data = capper.transform(data)
    """

    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.lower_limit = None
        self.upper_limit = None

    def fit(self, X, y=None):
        """
        Fit the capping transformer to the input data and calculate lower and upper limits.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data for fitting the transformer.
        y : array-like, optional (default=None)
            Ignored. There is no need for a target variable.

        Returns:
        --------
        self : object
            Returns self for method chaining.
        """
        Q1 = np.percentile(X, 25)
        Q3 = np.percentile(X, 75)
        IQR = Q3 - Q1
        self.lower_limit = Q1 - self.threshold * IQR
        self.upper_limit = Q3 + self.threshold * IQR
        return self

    def transform(self, X, y=None):
        """
        Apply capping to the input data using the precomputed lower and upper limits.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to be capped.

        y : array-like, optional (default=None)
            Ignored. There is no need for a target variable.

        Returns:
        --------
        capped_X : ndarray, shape (n_samples, n_features)
            The capped input data.
        """
        capped_X = np.copy(X)
        capped_X[capped_X < self.lower_limit] = self.lower_limit
        capped_X[capped_X > self.upper_limit] = self.upper_limit
        return capped_X

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for transformed data. In this case, the names are preserved.

        Parameters:
        -----------
        input_features : array-like, shape (n_features,), optional (default=None)
            Names of the input features.

        Returns:
        --------
        output_feature_names : array, shape (n_features,)
            The feature names, which are the same as the input feature names.
        """
        return input_features