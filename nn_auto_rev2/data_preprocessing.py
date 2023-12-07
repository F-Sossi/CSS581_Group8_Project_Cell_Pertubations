import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import TensorDataset
import torch
import joblib


def load_data(file_path):
    """
    Load data from a specified file path.
    """
    return pd.read_parquet(file_path)


def encode_categorical_features(data, columns):
    """
    Apply one-hot encoding to the specified categorical columns in the data.

    Parameters:
    - data: pandas DataFrame containing the categorical columns.
    - columns: list of strings, column names to be one-hot encoded.

    Returns:
    - A NumPy array containing the one-hot encoded features.
    """
    encoder = OneHotEncoder()
    # Save the fitted encoder to a file
    joblib.dump(encoder, 'encoder.joblib')
    output = encoder.fit_transform(data[columns]).toarray()
    return output


def get_gene_columns(data):
    """
    Obtain gene columns by excluding certain predefined columns.

    Parameters:
    - data: pandas DataFrame from which to extract gene column names.

    Returns:
    - An Index object containing the names of the gene columns.
    """
    return data.columns.difference(['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'])


def fill_missing_values(data, columns):
    """
    Fill missing values in the specified columns of the data with the mean of each column.

    Parameters:
    - data: pandas DataFrame containing the columns with missing values.
    - columns: list of strings, column names in which to fill missing values.

    Returns:
    - The pandas DataFrame with missing values filled.
    """
    data[columns] = data[columns].fillna(data[columns].mean())
    return data


def compute_group_stats(data, group_cols, data_cols):
    """
    Compute the mean and standard deviation for specified groups within the data.

    Parameters:
    - data: pandas DataFrame containing the data to group.
    - group_cols: list of strings, column names to group by.
    - data_cols: list of strings, column names for which to calculate statistics.

    Returns:
    - Two pandas DataFrames: the first containing the means, and the second containing the standard deviations.
    """
    grouped = data.groupby(group_cols)

    mean_list = []
    std_list = []

    for group_name, group_data in data.groupby(group_cols, as_index=False):
        if len(group_data) == 1:
            mean_list.append(group_data[data_cols].iloc[0])
            std_list.append(pd.Series(0, index=data_cols))
        else:
            mean_list.append(group_data[data_cols].mean())
            std_list.append(group_data[data_cols].std())

    mean_df = pd.DataFrame(mean_list)
    std_df = pd.DataFrame(std_list)

    mean_df[group_cols] = pd.DataFrame(grouped[data_cols].mean().reset_index()[group_cols].values)
    std_df[group_cols] = pd.DataFrame(grouped[data_cols].std().reset_index()[group_cols].values)

    mean_df.reset_index(drop=True, inplace=True)
    std_df.reset_index(drop=True, inplace=True)

    return mean_df, std_df


from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def preprocess_data(data, gene_columns):
    """
    Preprocess the dataset to prepare for model training. This includes encoding categorical features,
    filling missing values, and computing group statistics.

    Parameters:
    - data: pandas DataFrame containing the raw data.
    - gene_columns: list of strings, column names corresponding to gene data.

    Returns:
    - A tuple of processed features and column names: (encoded_features, mean_features, std_features, X, y, all_column_names)
    """
    # Drop the columns 'sm_lincs_id', 'SMILES'
    data = data.drop(['sm_lincs_id', 'SMILES'], axis=1)

    # One-hot encoding
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(data[['cell_type', 'sm_name']]).toarray()
    encoded_feature_names = encoder.get_feature_names_out(['cell_type', 'sm_name'])

    joblib.dump(encoder, '../encoders/encoder_rev2.joblib')

    # Fill missing values in the data
    data = fill_missing_values(data, gene_columns)

    # Compute mean and standard deviation statistics grouped by 'sm_name' only
    mean_features, std_features = compute_group_stats(data, ['sm_name'], gene_columns)

    # Merge the computed mean and standard deviation with the original data
    data = data.merge(mean_features, on='sm_name', suffixes=('', '_mean'))
    data = data.merge(std_features, on='sm_name', suffixes=('', '_std'))

    # Extract the mean and standard deviation features
    mean_std_feature_names = [col for col in data.columns if '_mean' in col or '_std' in col]
    mean_std_features = data[mean_std_feature_names].values

    # Prepare the feature matrix X by concatenating encoded features, scaled gene data, and mean_std features
    scaled_gene_data = data[gene_columns].values
    X = np.concatenate([encoded_features, scaled_gene_data, mean_std_features], axis=1)

    # Prepare the target variable y
    y = data[gene_columns].values

    # Combining all column names
    all_column_names = list(encoded_feature_names) + gene_columns.tolist() + mean_std_feature_names

    return encoded_features, mean_features, std_features, X, y, all_column_names


def oversample_data(data, column, desired_number=200):
    """
    Perform oversampling to balance the dataset. not used in the final model
    """
    minority_data = data[data[column].isin(['B cells', 'Myeloid cells'])]
    oversampled_data = resample(minority_data, replace=True, n_samples=desired_number, random_state=42)
    balanced_data = pd.concat([data, oversampled_data]).drop_duplicates().reset_index(drop=True)
    return balanced_data


def prepare_dataset(data, input_features, target_feature):
    """
    Convert data into a format suitable for training.
    """
    X = data[input_features].values
    y = data[target_feature].values
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))


def split_data(data, test_size=0.2):
    """
    Split the data into training and validation sets.
    """
    return train_test_split(data, test_size=test_size, random_state=42)


def process_features(data, encoder, mean_features, std_features):
    """
    Process the features of the data to prepare for inference.
    """
    # Apply the same one-hot encoding as was done to the training data
    encoded_features = encoder.transform(data[['cell_type', 'sm_name']]).toarray()

    return encoded_features
