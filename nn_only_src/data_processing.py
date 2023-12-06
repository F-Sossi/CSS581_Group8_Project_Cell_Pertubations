import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def load_data(file_path):
    """
    Load a dataset from a Parquet file located at the specified file path.

    Parameters:
    - file_path: string, path to the Parquet file to load.

    Returns:
    - A pandas DataFrame containing the data from the Parquet file.
    """
    return pd.read_parquet(file_path)


def get_gene_columns(data):
    """
    Obtain gene columns by excluding certain predefined columns.

    Parameters:
    - data: pandas DataFrame from which to extract gene column names.

    Returns:
    - An Index object containing the names of the gene columns.
    """
    return data.columns.difference(['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'])


def encode_categorical_features(data, columns):
    """
    Apply one-hot encoding to the specified categorical columns in the data.

    Parameters:
    - data: pandas DataFrame containing the categorical columns.
    - columns: list of strings, column names to be one-hot encoded.

    Returns:
    - A NumPy array containing the one-hot encoded features.
    """
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(data[columns])


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

    for group_name, group_data in grouped:
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


# def preprocess_data(data, gene_columns):
#     """
#     Preprocess the dataset to prepare for model training. This includes encoding categorical features,
#     filling missing values, and computing group statistics.
#
#     Parameters:
#     - data: pandas DataFrame containing the raw data.
#     - gene_columns: list of strings, column names corresponding to gene data.
#
#     Returns:
#     - A tuple of processed features: (encoded_features, mean_features, std_features, X, y)
#       where 'X' is the feature matrix and 'y' is the target variable.
#     """
#     encoded_features = encode_categorical_features(data, ['cell_type', 'sm_name'])
#     data = fill_missing_values(data, gene_columns)
#     mean_features, std_features = compute_group_stats(data, ['cell_type', 'sm_name'], gene_columns)
#     data = data.merge(mean_features, on=['cell_type', 'sm_name'], suffixes=('', '_mean'))
#     data = data.merge(std_features, on=['cell_type', 'sm_name'], suffixes=('', '_std'))
#
#     mean_std_features = data[[col for col in data.columns if '_mean' in col or '_std' in col]]
#     scaled_gene_data = data[gene_columns].values
#     X = np.concatenate([encoded_features, scaled_gene_data, mean_std_features], axis=1)
#     y = data[gene_columns].values
#
#     return encoded_features, mean_features, std_features, X, y
def preprocess_data(data, gene_columns):
    """
    Preprocess the dataset to prepare for model training. This includes encoding categorical features,
    filling missing values, and computing group statistics.

    Parameters:
    - data: pandas DataFrame containing the raw data.
    - gene_columns: list of strings, column names corresponding to gene data.

    Returns:
    - A tuple of processed features: (encoded_features, mean_features, std_features, X, y)
      where 'X' is the feature matrix and 'y' is the target variable.
    """
    # Keep the encoding for both 'cell_type' and 'sm_name'
    encoded_features = encode_categorical_features(data, ['cell_type', 'sm_name'])

    # Fill missing values in the data
    data = fill_missing_values(data, gene_columns)

    # Compute mean and standard deviation statistics grouped by 'sm_name' only
    mean_features, std_features = compute_group_stats(data, ['sm_name'], gene_columns)

    # Merge the computed mean and standard deviation with the original data
    # Note that the merge is based on 'sm_name' only
    data = data.merge(mean_features, on='sm_name', suffixes=('', '_mean'))
    data = data.merge(std_features, on='sm_name', suffixes=('', '_std'))

    # Extract the mean and standard deviation features
    mean_std_features = data[[col for col in data.columns if '_mean' in col or '_std' in col]]

    # Prepare the feature matrix X by concatenating encoded features, scaled gene data, and mean_std features
    scaled_gene_data = data[gene_columns].values
    X = np.concatenate([encoded_features, scaled_gene_data, mean_std_features], axis=1)

    # Prepare the target variable y
    y = data[gene_columns].values

    return encoded_features, mean_features, std_features, X, y


def prepare_combinations_data(combinations_file_path, data_file_path):
    """
    Prepare feature matrix for prediction by combining encoded categorical features, imputed means,
    and standard deviations for each combination of cell type and small molecule.

    Parameters:
    - combinations_file_path: string, path to the CSV file containing combinations of cell type and small molecule.
    - data_file_path: string, path to the Parquet file containing the training data.

    Returns:
    - X_combinations: NumPy array, feature matrix for the combinations.
    - combinations_data: pandas DataFrame, original combinations data with 'id'.
    """
    # Load the dataset and combinations data
    data = pd.read_parquet(data_file_path)
    combinations_data = pd.read_csv(combinations_file_path)
    gene_columns = data.columns.difference(['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'])

    # Encode, fill NaNs, compute means and stds
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(data[['cell_type', 'sm_name']])
    encoded_combinations = encoder.transform(combinations_data[['cell_type', 'sm_name']])
    data[gene_columns] = data[gene_columns].fillna(data[gene_columns].mean())
    mean_features = data.groupby('sm_name')[gene_columns].mean()
    std_features = data.groupby('sm_name')[gene_columns].std()

    # Impute mean and std
    imputed_mean = np.zeros((encoded_combinations.shape[0], len(gene_columns)))
    imputed_std = np.zeros((encoded_combinations.shape[0], len(gene_columns)))
    for i, row in combinations_data.iterrows():
        sm_name = row['sm_name']
        if sm_name in mean_features.index:
            imputed_mean[i] = mean_features.loc[sm_name].values
            imputed_std[i] = std_features.loc[sm_name].fillna(0).values

    # Combine all features
    dummy_gene_data = np.zeros((encoded_combinations.shape[0], len(gene_columns)))
    X_combinations = np.concatenate([encoded_combinations, dummy_gene_data, imputed_mean, imputed_std], axis=1)

    return X_combinations, combinations_data
