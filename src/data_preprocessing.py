import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path):
    # Load the data from a file
    return pd.read_parquet('../data/de_train_clustered.parquet')


def preprocess_data(data):
    # Apply preprocessing steps like encoding, normalization, etc.
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
    return encoded_data, encoder


def split_data(data, test_size=0.3):
    # Split the data into training and validation sets
    return train_test_split(data, test_size=test_size, random_state=42)
