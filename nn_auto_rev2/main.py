import data_preprocessing as dp
import pandas as pd
import torch
import os
from model import ComplexAutoencoder, ComplexNet
from train import train_autoencoder, train_nn

def main():
    # Load and preprocess data
    preprocessed_data_path = '../data/preprocessed_data_rev2.pkl'

    # Determine the device to use, use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the preprocessed data already exists
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data...")
        # Load the preprocessed data
        preprocessed_data = pd.read_pickle(preprocessed_data_path)
        gene_columns = preprocessed_data['gene_columns']
        encoded_features = preprocessed_data['encoded_features']
        mean_features = preprocessed_data['mean_features']
        std_features = preprocessed_data['std_features']
        X = preprocessed_data['X']
        y = preprocessed_data['y']
        all_column_names = preprocessed_data['all_column_names']
    else:
        print("Preprocessing data...")
        # Load and preprocess the data
        file_path = "../data/de_train.parquet"
        data = dp.load_data(file_path)
        gene_columns = dp.get_gene_columns(data)
        encoded_features, mean_features, std_features, X, y, all_column_names = dp.preprocess_data(data, gene_columns)

        # Save the preprocessed data for future use
        preprocessed_data = {
            'gene_columns': gene_columns,
            'encoded_features': encoded_features,
            'mean_features': mean_features,
            'std_features': std_features,
            'X': X,
            'y': y,
            'all_column_names': all_column_names
        }
        pd.to_pickle(preprocessed_data, preprocessed_data_path)

    # # Create a DataFrame for the processed data using the provided column names
    # processed_df = pd.DataFrame(X, columns=all_column_names)

    # # Save the first few rows of the processed data to a file
    # processed_df.head().to_csv("../data/processed_data.csv", index=False)

    # Instantiate and train the autoencoder
    input_size = X.shape[1]
    latent_size = 128  # Example latent size, adjust as needed
    autoencoder = ComplexAutoencoder(input_size, latent_size)
    train_autoencoder(autoencoder, X, X, epochs=50, lr_rate=0.001, batch_size=32)  # Adjust parameters as needed

    # Dimensionality reduction with autoencoder
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    reduced_X = autoencoder.encode(X_tensor)

    # Instantiate and train the ComplexNet
    output_size = 18211  # Number of gene expressions
    complex_net = ComplexNet(latent_size, output_size)
    train_nn(complex_net, reduced_X, y, epochs=100, lr_rate=0.001, batch_size=32)  # Adjust parameters as needed

    # Additional code for model evaluation, saving, or making predictions
    # ...

if __name__ == "__main__":
    main()
