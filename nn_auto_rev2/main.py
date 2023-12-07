import data_preprocessing as dp
import pandas as pd
import torch
import joblib
import os
from model import ComplexAutoencoder, ComplexNet
from train import train_autoencoder, train_nn


def main():
    """
    Main function for training and evaluating the model.
    assumes the data is in the data folder as per the README
    First run will do the initial preprocessing and save the preprocessed data to a pickle file
    Subsequent runs will load the preprocessed data from the pickle file
    This will speed up subsequent runs
    """
    # Load and preprocess data
    preprocessed_data_path = '../data/preprocessed_data_rev2.pkl'

    # Determine the device to use, use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the preprocessed data already exists
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data...")
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

    # print all sizes of data for debugging
    # print("gene_columns.shape:", gene_columns.shape)
    # print("encoded_features.shape:", encoded_features.shape)
    # print("mean_features.shape:", mean_features.shape)
    # print("std_features.shape:", std_features.shape)
    # print("X.shape:", X.shape)
    # print("y.shape:", y.shape)

    # Instantiate and train the autoencoder
    input_size = X.shape[1]
    latent_size = 128
    autoencoder = ComplexAutoencoder(input_size, latent_size).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    train_autoencoder(autoencoder, X_tensor, X_tensor, epochs=200, lr_rate=0.001, batch_size=32)

    # Dimensionality reduction with autoencoder
    reduced_X = autoencoder.encode(X_tensor)

    # Check for NaN and infinite values
    if torch.isnan(reduced_X).any():
        print("NaN values found in reduced data")
    if torch.isinf(reduced_X).any():
        print("Infinite values found in reduced data")

    # Convert to NumPy for visualization
    encoded_X_np = reduced_X.detach().cpu().numpy()

    # Visualize data distributions
    import matplotlib.pyplot as plt
    plt.hist(encoded_X_np, bins=30)
    plt.title('Distribution of Reduced Features')
    plt.show()

    # Compare statistics
    print("Mean of original data:", X.mean())
    print("Mean of reduced data:", encoded_X_np.mean())

    # Ensure encoded_features is a tensor and move to the correct device
    encoded_features_tensor = torch.tensor(encoded_features, dtype=torch.float32).to(device)

    # Instantiate and train the ComplexNet
    output_size = reduced_X.shape[1]

    # print("reduced_X.shape[1]:", reduced_X.shape[1])
    print("output_size:", output_size)

    complex_net = ComplexNet(encoded_features.shape[1], output_size).to(device)
    train_nn(complex_net, encoded_features_tensor, reduced_X, epochs=200, lr_rate=0.001, batch_size=32)

    # Load the new data
    new_data_path = "../data/id_map.csv"
    new_data = pd.read_csv(new_data_path)

    # Load the sample submission file
    sample_submission_path = '../data/sample_submission.csv'
    sample_submission = pd.read_csv(sample_submission_path)

    # Save models
    torch.save(autoencoder.state_dict(), '../models/autoencoder_rev2.pth')
    torch.save(complex_net.state_dict(), '../models/complex_net_rev2.pth')

    # Load the trained models (if not already loaded and in eval mode)
    autoencoder.eval()
    complex_net.eval()

    # Load the fitted encoder
    encoder = joblib.load('../encoders/encoder_rev2.joblib')

    # Process new data features
    processed_features_new = dp.process_features(new_data, encoder, mean_features, std_features)

    # Convert to tensor and predict latent space
    processed_features_tensor_new = torch.tensor(processed_features_new, dtype=torch.float32).to(device)
    predicted_latent_space_new = complex_net(processed_features_tensor_new)

    # Decode to get the full vector
    predicted_full_vector_new = autoencoder.decoder(predicted_latent_space_new).detach().cpu().numpy()

    # Create a DataFrame for the predicted gene expressions
    predicted_gene_expressions = predicted_full_vector_new[:, :18211]
    predicted_df = pd.DataFrame(predicted_gene_expressions,
                                columns=sample_submission.columns[1:])  # Exclude the 'Id' column

    # Add the ID column from 'id_map.csv'
    id_map = pd.read_csv('../data/id_map.csv')
    predicted_df.insert(0, 'Id', id_map['id'])  # Insert 'Id' at the first position

    # Save the predictions
    predicted_df.to_csv('../output/predictions_rev2.csv', index=False)


if __name__ == "__main__":
    main()
