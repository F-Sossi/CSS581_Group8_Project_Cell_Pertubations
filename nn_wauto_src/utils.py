import logging
import torch
import joblib
from model import ComplexNet
from model import ComplexAutoencoder
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def run_clustering():
    # Load dataset
    data = pd.read_parquet('../data/de_train.parquet')

    print(data.shape)

    # Function to convert SMILES to fingerprints
    def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        else:
            raise ValueError("Invalid SMILES string")

    # Generate fingerprints
    fingerprints = data['SMILES'].apply(smiles_to_fingerprint).tolist()

    # Calculate pairwise similarity matrix
    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
    for i in range(len(fingerprints)):
        for j in range(len(fingerprints)):
            similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])

    # Clustering
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                         distance_threshold=0.8)
    data['Cluster'] = clustering.fit_predict(1 - similarity_matrix)

    # Group by cluster and check conditions
    grouped_data = data.groupby('Cluster')
    cluster_conditions = {}

    for cluster, cluster_data in grouped_data:
        b_cell_count = cluster_data[cluster_data['cell_type'] == 'B cells'].shape[0]
        myeloid_cell_count = cluster_data[cluster_data['cell_type'] == 'Myeloid cells'].shape[0]

        condition_1 = b_cell_count >= 2  # 2 or more B cells
        condition_2 = myeloid_cell_count >= 2  # 2 or more Myeloid cells
        condition_3 = condition_1 and condition_2  # both 1 and 2

        cluster_conditions[cluster] = {'Condition 1': condition_1, 'Condition 2': condition_2,
                                       'Condition 3': condition_3}

    # print the number of rows in each cluster
    for cluster, cluster_data in grouped_data:
        print(f'Cluster {cluster}: {cluster_data.shape[0]} rows')

    # print the conditions for each cluster
    for cluster, conditions in cluster_conditions.items():
        print(f'Cluster {cluster}: {conditions}')

    # Save the clustered data
    data.to_parquet('../data/de_train_clustered.parquet')


def preprocess_data(df, encoder):
    """Preprocess the input data and return PyTorch tensor."""
    X = encoder.transform(df[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
    return torch.tensor(X, dtype=torch.float32)


def run_training(ae_params, nn_params):
    # Unpack parameters for autoencoder and neural network
    ae_epochs, ae_lr_rate, ae_batch_size, ae_latent_size, ae_oversample = ae_params
    nn_epochs, nn_lr_rate, nn_oversample = nn_params

    # Train autoencoder
    train_autoencoder(ae_epochs, ae_lr_rate, ae_batch_size, ae_latent_size, ae_oversample)

    # Train neural network and get RMSE
    rmse = train_nn(nn_epochs, nn_lr_rate, ae_batch_size, ae_latent_size, nn_oversample)

    return (ae_params, nn_params, rmse)


def make_output_file(latent_size):
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the full dataset to extract additional information
    full_data = pd.read_parquet('../data/de_train_clustered.parquet')
    additional_info = full_data[['sm_name', 'SMILES', 'Cluster']].drop_duplicates()

    # Load and enrich id_map.csv
    id_map_file_path = '../data/id_map.csv'
    id_map = pd.read_csv(id_map_file_path)
    id_map_enriched = pd.merge(id_map, additional_info, on='sm_name', how='left')

    # Check for missing 'sm_name'
    missing_sm_names = id_map_enriched[id_map_enriched['SMILES'].isna() | id_map_enriched['Cluster'].isna()]['sm_name']
    if not missing_sm_names.empty:
        print("Missing sm_name in de_train_clustered.parquet:", missing_sm_names.tolist())

    # Load the fitted encoder
    encoder = joblib.load('../encoders/encoder.joblib')

    # Preprocess the enriched id_map
    X_id_map_tensor = preprocess_data(id_map_enriched, encoder).to(device)

    # Load trained autoencoder for feature extraction
    autoencoder = ComplexAutoencoder(X_id_map_tensor.shape[1], latent_size)
    autoencoder.load_state_dict(torch.load('../models/complex_autoencoder.pth', map_location=device))
    autoencoder.to(device).eval()

    # Generate latent features using the autoencoder
    with torch.no_grad():
        X_latent = autoencoder.encode(X_id_map_tensor)

    # Load the trained ComplexNet model
    model = ComplexNet(latent_size, 18211)
    model.load_state_dict(torch.load('../models/complex_net_model.pth', map_location=device))
    model.to(device).eval()

    # Make predictions with ComplexNet
    with torch.no_grad():
        predictions = model(X_latent)

    # Move the predictions to CPU and then convert to NumPy array
    predictions_cpu = predictions.cpu().numpy()

    # Format the output to match sample_submission.csv
    sample_submission_df = pd.read_csv('../data/sample_submission.csv')
    output_df = pd.DataFrame(predictions_cpu, columns=sample_submission_df.columns[1:])
    output_df.insert(0, 'id', id_map['id'])

    # Ensure the order of rows matches that of sample_submission.csv
    output_df = output_df.set_index('id').reindex(sample_submission_df['id']).reset_index()

    # Save the formatted output
    output_file_path = '../output/output_predictions.csv'
    output_df.to_csv(output_file_path, index=False)
    print('output file in output folder updated')


# Function to calculate row-wise mean root mean squared error
def calculate_mrrmse(y_actual, y_pred):
    rowwise_mse = torch.mean((y_actual - y_pred) ** 2, dim=1)
    rowwise_rmse = torch.sqrt(rowwise_mse)
    mrrmse = torch.mean(rowwise_rmse)
    return mrrmse


def convert_to_tensor(data, data_type=torch.float32, device='cpu'):
    """
    Converts data to a PyTorch tensor.
    """
    return torch.tensor(data, dtype=data_type).to(device)


def save_model(model, file_path):
    """
    Saves the model's state dictionary.
    """
    torch.save(model.state_dict(), file_path)


def load_model(model, file_path, device='cpu'):
    """
    Loads a model's state dictionary.
    """
    model.load_state_dict(torch.load(file_path, map_location=device))
    return model


def save_encoder(encoder, file_path):
    """
    Saves a fitted encoder.
    """
    joblib.dump(encoder, file_path)


def load_encoder(file_path):
    """
    Loads a saved encoder.
    """
    return joblib.load(file_path)


def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def log_evaluation_metrics(logger, metrics):
    """
    Logs evaluation metrics.
    """
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

# Additional utility functions can be added here
