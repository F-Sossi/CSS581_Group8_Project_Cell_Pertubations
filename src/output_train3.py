import pandas as pd
import torch
from NnNet3 import ComplexNet
import joblib

def preprocess_data(df, encoder):
    X = encoder.transform(df[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
    return torch.tensor(X, dtype=torch.float32)

# Load the full dataset to extract 'SMILES' and 'Cluster'
full_data = pd.read_parquet('../data/de_train_clustered.parquet')
additional_info = full_data[['sm_name', 'SMILES', 'Cluster']].drop_duplicates()

# Load and enrich id_map.csv
id_map_file_path = '../data/id_map.csv'
id_map = pd.read_csv(id_map_file_path)

# Merge additional data into id_map
id_map_enriched = pd.merge(id_map, additional_info, on='sm_name', how='left')

# Check for missing 'sm_name'
missing_sm_names = id_map_enriched[id_map_enriched['SMILES'].isna() | id_map_enriched['Cluster'].isna()]['sm_name']
if not missing_sm_names.empty:
    print("Missing sm_name in de_train_clustered.parquet:", missing_sm_names.tolist())

# Load the fitted encoder and preprocess enriched id_map
encoder = joblib.load('../data/encoder.joblib')
X_id_map_tensor = preprocess_data(id_map_enriched, encoder)

# Load the trained model
model_path = '../models/complex_net_NN3.pth'
input_size = X_id_map_tensor.shape[1]
output_size = 18211  # Number of gene expressions
model = ComplexNet(input_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_id_map_tensor)

# Load sample_submission.csv to get the format
sample_submission_df = pd.read_csv('../data/sample_submission.csv')

# Format the output to match sample_submission.csv
output_df = pd.DataFrame(predictions.numpy(), columns=sample_submission_df.columns[1:])
output_df.insert(0, 'id', id_map['id'])

# Ensure the order of rows matches that of sample_submission.csv
output_df = output_df.set_index('id').reindex(sample_submission_df['id']).reset_index()

# Save the formatted output
output_file_path = '../data/output_predictions.csv'
output_df.to_csv(output_file_path, index=False)



