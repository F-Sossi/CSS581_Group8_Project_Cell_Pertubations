import pandas as pd
import torch
from NnNet import ComplexNet
import joblib


def preprocess_data(file_path, encoder):
    df = pd.read_csv(file_path)
    X = encoder.transform(df[['cell_type', 'sm_name']]).toarray()
    return torch.tensor(X, dtype=torch.float32)


# Load the fitted encoder for prediction
encoder = joblib.load('../data/encoder.joblib')

# Load and preprocess id_map.csv
id_map_file_path = '../data/id_map.csv'
X_id_map_tensor = preprocess_data(id_map_file_path, encoder)

# Load the trained model
model_path = '../models/complex_net_633.pth'
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
output_df.insert(0, 'id', pd.read_csv(id_map_file_path)['id'])

# Check for missing columns and add them with default values
for column in sample_submission_df.columns:
    if column not in output_df.columns:
        output_df[column] = 0  # or another appropriate default value

# Ensure the order of rows matches that of sample_submission.csv
output_df = output_df.set_index('id').reindex(sample_submission_df['id']).reset_index()

# Error handling to check the format
if not output_df.columns.equals(sample_submission_df.columns):
    print("Error: The format of the output predictions does not match sample_submission.csv")
else:
    # Save the formatted output only if formats match
    output_file_path = '../data/output_predictions.csv'
    output_df.to_csv(output_file_path, index=False)
