import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from NnNet import ComplexNet  # Ensure this module has the definition for ComplexNet

# Define a custom weighted loss function
def weighted_mse_loss(outputs, targets, weights):
    weights = weights.view(-1, 1)  # Adjust the shape for broadcasting
    return torch.mean(weights * (outputs - targets) ** 2)

# Load data
data = pd.read_parquet('../data/de_train_clustered.parquet')

# Identify gene expression columns
gene_expression_columns = data.select_dtypes(include=[np.number]).columns.tolist()
non_gene_columns = ['cell_type', 'sm_name', 'sm_links_id', 'control', 'Cluster', 'SMILES']
gene_expression_columns = [col for col in gene_expression_columns if col not in non_gene_columns]

# Prepare input features and target variables
encoder = OneHotEncoder()
X = encoder.fit_transform(data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
y = data[gene_expression_columns].values
joblib.dump(encoder, '../data/encoder.joblib')

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Split data indices into training, validation, and test sets
train_indices, temp_indices = train_test_split(range(len(data)), test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# Assign weights to samples (example: higher weight for 'B cells' and 'Myeloid cells')
weights = torch.ones(len(data), 1)  # Adjust the shape for broadcasting
weights[data['cell_type'] == 'B cells'] = 2  # Adjust these weights based on your requirements
weights[data['cell_type'] == 'Myeloid cells'] = 2

# Subset the weights for the training data
train_weights = weights[train_indices]

# DataLoader with weights
train_loader = DataLoader(TensorDataset(X_train, y_train, train_weights), batch_size=32, shuffle=True)

# Initialize the model
model = ComplexNet(X_train.shape[1], y_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
early_stopping_patience = 100
min_val_loss = np.inf
patience_counter = 0

# Training loop with validation and weighted loss
for epoch in range(5000):
    model.train()
    for X_batch, y_batch, w_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = weighted_mse_loss(outputs, y_batch, w_batch)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = torch.mean((val_outputs - y_val) ** 2)  # Regular MSE for validation

    print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    # Early stopping check
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Save the trained model
torch.save(model.state_dict(), '../models/complex_net_633.pth')

# Evaluate on Test Set
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_mse = mean_squared_error(y_test.numpy(), predictions.numpy())
    test_rmse = test_mse ** 0.5
    test_mae = mean_absolute_error(y_test.numpy(), predictions.numpy())

print(f'Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test MAE: {test_mae}')

# Convert predictions and actual values to pandas DataFrame
predictions_df = pd.DataFrame(predictions.numpy(), columns=gene_expression_columns)
actual_df = pd.DataFrame(y_test.numpy(), columns=gene_expression_columns)

# Add an identifier column to both DataFrames
actual_df['Type'] = 'Actual'
predictions_df['Type'] = 'Predicted'

# Reset the index and set 'Type' as part of the index
actual_df = actual_df.reset_index(drop=True).set_index(['Type'])
predictions_df = predictions_df.reset_index(drop=True).set_index(['Type'])

# Concatenate vertically
comparison_df = pd.concat([actual_df, predictions_df])

comparison_df.to_csv("../data/model_predictions_vs_actual.csv", index=False)
