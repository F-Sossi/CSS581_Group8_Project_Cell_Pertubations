import pandas as pd
import numpy as np
import torch
import joblib
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from NnNet3 import ComplexNet  # Import your neural network architecture
from autoEncoderNet import ComplexAutoencoder  # Import the autoencoder class


# Function to calculate row-wise mean root mean squared error
def calculate_mrrmse(y_actual, y_pred):
    rowwise_mse = torch.mean((y_actual - y_pred) ** 2, dim=1)
    rowwise_rmse = torch.sqrt(rowwise_mse)
    mrrmse = torch.mean(rowwise_rmse)
    return mrrmse


# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data = pd.read_parquet('../data/de_train_clustered.parquet')

# drop sm_lincs_id column
data = data.drop('sm_lincs_id', axis=1)

# Extract one B-cell and one Myeloid cell for testing
test_b_cell = data[data['cell_type'] == 'B cells'].sample(1)
test_myeloid_cell = data[data['cell_type'] == 'Myeloid cells'].sample(1)
test_data = pd.concat([test_b_cell, test_myeloid_cell])
data = data.drop(test_data.index)

# ---------------------------------------------------------------------
# oversampling
from sklearn.utils import resample

# Desired number of samples for each minority class
desired_number = 200

# Identifying the minority classes
b_cells = data[data['cell_type'] == 'B cells']
myeloid_cells = data[data['cell_type'] == 'Myeloid cells']

# Oversampling the minority classes
b_cells_oversampled = resample(b_cells, replace=True, n_samples=desired_number, random_state=42)
myeloid_cells_oversampled = resample(myeloid_cells, replace=True, n_samples=desired_number, random_state=42)

# Concatenating with the rest of the dataset
data_oversampled = pd.concat([data, b_cells_oversampled, myeloid_cells_oversampled])

data = data_oversampled

# ---------------------------------------------------------------------

# Identify gene expression columns and other relevant columns
gene_expression_columns = data.select_dtypes(include=[np.number]).columns.tolist()
non_gene_columns = ['cell_type', 'sm_name', 'sm_links_id', 'control', 'Cluster', 'SMILES']
gene_expression_columns = [col for col in gene_expression_columns if col not in non_gene_columns]

# Prepare input features and target variables
encoder = OneHotEncoder()
X = encoder.fit_transform(data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
y = data[gene_expression_columns].values

# Save the encoder for later use
joblib.dump(encoder, '../encoders/encoder.joblib')

# Split data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Load trained autoencoder (assuming it's already trained and saved)
latent_size = 512  # This should be the same value used during autoencoder training

# Load trained autoencoder for feature extraction
autoencoder = ComplexAutoencoder(X_train.shape[1], latent_size).to(device)
autoencoder.load_state_dict(torch.load('../models/complex_autoencoder.pth'))
autoencoder.eval()

# Use autoencoder to get latent features
with torch.no_grad():
    X_train_latent = autoencoder.encode(X_train_tensor)
    X_val_latent = autoencoder.encode(X_val_tensor)

# Initialize ComplexNet with latent features
model = ComplexNet(latent_size, len(gene_expression_columns)).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Early Stopping Criteria
early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0

# DataLoader for the training data
train_loader = DataLoader(TensorDataset(X_train_latent, y_train_tensor), batch_size=32, shuffle=True)

# Training loop with improvements
for epoch in range(400):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = torch.mean((outputs - y_batch) ** 2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:  # Assuming you have a validation data loader
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_outputs = model(X_batch)
            val_loss += torch.mean((val_outputs - y_batch) ** 2).item()

    avg_val_loss = val_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Learning Rate Scheduler Step
    scheduler.step(avg_val_loss)

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Save the trained ComplexNet model
torch.save(model.state_dict(), '../models/complex_net_model.pth')

# Evaluate on the test set
X_test_final = encoder.transform(test_data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
X_test_final_tensor = torch.tensor(X_test_final, dtype=torch.float32).to(device)
y_test_final = test_data[gene_expression_columns].values
y_test_final_tensor = torch.tensor(y_test_final, dtype=torch.float32).to(device)
X_test_final_latent = autoencoder.encode(X_test_final_tensor)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_final_latent)
    test_loss = calculate_mrrmse(y_test_final_tensor, test_outputs)
    print(f'Test MRRMSE: {test_loss.item()}')
