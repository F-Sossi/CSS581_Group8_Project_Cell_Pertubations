# Import required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from autoEncoderNet import ComplexAutoencoder

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_parquet('../data/de_train_clustered.parquet')

# drop sm_lincs_id column
data = data.drop('sm_lincs_id', axis=1)

# Extract one B-cell and one Myeloid cell for testing
test_b_cell = data[data['cell_type'] == 'B cells'].sample(1)
test_myeloid_cell = data[data['cell_type'] == 'Myeloid cells'].sample(1)
test_data = pd.concat([test_b_cell, test_myeloid_cell])

# Remove these rows from the original dataset
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

# Prepare input features
encoder = OneHotEncoder()
X = encoder.fit_transform(data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()

# For an autoencoder, input and output are the same
y = X

# Save the encoder for later use
joblib.dump(encoder, '../encoders/encoder_Auto.joblib')

# Split data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to PyTorch tensors and move them to the device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# DataLoader for the training data
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Initialize the autoencoder
latent_size = 512  # Adjust the latent size as needed
model = ComplexAutoencoder(X_train.shape[1], latent_size)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Early Stopping Criteria
early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0

# Training loop with improvements
for epoch in range(1000):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = torch.mean((outputs - y_batch) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

# Save the trained model
torch.save(model.state_dict(), '../models/complex_autoencoder.pth')

# Evaluation on the extracted test set
X_test_final = encoder.transform(test_data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
X_test_final = torch.tensor(X_test_final, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_final)
    test_loss = torch.mean((test_outputs - torch.tensor(X_test_final, dtype=torch.float32).to(device)) ** 2)
    print(f'Test Reconstruction Error: {test_loss.item()}')
