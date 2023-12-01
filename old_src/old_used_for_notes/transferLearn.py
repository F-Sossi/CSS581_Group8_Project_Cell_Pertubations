import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from NnNet2 import ComplexNet  # Import your neural network architecture

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a custom loss function
def custom_loss_function(outputs, targets):
    return torch.mean((outputs - targets) ** 2)

# Load data
data = pd.read_parquet('../data/de_train_clustered.parquet')

# Extract one B-cell and one Myeloid cell for testing
test_b_cell = data[data['cell_type'] == 'B cells'].sample(1)
test_myeloid_cell = data[data['cell_type'] == 'Myeloid cells'].sample(1)
test_data = pd.concat([test_b_cell, test_myeloid_cell])

# Remove these rows from the original dataset
data = data.drop(test_data.index)

# Identify gene expression columns and other relevant columns
gene_expression_columns = data.select_dtypes(include=[np.number]).columns.tolist()
non_gene_columns = ['cell_type', 'sm_name', 'sm_links_id', 'control', 'Cluster', 'SMILES']
gene_expression_columns = [col for col in gene_expression_columns if col not in non_gene_columns]
output_size = len(gene_expression_columns)

# Prepare input features and target variables
encoder = OneHotEncoder()
X = encoder.fit_transform(data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
input_size = X.shape[1]

y = data[gene_expression_columns].values

joblib.dump(encoder, '../data/encoder.joblib')  # Save the encoder for later use

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors and move them to the device
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# DataLoader for the training data
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

# Initialize and train the model
model = ComplexNet(X_train.shape[1], y_train.shape[1])
model.to(device)  # Move the model to the GPU
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(500):  # Adjust the number of epochs as needed this
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move batch data to the GPU
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = custom_loss_function(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = torch.mean((val_outputs - y_val) ** 2)  # Regular MSE for validation

    print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Save the trained model
torch.save(model.state_dict(), '../models/initial_complex_net.pth')

# ----------------- Transfer Learning for B-cells and Myeloid Cells -----------------

# Load the trained model
model = ComplexNet(input_size, output_size)
model.load_state_dict(torch.load('../models/initial_complex_net.pth'))
model.to(device)  # Move the model to the GPU

# Freeze layers up to layer5
for name, param in model.named_parameters():
    if 'layer6' in name or 'layer7' in name or 'layer8' in name or 'layer9' in name or 'layer10' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Prepare B-cells and myeloid cells data
b_myeloid_data = data[data['cell_type'].isin(['B cells', 'Myeloid cells'])]
X_b_myeloid = encoder.transform(b_myeloid_data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
y_b_myeloid = b_myeloid_data[gene_expression_columns].values

# Split this subset into training and validation
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_b_myeloid, y_b_myeloid, test_size=0.3, random_state=42)

# Convert new training and validation data to PyTorch tensors and move to the device
X_train_new, y_train_new = torch.tensor(X_train_new, dtype=torch.float32).to(device), torch.tensor(y_train_new, dtype=torch.float32).to(device)
X_val_new, y_val_new = torch.tensor(X_val_new, dtype=torch.float32).to(device), torch.tensor(y_val_new, dtype=torch.float32).to(device)

# DataLoader for the new training data
train_loader_new = DataLoader(TensorDataset(X_train_new, y_train_new), batch_size=8, shuffle=True)

# Adjust the optimizer for fine-tuning
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Fine-tuning training loop
for epoch in range(500):  # Adjust the number of epochs
    model.train()
    for X_batch, y_batch in train_loader_new:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move batch data to the GPU
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = custom_loss_function(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_new)
        val_loss = torch.mean((val_outputs - y_val_new) ** 2)  # Regular MSE for validation

    print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Save the fine-tuned model
torch.save(model.state_dict(), '../models/fine_tuned_complex_net.pth')

# At the end of the script, after training and fine-tuning
# Convert test_data to PyTorch tensors and move to the device
X_test_final = encoder.transform(test_data[['cell_type', 'sm_name', 'Cluster', 'SMILES']]).toarray()
y_test_final = test_data[gene_expression_columns].values
X_test_final = torch.tensor(X_test_final, dtype=torch.float32).to(device)
y_test_final = torch.tensor(y_test_final, dtype=torch.float32).to(device)

def calculate_mrrmse(y_actual, y_pred):
    # y_actual and y_pred are expected to be PyTorch tensors of the same shape
    rowwise_mse = torch.mean((y_actual - y_pred) ** 2, dim=1)
    rowwise_rmse = torch.sqrt(rowwise_mse)
    mrrmse = torch.mean(rowwise_rmse)
    return mrrmse

# ... [rest of the script]

# Evaluation on test_data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_final)
    test_loss = calculate_mrrmse(y_test_final, test_outputs)
    print(f'Test MRRMSE: {test_loss.item()}')
