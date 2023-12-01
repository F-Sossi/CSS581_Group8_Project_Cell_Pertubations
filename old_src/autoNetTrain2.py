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
from imblearn.over_sampling import ADASYN
# Add this import to the others at the top of your script
from sklearn.preprocessing import LabelEncoder


def calculate_mrrmse(y_actual, y_pred):
    rowwise_mse = torch.mean((y_actual - y_pred) ** 2, dim=1)
    rowwise_rmse = torch.sqrt(rowwise_mse)
    mrrmse = torch.mean(rowwise_rmse)
    return mrrmse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data = pd.read_parquet('../data/de_train_clustered.parquet')

# drop sm_lincs_id column
data = data.drop('sm_lincs_id', axis=1)

# Separate the test set
test_b_cell = data[data['cell_type'] == 'B cells'].sample(1)
test_myeloid_cell = data[data['cell_type'] == 'Myeloid cells'].sample(1)
test_data = pd.concat([test_b_cell, test_myeloid_cell])
data = data.drop(test_data.index)

# Separate features and target variable
X = data.drop('cell_type', axis=1)
y = data['cell_type']
le = LabelEncoder()
y = le.fit_transform(y)

# Encoding categorical columns
categorical_columns = ['sm_name', 'Cluster', 'SMILES']
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[categorical_columns]).toarray()

# Save the encoder for later use
joblib.dump(encoder, '../encoders/encoder.joblib')

# Combine encoded categorical data with other features
X_combined = np.hstack((X_encoded, X.drop(categorical_columns, axis=1)))

# Apply ADASYN for oversampling
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X_combined, y)

# Split data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Load trained autoencoder for feature extraction
latent_size = 128
autoencoder = ComplexAutoencoder(X_train.shape[1], latent_size).to(device)
autoencoder.load_state_dict(torch.load('../models/complex_autoencoder.pth'))
autoencoder.eval()

# Generate latent features
with torch.no_grad():
    X_train_latent = autoencoder.encode(X_train_tensor)
    X_val_latent = autoencoder.encode(X_val_tensor)

# Initialize ComplexNet with latent features
model = ComplexNet(latent_size, y_train.shape[1]).to(device)

# Set up optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Early stopping criteria
early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0

# DataLoader for training data
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

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_outputs = model(X_batch)
            val_loss += torch.mean((val_outputs - y_batch) ** 2).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Save the trained model
torch.save(model.state_dict(), '../models/complex_net_model.pth')

# Test set preparation and evaluation
X_test_final = encoder.transform(test_data[categorical_columns]).toarray()
X_test_final_combined = np.hstack((X_test_final, test_data.drop(categorical_columns, axis=1)))
X_test_final_tensor = torch.tensor(X_test_final_combined, dtype=torch.float32).to(device)
y_test_final = test_data.drop(categorical_columns + ['cell_type'], axis=1).values
y_test_final_tensor = torch.tensor(y_test_final, dtype=torch.float32).to(device)
X_test_final_latent = autoencoder.encode(X_test_final_tensor)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_final_latent)
    test_loss = calculate_mrrmse(y_test_final_tensor, test_outputs)
    print(f'Test MRRMSE: {test_loss.item()}')
