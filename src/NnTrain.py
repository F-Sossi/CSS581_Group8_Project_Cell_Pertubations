import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from NnNet import ComplexNet


# Load data from parquet file
data = pd.read_parquet('../data/de_train_clustered.parquet')

# Separate test set (4 samples each for 'B cells' and 'Myeloid cells')
test_set_b_cells = data[data['cell_type'] == 'B cells'].sample(4)
test_set_myeloid_cells = data[data['cell_type'] == 'Myeloid cells'].sample(4)
test_set = pd.concat([test_set_b_cells, test_set_myeloid_cells])

# Identify gene expression columns (assuming they are numerical)
gene_expression_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Exclude non-gene expression columns
non_gene_columns = ['cell_type', 'sm_name', 'Cluster', 'sm_links_id', 'control', 'SMILES']  # Add any other non-gene expression columns here
gene_expression_columns = [col for col in gene_expression_columns if col not in non_gene_columns]

# Select only the required columns for the train and test sets
train_set = data.drop(test_set.index)
test_set = test_set

# Prepare input features using OneHotEncoder
encoder = OneHotEncoder()
X_train = encoder.fit_transform(train_set[['cell_type', 'sm_name']]).toarray()
X_test = encoder.transform(test_set[['cell_type', 'sm_name']]).toarray()

joblib.dump(encoder, '../data/encoder.joblib')  # Save the encoder

# Prepare target variables (gene expressions)
y_train = train_set[gene_expression_columns].values
y_test = test_set[gene_expression_columns].values

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Rest of your neural network and training code...


# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Assuming X_train and y_train are your training data and labels
model = ComplexNet(X_train.shape[1], y_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluate on Test Set
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_mse = mean_squared_error(y_test.numpy(), predictions.numpy())
    test_rmse = test_mse ** 0.5
    test_mae = mean_absolute_error(y_test.numpy(), predictions.numpy())
    test_loss = criterion(predictions, y_test)

print(f'Test Loss: {test_loss.item()}')
print(f'Test MSE: {test_mse}')
print(f'Test RMSE: {test_rmse}')
print(f'Test MAE: {test_mae}')

# Convert predictions and actual values to pandas DataFrame
# Ensure to use the correct column names for all gene expression data
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

# After training the model
model_path = '../models/complex_net.pth'
torch.save(model.state_dict(), model_path)

