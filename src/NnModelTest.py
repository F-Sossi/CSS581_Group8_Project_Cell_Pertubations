import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# Load data from parquet file
data = pd.read_parquet('../data/de_train_clustered.parquet')

# Filter for cluster #1
cluster_1 = data[data['Cluster'] == 1]

# Separate one 'B cell' and one 'Myeloid cell' for the test set
test_set_b_cells = cluster_1[cluster_1['cell_type'] == 'B cells'].sample(1)
test_set_myeloid_cells = cluster_1[cluster_1['cell_type'] == 'Myeloid cells'].sample(1)
test_set = pd.concat([test_set_b_cells, test_set_myeloid_cells])

# Ensure the training set has at least one of each cell type
train_set = cluster_1.drop(test_set.index)

# Check if there's at least one of each cell type in the training set
if train_set[train_set['cell_type'] == 'B cells'].empty or train_set[train_set['cell_type'] == 'Myeloid cells'].empty:
    raise ValueError("Training set lacks either B cells or Myeloid cells.")

# Define genes of interest for the neural network
genes_of_interest = ['A1BG', 'A1BG-AS1', 'A2M', 'A2M-AS1', 'A2MP1', 'A4GALT', 'AAAS', 'AACS', 'AAGAB', 'AAK1',
                     'AAMDC', 'AAMP', 'AAR2']

# Columns to keep
columns_to_keep = ['Cluster', 'cell_type', 'sm_name'] + genes_of_interest

# Select only the required columns for the train and test sets
train_set = train_set.loc[:, columns_to_keep]
test_set = test_set.loc[:, columns_to_keep]

# # Temporarily adjust display options
# pd.set_option('display.max_rows', None)
#
# # Print the train and test sets
# print("Train Set:")
# print(train_set)
# print("\nTest Set:")
# print(test_set)
#
# # Reset display options to default
# pd.reset_option('display.max_rows')
# Prepare training and testing data for neural network

# I want to train a neural network to predict the gene expression of the test set for col 'A1BG', 'A1BG-AS1', 'A2M',
# 'A2M-AS1', 'A2MP1', 'A4GALT', 'AAAS', 'AACS', 'AAGAB', 'AAK1', 'AAMDC', 'AAMP', 'AAR2' The newtwork should try to
# find a mapping from the gene expression of the other genes to the gene expression of these genes the cells of
# interest are the ones with cell_type = 'Myeloid cells' or 'B cells' the network should be able to predict the gene
# expression of these cells given only the cell_type and sm_name (chemical compound) as input


# Prepare input features using OneHotEncoder
encoder = OneHotEncoder()
X_train = encoder.fit_transform(train_set[['cell_type', 'sm_name']]).toarray()
X_test = encoder.transform(test_set[['cell_type', 'sm_name']]).toarray()

# Prepare target variables (gene expressions)
y_train = train_set[genes_of_interest].values
y_test = test_set[genes_of_interest].values

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)


class ComplexNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.dropout(x)
        x = self.relu3(self.layer3(x))
        x = self.layer4(x)
        return x


# Assuming X_train and y_train are your training data and labels
model = ComplexNet(X_train.shape[1], y_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(5000):
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
predictions_df = pd.DataFrame(predictions.numpy(), columns=genes_of_interest)
actual_df = pd.DataFrame(y_test.numpy(), columns=genes_of_interest)

# Add an identifier column to both DataFrames
actual_df['Type'] = 'Actual'
predictions_df['Type'] = 'Predicted'

# Reset the index and set 'Type' as part of the index
actual_df = actual_df.reset_index(drop=True).set_index(['Type'])
predictions_df = predictions_df.reset_index(drop=True).set_index(['Type'])

# Concatenate vertically
comparison_df = pd.concat([actual_df, predictions_df])

comparison_df.to_csv("../data/model_predictions_vs_actual.csv", index=False)
