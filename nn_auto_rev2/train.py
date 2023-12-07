import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from model import ComplexAutoencoder, ComplexNet


def train_model_simple_split(model, X, y, epochs, lr_rate, batch_size, test_size=0.2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    criterion = torch.nn.MSELoss()

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return model


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


# def train_with_cross_validation(model, data, target, epochs, lr_rate, batch_size, k_folds=5):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr_rate)
#     criterion = torch.nn.MSELoss()
#
#     kf = KFold(n_splits=k_folds, shuffle=True)
#     for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
#         train_data, val_data = data[train_idx], data[val_idx]
#         train_target, val_target = target[train_idx], target[val_idx]
#
#         train_loader = DataLoader(TensorDataset(train_data, train_target), batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(TensorDataset(val_data, val_target), batch_size=batch_size)
#
#         for epoch in range(epochs):
#             train_loss = train_model(model, train_loader, criterion, optimizer, device)
#             val_loss = evaluate_model(model, val_loader, criterion, device)
#             print(
#                 f'Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
#
#     return model

def train_with_cross_validation(model, X, y, epochs, lr_rate, batch_size, k_folds=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    criterion = torch.nn.MSELoss()

    # Convert X and y to PyTorch tensors if they aren't already
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    kf = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split the data for this fold
        train_data = X_tensor[train_idx]
        val_data = X_tensor[val_idx]
        train_target = y_tensor[train_idx]
        val_target = y_tensor[val_idx]

        train_loader = DataLoader(TensorDataset(train_data, train_target), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_target), batch_size=batch_size)

        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_model(model, val_loader, criterion, device)
            print(
                f'Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return model


def train_autoencoder(autoencoder, X, y, epochs, lr_rate, batch_size):
    """
    Train the ComplexAutoencoder model with cross-validation.
    """
    return train_with_cross_validation(autoencoder, X, y, epochs, lr_rate, batch_size)


def train_nn(neural_net, X, y, epochs, lr_rate, batch_size):
    """
    Train the ComplexNet model with cross-validation.
    """
    return train_with_cross_validation(neural_net, X, y, epochs, lr_rate, batch_size)

# Example usage in main.py or another script:
# autoencoder = ComplexAutoencoder(input_size, latent_size)
# trained_autoencoder = train_autoencoder(autoencoder, X, y, epochs, lr_rate, batch_size)

# neural_net = ComplexNet(latent_size, output_size)
# trained_neural_net = train_nn(neural_net, reduced_X, y, epochs, lr_rate, batch_size)
