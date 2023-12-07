import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import ComplexAutoencoder, ComplexNet


def train_model_simple_split(model, X, y, epochs, lr_rate, batch_size, test_size=0.2):
    """
    Train a neural network model using a simple train-validation split.

    Args:
        model (torch.nn.Module): The neural network model to train.
        X (np.array): Input features.
        y (np.array): Target values.
        epochs (int): Number of training epochs.
        lr_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        test_size (float): Proportion of data to use for validation.

    Returns:
        model: The trained model.
    """
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
    """
    Train the model for one epoch. for testing

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.

    Returns:
        float: The average training loss for the epoch.
    """
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
    """
    Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_with_cross_validation(model, X, y, epochs, lr_rate, batch_size, k_folds=5, early_stopping_rounds=20,
                                verbose=True):
    """
    Train the model using K-fold cross-validation.

    Args:
        model (torch.nn.Module): The neural network model to train.
        X (np.array): Input features.
        y (np.array): Target values.
        epochs (int): Number of training epochs.
        lr_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        k_folds (int): Number of folds for cross-validation.
        early_stopping_rounds (int): Number of rounds for early stopping.
        verbose (bool): Whether to print progress messages.

    Returns:
        model: The trained model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=verbose)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    kf = KFold(n_splits=k_folds, shuffle=True)

    all_folds_results = []  # To store results from all folds

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_data = X_tensor[train_idx]
        val_data = X_tensor[val_idx]
        train_target = y_tensor[train_idx]
        val_target = y_tensor[val_idx]

        train_loader = DataLoader(TensorDataset(train_data, train_target), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_target), batch_size=batch_size)

        fold_results = {"train_loss": [], "val_loss": []}  # To store results for this fold
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            fold_results["train_loss"].append(train_loss)
            fold_results["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose:
                print(
                    f'Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            if epochs_no_improve >= early_stopping_rounds:
                if verbose:
                    print(
                        f'Early stopping triggered after {epochs_no_improve} epochs without improvement in fold {fold + 1}')
                break

        all_folds_results.append(fold_results)

    # After all folds
    # Analyze all_folds_results to determine the best hyperparameters

    return model


def train_final_model(model, X, y, epochs, lr_rate, batch_size, test_size=0.1, early_stopping_rounds=10, verbose=True):
    """
    Train the model using the entire dataset with a portion set aside for validation.

    Args:
        model (torch.nn.Module): The neural network model to train.
        X (np.array): Input features.
        y (np.array): Target values.
        epochs (int): Number of training epochs.
        lr_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        test_size (float): Proportion of data to use for validation.
        early_stopping_rounds (int): Number of rounds for early stopping.
        verbose (bool): Whether to print progress messages.

    Returns:
        model: The trained model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=verbose)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=test_size)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)

        # Use the validation loss for the learning rate scheduler and early stopping
        scheduler.step(val_loss)

        if verbose:
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_rounds:
            if verbose:
                print(f'Early stopping triggered after {epochs_no_improve} epochs without improvement')
            break

    return model


def train_autoencoder(autoencoder, X, y, epochs, lr_rate, batch_size):
    """
    Train the ComplexAutoencoder model.

    Args:
        autoencoder (ComplexAutoencoder): The autoencoder model to train.
        X (np.array): Input features for the autoencoder.
        y (np.array): Target values for the autoencoder.
        epochs (int): Number of training epochs.
        lr_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.

    Returns:
        autoencoder: The trained autoencoder model.
    """
    # return train_with_cross_validation(autoencoder, X, y, epochs, lr_rate, batch_size)
    return train_final_model(autoencoder, X, y, epochs, lr_rate, batch_size)


def train_nn(neural_net, X, y, epochs, lr_rate, batch_size):
    """
    Train the ComplexNet neural network model.

    Args:
        neural_net (ComplexNet): The ComplexNet model to train.
        X (np.array): Input features for the neural network.
        y (np.array): Target values for the neural network.
        epochs (int): Number of training epochs.
        lr_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.

    Returns:
        neural_net: The trained ComplexNet model.
    """
    # return train_with_cross_validation(neural_net, X, y, epochs, lr_rate, batch_size)
    return train_final_model(neural_net, X, y, epochs, lr_rate, batch_size)


