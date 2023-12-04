import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


def split_data(X, y, test_size=0.1):
    """
    Split data into training and validation sets.
    """
    return train_test_split(X, y, test_size=test_size)


def train_model(model, X, y, device, epochs=500, learning_rate=1e-2):
    """
    Train the model with the given data.
    """
    # Split the data
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Convert data to PyTorch tensors and move them to the device
    train_data = torch.tensor(X_train, dtype=torch.float32).to(device)
    val_data = torch.tensor(X_val, dtype=torch.float32).to(device)
    train_labels = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_labels = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Loss function and optimizer
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # Evaluate on validation data
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_labels)

        # Print training/validation loss
        print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

        # Step the scheduler
        scheduler.step(val_loss)

    return model


def save_model(model, save_path):
    """
    Save the model to a file.
    """
    torch.save({
        'model_state_dict': model.state_dict()
    }, save_path)
