import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(model, X, y, device):
    """
    Evaluate the model's performance on the provided dataset by calculating Mean Squared Error (MSE)
    and R-squared (R2) score. The model predictions are compared with the true labels.

    Parameters:
    - model: The trained PyTorch model to evaluate.
    - X: The feature data as a NumPy array.
    - y: The true labels as a NumPy array.
    - device: The device on which to perform the evaluation (typically 'cpu' or 'cuda').

    Outputs:
    - Prints the MSE and R2 score to the console.
    """
    model = model.to(device)
    data = torch.tensor(X, dtype=torch.float32).to(device)
    labels = torch.tensor(y, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(data)

    # Calculate performance metrics
    mse = mean_squared_error(labels.cpu().numpy(), predictions.cpu().numpy())
    r2 = r2_score(labels.cpu().numpy(), predictions.cpu().numpy())
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")


def make_predictions(model, X, device):
    """
    Use the trained model to make predictions on the provided dataset.

    Parameters:
    - model: The trained PyTorch model for making predictions.
    - X: The feature data as a NumPy array on which predictions are to be made.
    - device: The device on which to perform the predictions (typically 'cpu' or 'cuda').

    Returns:
    - The predictions as a NumPy array.
    """
    model = model.to(device)
    data = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(data)
    return predictions.cpu().numpy()


def visualize_predictions(actual, predicted, num_samples=100):
    """
    Visualize the actual vs predicted values using a line plot.

    Parameters:
    - actual: The true labels as a NumPy array.
    - predicted: The predicted labels as a NumPy array.
    - num_samples: The number of samples to plot (default 100).

    Outputs:
    - A matplotlib plot showing the actual and predicted values.
    """
    plt.figure(figsize=(15, 5))
    # Flatten the arrays to ensure one-dimensional data for plotting
    actual_flat = actual.flatten()[:num_samples]
    predicted_flat = predicted.flatten()[:num_samples]
    plt.plot(actual_flat, label='Actual')
    plt.plot(predicted_flat, label='Predicted')
    plt.title('Comparison of Actual and Predicted Values')
    plt.legend()
    plt.show()


def create_submission(model, X_combinations, combinations_data, submission_format_path, save_path):
    """
    Create a submission file with predictions made by the model on the provided dataset.
    The predictions are formatted according to the sample submission file format.

    Parameters:
    - model: The trained PyTorch model for making predictions.
    - X_combinations: The feature data for which the submission is to be created.
    - combinations_data: The DataFrame containing the 'id' column for the submission.
    - submission_format_path: Path to the sample submission file to mimic its format.
    - save_path: Path where the formatted submission file will be saved.

    Outputs:
    - A CSV file saved to 'save_path' containing the formatted predictions.
    """
    model = model.to('cpu')
    model.eval()
    with torch.no_grad():
        predictions_combinations = model(torch.tensor(X_combinations, dtype=torch.float32))
    predictions_combinations = predictions_combinations.numpy()

    # Load the sample submission file to understand its format
    sample_submission = pd.read_csv(submission_format_path)

    # Format predictions for submission
    submission_df = pd.DataFrame(predictions_combinations, columns=sample_submission.columns[1:])
    submission_df.insert(0, 'id', combinations_data['id'])
    submission_df.to_csv(save_path, index=False)
