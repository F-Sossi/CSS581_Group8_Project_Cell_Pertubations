import torch
import os
import pandas as pd
import data_processing as dp
import model as mdl
import training as tr
import evaluation as ev
import warnings


def main():

    warnings.filterwarnings("ignore")

    # Set device for training (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # Load and preprocess the data
    # data = dp.load_data('../data/de_train.parquet')
    # gene_columns = dp.get_gene_columns(data)
    # encoded_features, mean_features, std_features, X, y = dp.preprocess_data(data, gene_columns)

    preprocessed_data_path = '../data/preprocessed_data.pkl'

    # Check if the preprocessed data already exists
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data...")
        # Load the preprocessed data
        preprocessed_data = pd.read_pickle(preprocessed_data_path)
        gene_columns = preprocessed_data['gene_columns']
        encoded_features = preprocessed_data['encoded_features']
        mean_features = preprocessed_data['mean_features']
        std_features = preprocessed_data['std_features']
        X = preprocessed_data['X']
        y = preprocessed_data['y']
    else:
        print("Preprocessing data...")
        # Load and preprocess the data
        data = dp.load_data('../data/de_train.parquet')
        gene_columns = dp.get_gene_columns(data)
        encoded_features, mean_features, std_features, X, y = dp.preprocess_data(data, gene_columns)

        # Save the preprocessed data for future use
        preprocessed_data = {
            'gene_columns': gene_columns,
            'encoded_features': encoded_features,
            'mean_features': mean_features,
            'std_features': std_features,
            'X': X,
            'y': y
        }
        pd.to_pickle(preprocessed_data, preprocessed_data_path)

    # Initialize the model
    num_features = X.shape[1]
    num_labels = len(gene_columns)
    model = mdl.TransformerNN(num_features, num_labels).to(device)

    # Train the model
    model = tr.train_model(model, X, y, device)

    # Save the model
    model_save_path = '../models/y_trained_model.pth'
    tr.save_model(model, model_save_path)

    # Make predictions and evaluate (Optional)
    predictions = ev.make_predictions(model, X, device)
    ev.evaluate_model(model, X, y, device)  # Reusing the same function for evaluation

    # Visualize Predictions (Optional)
    ev.visualize_predictions(y, predictions)

    # Prepare combinations data for predictions
    X_combinations, combinations_data = dp.prepare_combinations_data('../data/id_map.csv', '../data/de_train.parquet')

    # Create submission file
    ev.create_submission(model, X_combinations, combinations_data, '../data/sample_submission.csv',
                         '../output/formatted_predictions.csv')


if __name__ == "__main__":
    main()
