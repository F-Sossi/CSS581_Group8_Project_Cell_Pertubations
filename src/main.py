import warnings
import concurrent.futures
from model import ComplexNet, ComplexAutoencoder
from train import train_autoencoder, train_nn
from utils import make_output_file


def run_training(ae_params, nn_params):
    # Unpack parameters for autoencoder and neural network
    ae_epochs, ae_lr_rate, ae_batch_size, ae_latent_size, ae_oversample = ae_params
    nn_epochs, nn_lr_rate, nn_oversample = nn_params

    # Train autoencoder
    train_autoencoder(ae_epochs, ae_lr_rate, ae_batch_size, ae_latent_size, ae_oversample)

    # Train neural network and get RMSE
    rmse = train_nn(nn_epochs, nn_lr_rate, ae_batch_size, ae_latent_size, nn_oversample)

    return (ae_params, nn_params, rmse)


def main():
    warnings.filterwarnings("ignore")
    # # Load and preprocess data
    # file_path = '../data/de_train_clustered.parquet'
    #
    # # Define ranges for each hyperparameter for the autoencoder
    # ae_epochs_options = [50, 100]
    # ae_lr_rate_options = [0.0001, 0.001]
    # ae_batch_size_options = [8, 16, 32, 64]
    # ae_latent_size_options = [128, 256, 512]
    # ae_oversample_options = [200]
    #
    # # Define ranges for each hyperparameter for the neural network
    # nn_epochs_options = [200, 100]
    # nn_lr_rate_options = [0.0001, 0.001]
    # nn_oversample_options = [200]
    #
    # best_rmse = float('inf')
    # best_params = {}
    # results = []
    #
    # # Create a list of all parameter combinations
    # all_params = [(ae_params, nn_params)
    #               for ae_params in
    #               zip(ae_epochs_options, ae_lr_rate_options, ae_batch_size_options, ae_latent_size_options,
    #                   ae_oversample_options)
    #               for nn_params in zip(nn_epochs_options, nn_lr_rate_options, nn_oversample_options)]
    #
    # # Using ProcessPoolExecutor to parallelize the grid search
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(run_training, ae_params, nn_params): (ae_params, nn_params) for ae_params, nn_params
    #                in all_params}
    #     for future in concurrent.futures.as_completed(futures):
    #         ae_params, nn_params, rmse = future.result()
    #         results.append((ae_params, nn_params, rmse))
    #         if rmse < best_rmse:
    #             best_rmse = rmse
    #             best_params = {'ae_params': ae_params, 'nn_params': nn_params}
    #
    # # Sort and print results
    # results.sort(key=lambda x: x[2])
    # for result in results:
    #     print(f"Params: {result[0]}, {result[1]}, RMSE: {result[2]}")
    #
    # print(f"Best RMSE: {best_rmse}")
    # print(f"Best Parameters: {best_params}")

    # Best Parameters: {'ae_params': (50, 0.0001, 8, 128, 200), 'nn_params': (200, 0.0001, 200)}



    # Load and preprocess data
    file_path = '../data/de_train_clustered.parquet'

    epochs = 50
    lr_rate = 0.0001
    batch_size = 8
    latent_size = 128
    oversample = 200
    train_autoencoder(epochs, lr_rate, batch_size, latent_size, oversample)

    n_epochs = 200
    lr_rate = 0.0001
    oversample = 200
    rmse = train_nn(n_epochs, lr_rate, batch_size, latent_size, oversample)

    print(f"RMSE: {rmse}")

    make_output_file(latent_size)


if __name__ == "__main__":
    main()
