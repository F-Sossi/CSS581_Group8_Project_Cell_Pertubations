import torch.nn as nn


class ComplexAutoencoder(nn.Module):
    """
    ComplexAutoencoder is an autoencoder neural network used for reducing the feature space.
    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder.
        latent (nn.Linear): Layer representing the latent space.
        decoder (nn.Sequential): The decoder part of the autoencoder.
    """

    def __init__(self, input_size, latent_size):
        super(ComplexAutoencoder, self).__init__()
        self.encoder = self._create_encoder(input_size, latent_size)
        self.decoder = self._create_decoder(latent_size, input_size)
        self.latent = nn.Linear(256, latent_size)

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        """
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        """
        Encodes input into a reduced feature space (latent space representation).
        """
        x = self.encoder(x)
        x = self.latent(x)
        return x

    def _create_encoder(self, input_size, latent_size):
        """
        Create the encoder part of the autoencoder.
        """
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def _create_decoder(self, latent_size, output_size):
        """
        Create the decoder part of the autoencoder.
        """
        return nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)
        )


class ComplexNet(nn.Module):
    """
    ComplexNet is a neural network model that uses the output of an autoencoder as input.
    Attributes:
        layer1, layer2, layer3 (nn.Linear): Linear layers.
        relu1, relu2 (nn.ReLU): Activation functions.
        dropout1, dropout2 (nn.Dropout): Dropout layers.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder layer.
    """

    def __init__(self, input_size, output_size):
        super(ComplexNet, self).__init__()
        self.layer1 = self._create_layer(input_size, 256)
        self.layer2 = self._create_layer(256, 256)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=16, dim_feedforward=512, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)
        self.layer3 = nn.Linear(256, output_size)

    def forward(self, x):
        """
        Forward pass through the ComplexNet.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.unsqueeze(1).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).squeeze(1)
        x = self.layer3(x)
        return x

    def _create_layer(self, input_size, output_size):
        """
        Helper method to create a layer with Linear, ReLU, and Dropout.
        """
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
