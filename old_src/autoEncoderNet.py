import torch.nn as nn

class ComplexAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(ComplexAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Latent Representation
        self.latent = nn.Linear(256, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        return x
