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


class ComplexNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNet, self).__init__()
        # Reduced layer width
        self.layer1 = nn.Linear(input_size, 256)  # Reduced width
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Adjusted Transformer layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,  # Adjusted to match the width of layer2
            nhead=8,  # Reduced number of heads
            dim_feedforward=512,  # Reduced size
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        # Final output layer
        self.layer3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)

        x = x.unsqueeze(1).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).squeeze(1)

        x = self.layer3(x)
        return x
