import torch
import torch.nn as nn

class ComplexNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNet, self).__init__()
        # Wider layers
        self.layer1 = nn.Linear(input_size, 512)  # Further increased width
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, 512)  # Consistent wide layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # Adjusted Transformer layer to match the increased width
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,  # Adjusted to match the width of layer2
            nhead=64,     # You can experiment with this number
            dim_feedforward=1024,  # Increased size of the feedforward network
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        # Final output layer
        self.layer3 = nn.Linear(512, output_size)  # Adjusted to the width

    def forward(self, x):
        # Forward pass through the wider layers
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)

        # Transformer layer
        x = x.unsqueeze(1).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).squeeze(1)

        # Final output
        x = self.layer3(x)
        return x


