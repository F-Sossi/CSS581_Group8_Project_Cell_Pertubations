import torch.nn as nn

class ComplexNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNet, self).__init__()
        # Existing layers
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(256, 512)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.layer4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()

        # New layers for fine-tuning
        self.layer6 = nn.Linear(128, 128)
        self.relu6 = nn.ReLU()
        self.layer7 = nn.Linear(128, 64)
        self.relu7 = nn.ReLU()
        self.layer8 = nn.Linear(64, 64)
        self.relu8 = nn.ReLU()
        self.layer9 = nn.Linear(64, 32)
        self.relu9 = nn.ReLU()

        # Final output layer
        self.layer10 = nn.Linear(32, output_size)

    def forward(self, x):
        # Forward pass through existing layers
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.dropout1(x)
        x = self.relu3(self.layer3(x))
        x = self.dropout2(x)
        x = self.relu4(self.layer4(x))
        x = self.relu5(self.layer5(x))

        # Forward pass through new layers
        x = self.relu6(self.layer6(x))
        x = self.relu7(self.layer7(x))
        x = self.relu8(self.layer8(x))
        x = self.relu9(self.layer9(x))

        # Final output
        x = self.layer10(x)
        return x


