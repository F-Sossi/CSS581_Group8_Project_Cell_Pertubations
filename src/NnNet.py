import torch.nn as nn


class ComplexNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNet, self).__init__()
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
        self.layer6 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.dropout1(x)
        x = self.relu3(self.layer3(x))
        x = self.dropout2(x)
        x = self.relu4(self.layer4(x))
        x = self.relu5(self.layer5(x))
        x = self.layer6(x)
        return x
