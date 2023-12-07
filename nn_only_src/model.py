import torch
import torch.nn as nn


class TransformerNN(nn.Module):
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.3):
        super(TransformerNN, self).__init__()
        self.num_target_encodings = 18211  # Will throw error if this changes
        self.num_sparse_features = num_features - self.num_target_encodings

        self.sparse_feature_embedding = nn.Linear(self.num_sparse_features, d_model)
        self.target_encoding_embedding = nn.Linear(self.num_target_encodings, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear(2 * d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, activation=nn.GELU(),
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        sparse_features = x[:, :self.num_sparse_features]
        target_encodings = x[:, self.num_sparse_features:]

        sparse_features = self.sparse_feature_embedding(sparse_features)
        target_encodings = self.target_encoding_embedding(target_encodings)

        combined_features = torch.cat((sparse_features, target_encodings), dim=1)
        combined_features = self.concatenation_layer(combined_features)
        combined_features = self.norm(combined_features)

        x = self.transformer(combined_features)
        x = self.norm(x)

        x = self.fc(x)
        return x
