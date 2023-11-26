import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Load dataset
data = pd.read_parquet('../data/de_train.parquet')

print(data.shape)


# Function to convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    else:
        raise ValueError("Invalid SMILES string")


# Generate fingerprints
fingerprints = data['SMILES'].apply(smiles_to_fingerprint).tolist()

# Calculate pairwise similarity matrix
similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
for i in range(len(fingerprints)):
    for j in range(len(fingerprints)):
        similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])

# Clustering
clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                     distance_threshold=0.8)
data['Cluster'] = clustering.fit_predict(1 - similarity_matrix)

# Group by cluster and check conditions
grouped_data = data.groupby('Cluster')
cluster_conditions = {}

for cluster, cluster_data in grouped_data:
    b_cell_count = cluster_data[cluster_data['cell_type'] == 'B cells'].shape[0]
    myeloid_cell_count = cluster_data[cluster_data['cell_type'] == 'Myeloid cells'].shape[0]

    condition_1 = b_cell_count >= 2  # 2 or more B cells
    condition_2 = myeloid_cell_count >= 2  # 2 or more Myeloid cells
    condition_3 = condition_1 and condition_2  # both 1 and 2

    cluster_conditions[cluster] = {'Condition 1': condition_1, 'Condition 2': condition_2, 'Condition 3': condition_3}

# print the number of rows in each cluster
for cluster, cluster_data in grouped_data:
    print(f'Cluster {cluster}: {cluster_data.shape[0]} rows')

# print the conditions for each cluster
for cluster, conditions in cluster_conditions.items():
    print(f'Cluster {cluster}: {conditions}')

# Save the clustered data
data.to_parquet('../data/de_train_clustered.parquet')
