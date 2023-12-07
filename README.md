# CSS581_Group8_Project_Cell_Pertubations

https://www.kaggle.com/competitions/open-problems-single-cell-perturbations

File structure for the data folder

```
├── data
│   ├── adata_obs_meta.csv
│   ├── adata_train.parquet
│   ├── de_train.parquet
│   ├── id_map.csv
│   ├── multiome_obs_meta.csv
│   ├── multiome_train.parquet
│   ├── multiome_var_meta.csv
│   ├── open-problems-single-cell-perturbations.zip
│   └── sample_submission.csv
├── Data_exploration.ipynb
└── src
```

### Requirements (add to these as we go)
in requirements.txt to install run
```angular2html
pip install -r requirements.txt
```


### Repository for Predictive Modeling in Cellular Response Analysis

---

#### Model Architecture

Our predictive model, implemented in `model.py`, is a transformer-based neural network, `TransformerNN`, developed using PyTorch.

- **TransformerNN**: A subclass of PyTorch's `nn.Module`, `TransformerNN` features multi-head attention, customizable layers, and dropout rate. It's designed to capture cell responses to different chemical compounds.

- **Sparse Features & Target Encoding**: Distinct representations are used for target encoding and sparse features, encoding cell type and chemical interactions.

---

#### Training Process

Outlined in `training.py`, our training methodology includes:

- **Data Split**: Using sklearn's `train_test_split` to create training and validation sets.
- **Training Mechanics**: `train_model` function manages the training epochs, learning rate, and device setup.
- **Optimization & Learning Rate Adjustment**: Adam optimizer and PyTorch's `ReduceLROnPlateau` scheduler are used, along with the Huber loss function for stability and reduced outlier sensitivity.

Implemented in `nn_only.src`.

---

#### Second Approach: ComplexAutoencoder and ComplexNet

- **ComplexAutoencoder**: For dimensionality reduction, comprising an encoder, latent space, and decoder. Targets essential data features while preventing overfitting.
- **ComplexNet**: Utilizes latent space representations for predictions, integrating linear layers, ReLU activation, dropout, and a transformer encoder layer.

Training Process:

- **Autoencoder Training**: Focuses on optimizing latent space representation.
- **ComplexNet Training**: Concentrates on learning from reduced feature space after autoencoder training.

Integration Steps:

1. **Data Preparation**: Loading and preprocessing from `id_map.csv`.
2. **Model Setup**: Loading and setting `ComplexAutoencoder` and `ComplexNet` to evaluation mode.
3. **Feature Encoding & Prediction**: Encoding features and predicting latent space representation.
4. **Decoding and Gene Expression Prediction**: Using the decoder to predict gene expressions.
5. **Post-Processing**: Structuring predictions for submission.

Implemented in `nn_auto_rev2.src`.

---

#### Results and Evaluation

Our submission in the Kaggle competition:

- **Performance Metric**: Achieved a MRRMSE of 0.822, ranking 749th.
- **Benchmark**: The top score was 0.729 by N. Jean Kouagou.
- **Analysis**: Our performance was influenced by our first-time use of `ComplexAutoencoder` and `ComplexNet` and project time constraints.

---

#### Future Directions

- **Generative Adversarial Networks**: Exploring GANs for modeling cellular reactions.
- **Chemical Analysis Libraries**: Augmenting data processing with tools like RDKit.
- **Training and Tuning Improvements**: Advancing optimization techniques, loss functions, and network architectures.

---

This README documents our methods, results, and future plans in developing predictive models for cellular response analysis to chemical compounds.



