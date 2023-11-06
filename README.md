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

### Requirments (add to these as we go)
pyarrow -> for file access
Pandas


# Dataset Description:

The data is derived from a novel experiment conducted on human Peripheral Blood Mononuclear Cells (PBMCs), treated with 144 compounds from the LINCS Connectivity Map dataset.
- The gene expression profiles were measured post 24-hour treatment, across three healthy human donors.
- The PBMCs are chosen due to their availability and relevance to diseases, and they contain various mature cell types like T-cells, B-cells, myeloid cells, and NK cells.
- The experiment includes additional multi-omic data to establish biological priors, aiming to explain gene perturbation responses in different biological contexts.

# Technical Details:

- The experiment involved plating PBMCs on 96-well plates, with designated wells for positive and negative controls, and the others for the 72 compounds.
- Gene expression data was collected and used to assign each cell to a particular cell type computationally.
- Cell Multiplexing, a process of pooling samples from each row into a single pool for sequencing, was employed, which introduces some technical bias.
- The competition focuses on modeling Differential Expression (DE) to estimate the impact of each compound on the expression level of every gene (18211 in total).

# Differential Expression Calculation:

- DE calculation involves averaging the raw gene expression counts in each cell type for each sample (pseudobulking) and fitting a linear model using Limma, including several covariates.
- The model estimates fold-change in gene expression and computes a p-value indicating the dependence of a gene's expression on the experimental compound.

# Task Description:

You are to predict DE values for Myeloid and B cells for a majority of compounds, training your model on data from all compounds in T cells and NK cells, and a small portion (10%) of compounds in Myeloid and B cells.

# Data Splits:

- Training: All compounds in T, NK cells plus 15 compounds with controls in B and myeloid cells.
- Public Test: 50 randomly selected compounds in B and myeloid cells.
- Private Test: 79 randomly selected compounds in B and myeloid cells.

# Files Provided:

- `de_train.parquet`: Main competition data, containing aggregated differential expression data.
- `adata_train.parquet` & `adata_obs_meta.csv`: Unaggregated count and normalized data, with observation metadata.
- `multiome_train.parquet`, `multiome_obs_meta.csv` & `multiome_var_meta.csv`: Optional additional 10x Multiome data at baseline.
- `id_map.csv`: Identifies the cell_type / sm_name pair to be predicted.
- `sample_submission.csv`: A sample submission file to guide you on the submission format.

# How to Proceed:

1. Familiarize yourself with the data, especially the structure and contents of the provided files.
2. Conduct Exploratory Data Analysis (EDA) to understand the distribution, trends, and relationships in the data.
3. Read up on the methods used for calculating differential expression, especially the Limma package and pseudobulking technique.
4. Develop a model to predict the DE values as instructed, ensuring to handle the technical biases and nuances discussed in the dataset description.
5. Split your data based on the provided split guidelines, and evaluate your model's performance using appropriate metrics.
6. Iterate on your model, tuning parameters, and perhaps exploring different modeling techniques to improve performance.
7. Engage with the Kaggle community to learn from others, and maybe team up with other competitors to leverage different expertise.


# Current plan

# Gene Expression Prediction Plan

## Objective
Predict the gene expression values for underrepresented cell types in `id_map.csv` based on data from well-represented cell types in `de_train.parquet`.

## Approach
1. Use the data from well-represented cell types to infer the behavior of underrepresented cell types when exposed to the same compounds.
2. Analyze the existing relationships between cell types in the data.
3. In order to predict the reactions of the B cells and the Myeloid cells what we need to do is add features to get similarities between Sm_names(chemicals) where we know how the B cells and the Myeloid cells reacted to chemicals that are similar in structure so we can infer the perturbations in unknown combinations of cell types and chemicals.

## Steps
### Data Preparation
- Load all necessary datasets.
- Clean and merge the datasets appropriately.
- Ensure all necessary features are included for the analysis.
- Add features for chemical similarity.

### Correlation Analysis
- Conduct a correlation analysis to understand how gene expression in well-represented cell types correlates with that in underrepresented cell types.
- This will require identifying overlapping compound treatments across cell types.

### Modeling Relationships
- Train models to predict gene expression for each well-represented cell type.
- Use these models to predict the gene expression for the same compounds in the underrepresented cell types.
- Analyze the predictions to identify patterns or relationships between responses.

### Inference
- Establish relationships and use them to infer gene expression for underrepresented cell types for compounds with missing data.

### Validation
- Validate the model using any available overlap in data where underrepresented cell types have been tested.

### Prediction
- Predict the response of underrepresented cell types to compounds with currently no data based on inferred relationships.

## Memory Efficiency Strategies
- Optimize data types to reduce memory footprint.
- Process data in chunks to avoid loading the entire dataset into memory.
- Use sparse data structures for datasets with many zeros or missing values.
- Calculate correlations in batches for subsets of genes to manage memory usage effectively.

## Next Steps
- Implement a more memory-efficient approach for correlation analysis.
- Calculate correlations in smaller batches to ensure the process does not exceed available system memory.

# Data Quality and Distribution Summary

## Duplicated Rows
- `de_train`: 0 duplicates
- `adata_obs_meta`: 239,470 duplicates

## Cell Type Distribution
### In `de_train`
- NK cells: 146
- T cells CD4+: 146
- T regulatory cells: 146
- T cells CD8+: 142
- B cells: 17
- Myeloid cells: 17

### In `id_map`
- B cells: 128
- Myeloid cells: 127

## Compound Name Discrepancies
### Present in `de_train` but not in `id_map`
- O-Demethylated Adapalene
- CHIR-99021
- Penfluridol
- Linagliptin
- Palbociclib
- Belinostat
- Dactolisib
- Dabrafenib
- Porcn Inhibitor III
- Idelalisib
- LDN 193189
- R428
- MLN 2238
- Alvocidib
- Crizotinib
- Foretinib
- Oprozomib (ONX 0912)

### Present in `id_map` but not in `de_train`
- None


