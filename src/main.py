from data_preprocessing import preprocess_data
from data_warehouse import create_data_warehouse
from eda import run_eda
from clustering import run_clustering
from modeling import run_models

import os

os.makedirs("outputs/figures", exist_ok=True)

# Preprocessing, Feature Engineering, and Outlier Analysis
train, train_clean = preprocess_data(
    "dataset/train.csv",
    "outputs/cleaned_train.csv"
)

# Data Warehouse Creation
create_data_warehouse(train_clean)

# EDA, Correlation Analysis, and Feature Importance
run_eda(train)

# Clustering
train = run_clustering(train)

# Modeling
results = run_models(train)

print("\nPipeline Completed Successfully")
