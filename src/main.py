from data_preprocessing import preprocess_data
from eda import run_eda
from clustering import run_clustering
from modeling import run_models

import os

os.makedirs("outputs/figures", exist_ok=True)

# Preprocessing
train = preprocess_data(
    "dataset/train.csv",
    "outputs/cleaned_train.csv"
)

# EDA
run_eda(train)

# Clustering
train = run_clustering(train)

# Modeling
results = run_models(train)

print("\nPipeline Completed Successfully")
