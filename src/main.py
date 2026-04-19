from data_preprocessing import preprocess_data
from eda import run_eda
from clustering import run_clustering
from modeling import run_models

import os

# Create folders if not exist
os.makedirs("outputs/figures", exist_ok=True)

# Step 1: Preprocessing
train = preprocess_data(
    "dataset/train.csv",
    "outputs/cleaned_train.csv"
)

# Step 2: EDA
run_eda(train)

# Step 3: Clustering
train = run_clustering(train)

# Step 4: Modeling
results = run_models(train)

print("\nPipeline Completed Successfully")
