import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    train = pd.read_csv(input_path)

    print("Initial Shape:", train.shape)

    # Handling Missing Values
    drop_cols = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence']
    train.drop(columns=drop_cols, inplace=True)

    # Fill categorical with 'None' and numerical with median
    cat_cols = train.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        train[col] = train[col].fillna("None")

    # Fill numerical columns with median
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if train[col].isnull().sum() > 0:
            train[col] = train[col].fillna(train[col].median())

    print("Missing values handled")

    # Feature Engineering
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    train['HouseAge'] = train['YrSold'] - train['YearBuilt']
    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd']

    # Outlier Removal
    before = train.shape[0]
    train = train[train['GrLivArea'] < 4000]
    print(f"Removed {before - train.shape[0]} outliers")

    # Encoding
    train = pd.get_dummies(train)

    print("After Encoding:", train.shape)

    train.to_csv(output_path, index=False)

    return train
