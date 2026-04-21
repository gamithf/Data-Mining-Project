import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_data_warehouse(df):

    # =========================
    # 0. CREATE FOLDERS
    # =========================
    os.makedirs("outputs/warehouse", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # =========================
    # 1. DIMENSION TABLES
    # =========================

    # Location Dimension
    dim_location = df[['Neighborhood', 'MSZoning']].drop_duplicates().reset_index(drop=True)
    dim_location['location_id'] = dim_location.index + 1

    # Property Dimension
    dim_property = df[['BldgType', 'HouseStyle', 'OverallQual']].drop_duplicates().reset_index(drop=True)
    dim_property['property_id'] = dim_property.index + 1

    # Time Dimension
    dim_time = df[['YrSold', 'MoSold']].drop_duplicates().reset_index(drop=True)
    dim_time['time_id'] = dim_time.index + 1

    # =========================
    # 2. FACT TABLE
    # =========================

    fact = df.copy()

    # Merge to attach keys
    fact = fact.merge(dim_location, on=['Neighborhood', 'MSZoning'], how='left')
    fact = fact.merge(dim_property, on=['BldgType', 'HouseStyle', 'OverallQual'], how='left')
    fact = fact.merge(dim_time, on=['YrSold', 'MoSold'], how='left')

    # Select final columns
    fact_table = fact[[
        'SalePrice', 'GrLivArea', 'TotalSF',
        'OverallQual', 'HouseAge', 'GarageCars',
        'location_id', 'property_id', 'time_id'
    ]].copy()

    # Add primary key
    fact_table['fact_id'] = range(1, len(fact_table) + 1)

    # =========================
    # 3. SAVE TABLES
    # =========================

    dim_location.to_csv("outputs/warehouse/dim_location.csv", index=False)
    dim_property.to_csv("outputs/warehouse/dim_property.csv", index=False)
    dim_time.to_csv("outputs/warehouse/dim_time.csv", index=False)
    fact_table.to_csv("outputs/warehouse/fact_house_sales.csv", index=False)

    print("Data Warehouse Created Successfully")

    # =========================
    # 4. ANALYTICS (OLAP STYLE)
    # =========================

    # Merge for queries
    merged_loc = fact_table.merge(dim_location, on='location_id')
    merged_time = fact_table.merge(dim_time, on='time_id')
    merged_prop = fact_table.merge(dim_property, on='property_id')

    # =========================
    # 5. VISUALS
    # =========================

    # 🔹 1. Fact Table Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(fact_table['SalePrice'], bins=40, kde=True)
    plt.title("Fact Table: SalePrice Distribution")
    plt.xlabel("SalePrice")
    plt.tight_layout()
    plt.savefig("outputs/figures/fact_saleprice_distribution.png")
    plt.close()

    # 🔹 2. Avg Price by Neighborhood
    avg_price_loc = merged_loc.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12,6))
    avg_price_loc.head(10).plot(kind='bar')
    plt.title("Top 10 Neighborhoods by Average House Price")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/query_avg_price_neighborhood.png")
    plt.close()

    # 🔹 3. Price Trend by Year
    avg_price_year = merged_time.groupby('YrSold')['SalePrice'].mean()

    plt.figure(figsize=(8,5))
    avg_price_year.plot(marker='o')
    plt.title("Average House Price by Year")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig("outputs/figures/query_price_trend.png")
    plt.close()

    # 🔹 4. Avg Price by Property Type
    avg_price_prop = merged_prop.groupby('BldgType')['SalePrice'].mean()

    plt.figure(figsize=(8,5))
    avg_price_prop.plot(kind='bar')
    plt.title("Average Price by Property Type")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/query_property_price.png")
    plt.close()

    # 🔹 5. Count of Houses by Neighborhood
    count_loc = merged_loc['Neighborhood'].value_counts().head(10)

    plt.figure(figsize=(10,5))
    count_loc.plot(kind='bar')
    plt.title("Top 10 Neighborhoods by Number of Houses")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/neighborhood_counts.png")
    plt.close()

    print("Warehouse Analytics Visuals Created Successfully")
    