import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(train):
    plt.figure(figsize=(8,5))
    sns.histplot(train["SalePrice"], bins=50, kde=True)
    plt.title("SalePrice Distribution")
    plt.savefig("outputs/figures/saleprice_distribution.png")
    plt.close()

    # Correlation heatmap
    corr = train.corr(numeric_only=True)
    top_corr = corr["SalePrice"].abs().sort_values(ascending=False).head(15).index

    plt.figure(figsize=(10,8))
    sns.heatmap(train[top_corr].corr(), annot=True, cmap="coolwarm")
    plt.title("Top Correlated Features")
    plt.savefig("outputs/figures/correlation_heatmap.png")
    plt.close()

    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"]

    for col in features:
        plt.figure()
        sns.scatterplot(x=train[col], y=train["SalePrice"])
        plt.title(f"{col} vs SalePrice")
        plt.savefig(f"outputs/figures/scatter_{col}.png")
        plt.close()
