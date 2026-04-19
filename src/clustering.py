from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_clustering(train):
    features = train[['GrLivArea', 'OverallQual', 'SalePrice']]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    train['Cluster'] = kmeans.fit_predict(scaled)

    print("Cluster Counts:")
    print(train['Cluster'].value_counts())

    plt.figure(figsize=(8,6))
    plt.scatter(train['GrLivArea'], train['SalePrice'], c=train['Cluster'], cmap='viridis')
    plt.xlabel("GrLivArea")
    plt.ylabel("SalePrice")
    plt.title("House Clusters")
    plt.savefig("outputs/figures/clustering_plot.png")
    plt.close()

    return train
