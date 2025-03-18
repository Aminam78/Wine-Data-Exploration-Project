# src/main.py
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import load_and_prepare_data
df, feature_names, target = load_and_prepare_data()

# Load the wine dataset
wine = load_wine()
data = wine.data
feature_names = wine.feature_names
target = wine.target
target_names = wine.target_names

# Convert to DataFrame for easier exploration
df = pd.DataFrame(data, columns=feature_names)
df['target'] = target

# a: Display initial dataset information
print("Dataset Overview:")
print(f"Feature Names: {feature_names}")
print(f"Target Names: {target_names}")
print(f"Number of Samples: {data.shape[0]}")
print(f"Number of Features: {data.shape[1]}")
print("\nFirst few rows of the dataset:")
print(df.head())

# b: Explore the dataset
print("\nSummary Statistics:")
print(df.describe())

min_values = df[feature_names].min()
smallest_feature = min_values.idxmin()
smallest_value = min_values.min()
print(f"\nFeature with smallest values: {smallest_feature} (Minimum: {smallest_value})")

max_values = df[feature_names].max()
largest_feature = max_values.idxmax()
largest_value = max_values.max()
print(f"Feature with largest values: {largest_feature} (Maximum: {largest_value})")

ranges = max_values - min_values
largest_range_feature = ranges.idxmax()
largest_range = ranges.max()
print(f"Feature with largest range: {largest_range_feature} (Range: {largest_range})")

# c: Calculate correlation matrix
correlation_matrix = df[feature_names].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

strongest_correlation = correlation_matrix.abs().unstack().sort_values(ascending=False)
strongest_correlation = strongest_correlation[strongest_correlation != 1].head(1)
print(f"\nStrongest correlation (excluding self): {strongest_correlation}")

# d: PCA on unstandardized data
pca_unstandardized = PCA(n_components=2)
principal_components_unstandardized = pca_unstandardized.fit_transform(data)
print("\nExplained Variance Ratio (Unstandardized):", pca_unstandardized.explained_variance_ratio_)

pca_df_unstandardized = pd.DataFrame(data=principal_components_unstandardized, columns=['PC1', 'PC2'])
pca_df_unstandardized['target'] = target

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df_unstandardized['PC1'], pca_df_unstandardized['PC2'], c=pca_df_unstandardized['target'], cmap='viridis')
plt.colorbar(scatter, label='Target Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Unstandardized Wine Data (First Two Components)')
plt.savefig('output/unstandardized_pca.png')  # Save the plot
plt.close()

# e: Standardize the dataset and PCA on standardized data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

pca_standardized = PCA(n_components=2)
principal_components_standardized = pca_standardized.fit_transform(data_standardized)
print("\nExplained Variance Ratio (Standardized):", pca_standardized.explained_variance_ratio_)

pca_df_standardized = pd.DataFrame(data=principal_components_standardized, columns=['PC1', 'PC2'])
pca_df_standardized['target'] = target

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df_standardized['PC1'], pca_df_standardized['PC2'], c=pca_df_standardized['target'], cmap='viridis')
plt.colorbar(scatter, label='Target Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Standardized Wine Data (First Two Components)')
plt.savefig('output/standardized_pca.png')  # Save the plot
plt.close()

print("\nProject executed successfully! Plots are saved in the output folder.")