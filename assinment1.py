#%%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#%%
# Load the Breast Cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

#%%
# Display the first and last 10 rows of the dataset
print("First 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))

#%%
# Check data types and for missing values
print("\nData Types and Missing Values:")
print(df.info())

### Explanation of Findings:
#Data Types:
# Most features are of type `float64`, which indicates that they are numerical and can be used for mathematical operations and analyses.
# The `target` column is of type `int64`, which suggests that it contains categorical values representing the class of cancer (e.g., malignant or benign).

#Missing Values:
#The `Non-Null Count` for all columns is 569, which means there are no missing values in the dataset. Each feature and the target variable have complete data for all 569 samples.

#%%
# Sampling Methods
## Random Sampling
random_sample = df.sample(n=150, random_state=42)
print("\nRandom Sample:")
print(random_sample.head())
#Fair Representation: Each sample has an equal chance of being included, which makes the sample unbiased with respect to the entire dataset.
#No Class Bias: Random sampling does not consider the distribution of classes (e.g., malignant vs benign), so the resulting sample may not represent the class distribution perfectly.

## Stratified Sampling
strat_sample, _ = train_test_split(df, test_size=0.7, stratify=df['target'], random_state=42)
print("\nStratified Sample:")
print(strat_sample.head())
#Preserves Class Proportions: Stratified sampling ensures that the proportion of malignant and benign cases in the sample reflects the proportion in the original dataset. This is critical for classification tasks where preserving class balance is important to avoid bias towards one class.
#Reduces Variance: Stratified sampling reduces variability between samples and ensures that the results are more stable and representative

## Systematic Sampling
step = len(df) // 150
systematic_sample = df.iloc[::step]
print("\nSystematic Sample:")
print(systematic_sample.head())
#Easy to Implement: Systematic sampling is straightforward and easier to implement than other methods, especially for large datasets.
#Risk of Bias: If there is a hidden pattern in the dataset (e.g., if it’s sorted in a certain way), systematic sampling could introduce bias. However, if the data is randomly ordered or unsorted, this method can work effectively.
#Even Distribution: This method ensures that the samples are spread evenly across the dataset, avoiding clusters of similar data points.

#%%
# Remove the target variable and create a correlation matrix
df_no_target = df.drop(columns=['target'])
corr_matrix = df_no_target.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
#Top 3 Pairs of Features with the Highest Correlation:
#Mean Radius and Mean Perimeter
#Correlation Coefficient: 0.997855
#Mean Radius and Mean Area
#Correlation Coefficient: 0.987357
#Mean Perimeter and Mean Area
#Correlation Coefficient: 0.98650
#The high correlations among variables in your data have several important implications for further analysis:

##Multicollinearity Issues:
#Impact: Leads to unstable coefficient estimates in regression models and makes it difficult to determine individual predictor effects.
#Actions: Use feature selection, regularization techniques (e.g., Ridge/Lasso), or tree-based models.

##Redundancy of Information:
#Impact: Causes inefficient models and increases computation time.
#Actions: Apply dimensionality reduction (e.g., PCA) or remove highly correlated features.

##Impact on Model Interpretability:
#Impact: Confounds interpretations, making it harder to assess feature influence.
#Actions: Simplify models or use explainable AI techniques like SHAP values.

##Overfitting Risk:
#Impact: Redundant features may lead to overfitting and poor generalization.
#Actions: Use cross-validation and feature engineering to reduce overfitting.
#%%
# Identify top 3 pairs of features with the highest correlation
top_corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
top_corr_pairs = top_corr_pairs[top_corr_pairs < 1]
print("\nTop 3 Pairs with Highest Correlation:")
print(top_corr_pairs.head(3))
#%%
# Normalize the dataset using different methods
scalers = {
    'Standardization': StandardScaler(),
    'Min-Max Scaling': MinMaxScaler(),
    'Robust Scaling': RobustScaler()
}

normalized_datasets = {}
for name, scaler in scalers.items():
    scaled_data = scaler.fit_transform(df_no_target)
    normalized_datasets[name] = pd.DataFrame(scaled_data, columns=df_no_target.columns)

# Compare distributions using histograms
for name, dataset in normalized_datasets.items():
    plt.figure(figsize=(12, 6))
    dataset.plot.hist(alpha=0.5, bins=30, title=f'Distribution after {name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
#Advantages and Disadvantages

##Standardization (Z-score Normalization)
#Advantages:
#Useful when the data has varying scales or is not bounded.
#Handles outliers better than Min-Max Scaling.
#Disadvantages:
#Assumes data is normally distributed.
#Does not bound the data to a specific range, which might be problematic for algorithms requiring normalized input.

##Min-Max Scaling
#Advantages:
#Transforms data to a bounded range [0, 1], making it easier to interpret.
#Suitable for algorithms that require a specific range.
#Disadvantages:
#Sensitive to outliers; a single outlier can distort the scaling.
#Can lead to poor performance if the data contains extreme values.

##Robust Scaling
#Advantages:
#Robust to outliers because it uses the median and interquartile range.
#Useful for data with significant outliers.
#Disadvantages:
#Less intuitive compared to Min-Max Scaling.
#The transformed data may not be in a [0, 1] range, which might not be suitable for some algorithms.

#%%
# Principal Component Analysis (PCA)
pca = PCA()
pca.fit(StandardScaler().fit_transform(df_no_target))
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Plot the cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Determine the number of components explaining at least 85% of the variance
n_components = np.argmax(cumulative_variance >= 0.85) + 1
print(f"\nNumber of Principal Components needed to explain at least 85% variance: {n_components}")
#Number of Principal Components needed to explain at least 85% variance: 6

#%%
# Load the standardized dataset and perform PCA
pca = PCA()
standardized_data = StandardScaler().fit_transform(df_no_target)
pca.fit(standardized_data)

# Explained variance by each principal component
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance by each principal component
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.xticks(range(1, len(explained_variance) + 1))
plt.show()

# Plot the scree plot (explained variance)
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.xticks(range(1, len(explained_variance) + 1))
plt.show()

#%%
# Apply t-SNE to the standardized dataset
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(standardized_data)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=df['target'], cmap='viridis', edgecolor='k')
plt.title('t-SNE Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Target')
plt.show()

# %%
##Principal Component Analysis (PCA):
#Strengths:
#Linear Relationships: PCA captures linear relationships and is effective for reducing dimensionality while preserving variance.
#Computational Efficiency: PCA is computationally less intensive and faster for high-dimensional datasets.
#Variance Explained: It provides insights into the variance explained by each principal component, which helps in understanding data distribution.
#Weaknesses:
#Linearity: PCA assumes linear relationships among features, which might not capture complex structures in the data.
#Interpretability: Principal components are linear combinations of the original features, which might be harder to interpret.

##t-Distributed Stochastic Neighbor Embedding (t-SNE):
#Strengths:
#Non-linearity: t-SNE is effective in capturing non-linear relationships and preserving local structure in the data.
#Visualization: It produces visually appealing 2D or 3D plots that can help in understanding clusters and data distribution.
#Weaknesses:
#Computational Intensity: t-SNE can be computationally expensive and slower, especially with large datasets.
#Interpretability: It doesn’t provide an easy way to understand variance explained, as it focuses on preserving local similarities rather than variance