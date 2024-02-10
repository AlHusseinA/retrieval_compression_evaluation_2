import numpy as np
from sklearn.decomposition import PCA
import torch

def fit_pca_features(features, n_components, whiten=False):
    """
    Applies PCA to reduce the dimensionality of the given feature tensor.

    Parameters:
    - features: PyTorch tensor of shape (number of images, 2048)
    - n_components: int, the number of principal components to keep
    - whiten: bool, whether to whiten the features

    Returns:
    - pca: The trained PCA model
    - pca_features: The features transformed by PCA
    """
    assert n_components > 0, "n_components must be a positive integer"

    # Convert the PyTorch tensor to a numpy array
    features_np = features.cpu().data.numpy()

    # Check for NaN or infinite values in the features
    if np.isnan(features_np).any() or np.isinf(features_np).any():
        raise ValueError("Features contain NaN or infinite values. Something went wrong.")

    # Print the shape of features for confirmation
    print(f"#####"*30)
    print(f"Required compression to {n_components} features")
    print(f"Shape of features before PCA: {features_np.shape}")

    # Apply PCA
    pca = PCA(n_components=n_components, whiten=whiten)
    pca_features = pca.fit_transform(features_np)

    # Print the shape of features after PCA for confirmation
    print(f"Shape of features after PCA: {pca_features.shape}")
    print(f"#####"*30)

    return pca, pca_features
