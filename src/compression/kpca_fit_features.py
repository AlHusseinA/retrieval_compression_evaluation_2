import numpy as np
from sklearn.decomposition import KernelPCA
import torch

def fit_kpca_features(features, n_components, whiten=False, kernel='rbf', gamma=None, degree=3, coef0=1):
    # TODO:
    # add hyperparameter tuning
    """
    Applies Kernel PCA to reduce the dimensionality of the given feature tensor.

    Parameters:
    - features: PyTorch tensor of shape (number of images, feature_dimension)
    - n_components: int, the number of principal components to keep
    - kernel: string, the type of kernel to use in kPCA. Common options include 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
    - whiten: bool, whether to whiten the features (KernelPCA does not support whitening directly. This will be ignored.)
    - gamma: float, Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If None, defaults to 1/n_features
    - degree: int, Degree for the polynomial kernel
    - coef0: float, Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'

    Returns:
    - kpca: The trained kPCA model
    - kpca_features: The features transformed by kPCA
    """
    assert n_components > 0, "n_components must be a positive integer"
    # assert 
    # Convert the PyTorch tensor to a numpy array
    features_np = features.cpu().data.numpy()

    # Check for NaN or infinite values in the features
    if np.isnan(features_np).any() or np.isinf(features_np).any():
        raise ValueError("Features contain NaN or infinite values. Something went wrong.")

    # Print the shape of features for confirmation
    print("#####"*30)
    print(f"Required compression to {n_components} features with kernel='{kernel}'")
    print(f"Shape of features before kPCA: {features_np.shape}")

    # Apply Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, copy_X=True)
    kpca_features = kpca.fit_transform(features_np)

    # Note: Whiten is not directly supported in KernelPCA as of sklearn's last updates. Post-processing could be applied if necessary.

    # Print the shape of features after kPCA for confirmation
    print(f"Shape of features after kPCA: {kpca_features.shape}")
    print("#####"*30)

    return kpca, kpca_features
