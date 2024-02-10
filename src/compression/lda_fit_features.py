import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import torch

def fit_lda_features(features, labels, n_components=None, solver='svd'):
    #### expirement with solver="eigen’ later if svd didn't work
    
    """
    Applies LDA to reduce the dimensionality of the given feature tensor based on the class labels.

    Parameters:
    - features: PyTorch tensor of shape (number of images, 2048)
    - labels: Array-like of shape (number of images,) class labels for each image
    - n_components: int or None, the number of components to keep. If None, will be set to min(n_classes - 1, n_features)
    - solver: str, the solver to use for the LDA computation. Options are 'svd' (default), 'lsqr', and 'eigen'.

    Returns:
    - lda: The trained LDA model
    - lda_features: The features transformed by LDA
    """
    if n_components is not None:
        assert n_components > 0, "n_components must be a positive integer"
        assert n_components <= labels.max() - 1, "n_components must be less than or equal to the number of classes"
    # Convert the PyTorch tensor to a numpy array
    features_np = features.cpu().data.numpy()

    # Check for NaN or infinite values in the features
    if np.isnan(features_np).any() or np.isinf(features_np).any():
        raise ValueError("Features contain NaN or infinite values. Something went wrong.")

    # Print the shape of features for confirmation
    print(f"#####"*30)
    print(f"Applying LDA for dimensionality reduction to {n_components if n_components is not None else 'n_classes - 1'} components")
    print(f"Shape of features before LDA: {features_np.shape}")

    # Apply LDA
    lda = LDA(n_components=n_components, solver=solver)
    lda_features = lda.fit_transform(features_np, labels)

    # Print the shape of features after LDA for confirmation
    print(f"Shape of features after LDA: {lda_features.shape}")
    print(f"#####"*30)

    return lda, lda_features
