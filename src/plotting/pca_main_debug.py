# import os
# import sys
# print(f"####"*50)
# ROOT="/home/alabutaleb/Desktop/myprojects/debugging_mvp"
# sys.path.append(f"{ROOT}/src")
# print(f"{sys.path=}")
# print(f"{os.getcwd()=}")
# print(f"####"*50)
# import torch
# import numpy as np
# # Assume fit_pca_features is defined as per the previous discussions
# from compression.pca_fit_features import fit_pca_features

# def main():
#     # Load or generate your training features
#     # For demonstration, assume train_features is a PyTorch tensor of shape (num_train_images, 2048)
#     train_features = torch.randn(1000, 2048)  # Example data

#     # Specify PCA parameters
#     n_components = 32
#     whiten = True

#     # Apply PCA on training features
#     pca_model, train_pca_features = fit_pca_features(train_features, n_components, whiten)

#     # Assuming test_features is also a PyTorch tensor of shape (num_test_images, 2048)
#     test_features = torch.randn(500, 2048)  # Example data

#     # Convert test features to numpy array
#     test_features_np = test_features.cpu().data.numpy()

#     # Check for NaN or infinite values in the test features
#     if np.isnan(test_features_np).any() or np.isinf(test_features_np).any():
#         raise ValueError("Test features contain NaN or infinite values. Something went wrong.")

#     # Use the trained PCA model to transform the test features
#     test_pca_features = pca_model.transform(test_features_np)

#     # Now, train_pca_features and test_pca_features are ready for further analysis or machine learning tasks
#     print(f"Shape of train features before PCA: {train_features.shape}")
#     print(f"Shape of test features before PCA: {test_features.shape}")
#     print(f"Shape of PCA-transformed train features: {train_pca_features.shape}")
#     print(f"Shape of PCA-transformed test features: {test_pca_features.shape}")

# # Ensure this script block runs only if this script is executed as the main program
# if __name__ == "__main__":
#     main()
