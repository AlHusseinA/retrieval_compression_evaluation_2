
import torch
import numpy as np


def retrieve(gallery_features, query_features, batch_size, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Epsilon value for numerical stability
    eps = 1e-6        


    # Type and dimension checks
    assert isinstance(gallery_features, (list, torch.Tensor, np.ndarray)), \
        "gallery_features must be a list, numpy array, or torch tensor."
    
    assert isinstance(query_features, (list, torch.Tensor, np.ndarray)), \
        "query_features must be a list, numpy array, or torch tensor."

    if isinstance(gallery_features, list):
        gallery_features = np.array(gallery_features)
    if isinstance(query_features, list):
        query_features = np.array(query_features)

    assert gallery_features.ndim == 2, "gallery_features must be a 2D array."
    assert query_features.ndim == 2, "query_features must be a 2D array."

    # Check for empty inputs
    assert len(gallery_features) > 0, "gallery_features is empty."
    assert len(query_features) > 0, "query_features is empty."



    # Move tensors to the specified device
    gallery_features = gallery_features.to(device) if isinstance(gallery_features, torch.Tensor) \
        else torch.tensor(gallery_features).to(device)
    
    query_features = query_features.to(device) if isinstance(query_features, torch.Tensor) \
        else torch.tensor(query_features).to(device)



    # Normalize the features
    query_norm = torch.nn.functional.normalize(query_features, p=2, dim=1, eps=eps)
    gallery_norm = torch.nn.functional.normalize(gallery_features, p=2, dim=1, eps=eps)


    # Calculate the similarity matrix
    similarity_matrix = torch.matmul(query_norm, gallery_norm.T)        

    # Sort the similarity matrix row-wise (descending order) and get original indices
    sorted_similarity_matrix, sorted_indices = torch.sort(similarity_matrix, descending=True, dim=1)

    return similarity_matrix, sorted_similarity_matrix, sorted_indices



def retrieve_one_image(gallery_features, single_query_feature, batch_size, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"


    # Type and dimension checks
    assert isinstance(gallery_features, (list, torch.Tensor, np.ndarray)), \
        "gallery_features must be a list, numpy array, or torch tensor."
    assert isinstance(single_query_feature, (list, torch.Tensor, np.ndarray)), \
        "single_query_feature must be a list, numpy array, or torch tensor."

    if isinstance(gallery_features, list):
        gallery_features = np.array(gallery_features)
    if isinstance(single_query_feature, list):
        single_query_feature = np.array(single_query_feature)

    assert gallery_features.ndim == 2, "gallery_features must be a 2D array."
    assert single_query_feature.ndim == 1, "single_query_feature must be a 1D array."

    # Check for empty inputs
    assert len(gallery_features) > 0, "gallery_features is empty."
    assert len(single_query_feature) > 0, "single_query_feature is empty."


    # Epsilon value for numerical stability
    eps = 1e-6     


    # Move tensors to the specified device
    gallery_features = gallery_features.to(device) if isinstance(gallery_features, torch.Tensor) \
        else torch.tensor(gallery_features).to(device)
    
    single_query_vector = single_query_feature.to(device) if isinstance(single_query_feature, torch.Tensor) \
        else torch.tensor(single_query_feature).to(device)

    # Normalize the single query vector and the gallery features
    query_norm = torch.nn.functional.normalize(single_query_vector.unsqueeze(0), p=2, dim=1, eps=eps)
    gallery_norm = torch.nn.functional.normalize(gallery_features, p=2, dim=1, eps=eps)

    # Calculate the similarity vector, it will be of size torch.Size([m]) due to the squeeze operation NOT torch.Size([1, m]) where m is the number of gallery features
    similarity_vector = torch.matmul(query_norm, gallery_norm.T).squeeze(0)

    # Sort the similarity scores in descending order and get original indices
    sorted_scores, sorted_indices = torch.sort(similarity_vector, descending=True)


    return similarity_vector, sorted_scores, sorted_indices

