
import numpy as np
import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall


def calculate_map_torchmetrics(similarity_matrix, sorted_similarity_matrix, sorted_indices, gallery_labels, query_labels):
    # Assertions to ensure input validity
    assert isinstance(similarity_matrix, torch.Tensor), "similarity_matrix must be a torch.Tensor"
    assert isinstance(sorted_similarity_matrix, torch.Tensor), "sorted_similarity_matrix must be a torch.Tensor"
    assert isinstance(gallery_labels, torch.Tensor), "gallery_labels must be a torch.Tensor"
    assert isinstance(query_labels, torch.Tensor), "query_labels must be a torch.Tensor"
    assert similarity_matrix.dim() == 2, "similarity_matrix must be a 2D tensor"
    assert sorted_similarity_matrix.dim() == 2, "sorted_similarity_matrix must be a 2D tensor"
    assert similarity_matrix.shape == sorted_similarity_matrix.shape, "similarity_matrix and sorted_similarity_matrix must have the same shape"
    try:
        num_queries = similarity_matrix.shape[0]
        num_gallery_items = similarity_matrix.shape[1]

        # Prepare the indexes tensor. Each query gets a unique index.
        indexes = torch.arange(num_queries).unsqueeze(1).expand(-1, num_gallery_items)

        # Create binary ground truth tensor for each query-gallery pair
        expanded_query_labels = query_labels.unsqueeze(1).expand(-1, num_gallery_items)  # Reshape for broadcasting
        expanded_gallery_labels = gallery_labels.unsqueeze(0).expand(num_queries, -1)  # Reshape for broadcasting
        # ground_truths = (query_labels == gallery_labels).int()
        ground_truths = (expanded_query_labels == expanded_gallery_labels).int()

        # print(f"####"*30)
        # print(f"sum of the first row of ground truths: {ground_truths[0].sum()}")
        # print(f"expanded_query_labels.shape: {expanded_query_labels.shape}")
        # # print(f"First 10 elements of expanded_query_labels: {expanded_query_labels[:10]}")
        # print(f"expanded_gallery_labels.shape: {expanded_gallery_labels.shape}")
        # # print(f"First 10 elements of expanded_gallery_labels: {expanded_gallery_labels[:10]}")
        # print(f"ground_truths.shape: {ground_truths.shape}")
        # print(f"First 10 elements of ground_truths: {ground_truths[:10][0]}")
        # print(f"sorted_similarity_matrix.shape: {sorted_similarity_matrix.shape}")
        # # print(f"First 10 elements of sorted_similarity_matrix: {sorted_similarity_matrix[:10]}")
        # print(f"sorted_indices.shape: {sorted_indices.shape}")
        # print(f"indexes.shape: {indexes.shape}")
        # # print(f"First 10 elements of indexes: {indexes[:10]}")
        # print(f"####"*30)

        # Initialize the RetrievalMAP metric
        retrieval_map_metric = RetrievalMAP()

        # Calculate mAP
        # map_score = retrieval_map_metric(sorted_similarity_matrix, ground_truths, indexes)#.compute()
        map_score = retrieval_map_metric(sorted_similarity_matrix, ground_truths, sorted_indices)#.compute()

        return map_score
        
    except Exception as e:
        raise RuntimeError("Error in computing mAP: " + str(e))

def calculate_recall_at_k(similarity_matrix, sorted_similarity_matrix, sorted_indices, gallery_labels, query_labels, k):
    # Validate inputs
    assert isinstance(similarity_matrix, torch.Tensor), "similarity_matrix must be a torch.Tensor"
    assert isinstance(sorted_similarity_matrix, torch.Tensor), "sorted_similarity_matrix must be a torch.Tensor"
    assert isinstance(sorted_indices, torch.Tensor), "sorted_indices must be a torch.Tensor"
    assert isinstance(gallery_labels, torch.Tensor), "gallery_labels must be a torch.Tensor"
    assert isinstance(query_labels, torch.Tensor), "query_labels must be a torch.Tensor"
    assert len(similarity_matrix.shape) == 2, "similarity_matrix must be a 2D tensor"
    assert similarity_matrix.shape == sorted_similarity_matrix.shape, "similarity_matrix and sorted_similarity_matrix must have the same shape"
    assert k > 0, "k must be a positive integer"
    try:
        # Number of queries and gallery items
        num_queries = similarity_matrix.shape[0]
        num_gallery_items = similarity_matrix.shape[1]

        # Prepare the indexes tensor
        indexes = torch.arange(num_queries).unsqueeze(1).expand(-1, num_gallery_items)

        # Create binary ground truth tensor for each query-gallery pair
        query_labels_expanded = query_labels.unsqueeze(1).expand(-1, num_gallery_items)  # Reshape for broadcasting
        gallery_labels_expanded = gallery_labels.unsqueeze(0).expand(num_queries, -1)  # Reshape for broadcasting
        ground_truths = (query_labels_expanded == gallery_labels_expanded).int()

        # Initialize the RetrievalRecall metric with top_k
        retrieval_recall_metric = RetrievalRecall(top_k=k)

        # Calculate Recall at K
        recall_at_k = retrieval_recall_metric(sorted_similarity_matrix, ground_truths, indexes)#.compute()
        # recall_at_k = retrieval_recall_metric(sorted_similarity_matrix, ground_truths, sorted_indices)#.compute()

        return recall_at_k

    except Exception as e:
        raise RuntimeError(f"Error in calculate_recall_at_k: {e}")


def calculate_map_single_query(similarity_vector, sorted_similarity_vector, sorted_indices, gallery_labels, query_label):
    # Assertions to ensure input validity
    assert isinstance(similarity_vector, torch.Tensor), "similarity_vector must be a torch.Tensor"
    assert isinstance(sorted_similarity_vector, torch.Tensor), "sorted_similarity_vector must be a torch.Tensor"
    assert isinstance(gallery_labels, torch.Tensor), "gallery_labels must be a torch.Tensor"
    assert isinstance(query_label, (int, float, torch.Tensor)), "query_label must be an int, float, or a torch.Tensor"
    assert similarity_vector.dim() == 1, "similarity_vector must be a 1D tensor"
    assert sorted_similarity_vector.dim() == 1, "sorted_similarity_vector must be a 1D tensor"
    assert similarity_vector.shape == sorted_similarity_vector.shape, "similarity_vector and sorted_similarity_vector must have the same shape"

    # Convert query_label to int if it's a tensor
    if isinstance(query_label, torch.Tensor):
        assert query_label.numel() == 1, "query_label tensor must have a single element"
        query_label = query_label.item()

    try:
        num_gallery_items = similarity_vector.shape[0]

        indexes = torch.zeros(num_gallery_items, dtype=torch.long)
        ground_truths = (gallery_labels == query_label).int()

        retrieval_map_metric = RetrievalMAP()

        map_score = retrieval_map_metric(sorted_similarity_vector, ground_truths, indexes)#.compute
        # map_score = retrieval_map_metric(sorted_similarity_vector, ground_truths, sorted_indices)
        return map_score

    except Exception as e:
        raise RuntimeError("Error in computing mAP for single query: " + str(e))


def calculate_recall_at_k_single_query(single_query_similarity, gallery_labels, sorted_indices, query_label, k):
    # Validate inputs
    assert isinstance(single_query_similarity, torch.Tensor), "single_query_similarity must be a torch.Tensor"
    assert isinstance(gallery_labels, torch.Tensor), "gallery_labels must be a torch.Tensor"
    assert isinstance(query_label, (int, torch.Tensor)), "query_label must be an int or a torch.Tensor"
    assert single_query_similarity.dim() == 1, "single_query_similarity must be a 1D tensor"
    assert gallery_labels.dim() == 1, "gallery_labels must be a 1D tensor"
    assert k > 0, "k must be a positive integer"

    try:
        # Number of gallery items
        num_gallery_items = single_query_similarity.shape[0]

        # Since we have only one query, indexes will be all zeros
        indexes = torch.zeros(num_gallery_items, dtype=torch.long)

        # Create binary ground truth tensor for the query-gallery pairs
        ground_truths = (gallery_labels == query_label).int()

        # Initialize the RetrievalRecall metric with top_k
        retrieval_recall_metric = RetrievalRecall(top_k=k)

        # Calculate Recall at K for the single query
        recall_at_k = retrieval_recall_metric(single_query_similarity, ground_truths, indexes)#.compute()
        # recall_at_k = retrieval_recall_metric(single_query_similarity, ground_truths, sorted_indices)#.compute()

        return recall_at_k

    except Exception as e:
        raise RuntimeError(f"Error in calculate_recall_at_k_single_query: {e}")





def average_precision(retrieved, relevant):
    """
    Calculate Average Precision for a single query.
    :param retrieved: List of retrieved item indices.
    :param relevant: List of relevant item indices.
    :return: Average Precision score.
    """
    retrieved = np.array(retrieved)
    relevant = np.array(relevant)
    rel_mask = np.in1d(retrieved, relevant)

    cum_rel = np.cumsum(rel_mask)
    precision_at_k = cum_rel / (np.arange(len(retrieved)) + 1)
    average_precision = np.sum(precision_at_k * rel_mask) / len(relevant)
    
    return average_precision

def mean_average_precision(retrieved_lists, relevant_lists):
    """
    Calculate Mean Average Precision (mAP) for a set of queries.
    :param retrieved_lists: List of lists, each containing retrieved item indices for a query.
    :param relevant_lists: List of lists, each containing relevant item indices for a query.
    :return: Mean Average Precision score.
    """
    ap_scores = [average_precision(retrieved, relevant)
                 for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
    
    return np.mean(ap_scores)


    # mean_average_precision(formatted_all_retrieved_indices, formatted_all_relevant_indices)



def mean_recall_at_k(predictions, retrieval_solution, k):
    """Computes mean recall at K for retrieval prediction.
    Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
            to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
            IDs.
        k: The number of top predictions to consider for calculating recall.
    Returns:
        mean_recall: Mean recall at K score (float).
    Raises:
        ValueError: If a test image in `predictions` is not included in
            `retrieval_solution`.
    """
    num_test_images = len(retrieval_solution.keys())
    total_recall = 0.0

    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError('Test image %s is not part of retrieval_solution' % key)

        relevant_items = set(retrieval_solution[key])
        top_k_predictions = prediction[:k]
        num_relevant_at_k = len([pred for pred in top_k_predictions if pred in relevant_items])
        recall_at_k = num_relevant_at_k / len(relevant_items) if relevant_items else 0
        total_recall += recall_at_k

    mean_recall = total_recall / num_test_images

    return mean_recall
