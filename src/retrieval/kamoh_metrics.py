from tqdm.auto import tqdm
import torch
import os



def evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices, query_labels, gallery_labels, dir_results, features_size, top_k=100, kd=None, temperature=None, pca=None, lda=None, kpca=None):
    # Basic input validation
    assert isinstance(sorted_indices, torch.Tensor), "sorted_indices must be a PyTorch Tensor"
    assert isinstance(query_labels, torch.Tensor), "query_labels must be a PyTorch Tensor"
    assert isinstance(gallery_labels, torch.Tensor), "gallery_labels must be a PyTorch Tensor"
    assert isinstance(top_k, int) and top_k > 0, "top_k must be a positive integer"
    assert isinstance(dir_results, str), "dir_results must be a string"

    # Ensure the lengths of the sorted_indices and query_labels match
    assert sorted_indices.shape[0] == len(query_labels), "Length of sorted_indices and query_labels must match"

    # Prepare directory to save results
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)    

    # if isinstance(features_size, str): # this will get triggered if we're evaluation the vanilla model since feature_size will contain the word 'vanilla'
    #     print(f"Parameter features_size a string {features_size}")
    #     print(f"Retrieval results for feature size {features_size} will be saved to {dir_results}\n")
    #     result_file_path = os.path.join(dir_results, f"retrieval_results_size_VANILLA_topK_{top_k}.txt")
    #     # print(f"the full dir+filename is: {result_file_path}\n")

    # elif isinstance(features_size, int): # this will get triggerd if we're evaluation baseline models as feature_size will be the size of the feature layer
    #     print(f"Parameter features_size an integer {features_size}")
    #     print(f"Retrieval results for feature size {features_size} will be saved to {dir_results}\n")
    #     result_file_path = os.path.join(dir_results, f"retrieval_results_size_{features_size}_topK_{top_k}.txt")
    #     # print(f"the full dir+filename is: {result_file_path}\n")

    # else:
    #     raise ValueError("features_size must be a positive integer >=1")

    if kd is not None and pca is None:
        result_file_path = os.path.join(dir_results, f"temperature_{temperature}_retrieval_results_size_{features_size}_topK_{top_k}_KD.txt")
    elif kd is None and pca is not None:
        result_file_path = os.path.join(dir_results, f"PCA_retrieval_results_compressed_size_{features_size}_topK_{top_k}.txt")
    elif kd is None and pca is None and lda is not None:
        result_file_path = os.path.join(dir_results, f"LDA_retrieval_results_compressed_size_{features_size}_topK_{top_k}.txt")
    elif kd is None and pca is None and lda is None and kpca is not None:
        result_file_path = os.path.join(dir_results, f"KPCA_retrieval_results_compressed_size_{features_size}_topK_{top_k}.txt")
    else:
        result_file_path = os.path.join(dir_results, f"retrieval_results_size_{features_size}_topK_{top_k}.txt")

    APs = []
    AP_top_ks = []
    Rs = [1, 5, 10, 20]
    recalls = {R: [] for R in Rs}
    precisions = {R: [] for R in Rs}

    # Set offset based on whether query & gallery are completely different
    offset = 1  # Adjust as needed

    for qi in tqdm(range(len(query_labels))):
        query_label = query_labels[qi]

        query_retrieved_indices_full = sorted_indices[qi][offset:]
        retrieved_labels_full = gallery_labels[query_retrieved_indices_full]

        # For mAP calculation, limit to top_k
        query_retrieved_indices_top_k = sorted_indices[qi][offset:offset+top_k] 
        retrieved_labels_top_k = gallery_labels[query_retrieved_indices_top_k]

        imatch_full = torch.eq(retrieved_labels_full, query_label)
        imatch_top_k = torch.eq(retrieved_labels_top_k, query_label)

        Lx_full = torch.cumsum(imatch_full, dim=0)
        Lx_top_k = torch.cumsum(imatch_top_k, dim=0)

        Px = Lx_full.float() / torch.arange(1, len(imatch_full)+1, 1).to(Lx_full)
        rel = torch.sum(imatch_full)  # number of relevant items
        ranking = Px * imatch_full  # this is to obtain the score of the matched item
        AP = ranking.sum() / rel.clamp(min=1)  # clamp is to avoid division by zero if no relevant item retrieved

        Px_top_k = Lx_top_k.float() / torch.arange(1, len(imatch_top_k)+1, 1).to(Lx_top_k)
        rel_top_k = torch.sum(imatch_top_k)
        ranking_top_k = Px_top_k * imatch_top_k
        AP_top_k = ranking_top_k.sum() / rel_top_k.clamp(min=1)  # Average precision for top_k
        
        Lx_for_recall = (Lx_full >= 1).float()
        # Lx_for_recall_top_k = (Lx_top_k >= 1).float()

        for R in Rs:
            rel_R = torch.sum(imatch_full[:R])
            recalls[R].append(Lx_for_recall[R - 1])
            precisions[R].append(rel_R / R)
        
        APs.append(AP)
        AP_top_ks.append(AP_top_k)

    APs = torch.tensor(APs)
    AP_top_ks = torch.tensor(AP_top_ks)

    recalls = {R: torch.tensor(recalls[R]) for R in Rs}
    precisions = {R: torch.tensor(precisions[R]) for R in Rs}

    mean_ap = APs.mean()
    mean_ap_top_k = AP_top_ks.mean()
    mean_recalls = {R: recalls[R].mean() for R in Rs}
    mean_precisions = {R: precisions[R].mean() for R in Rs}

    # Saving results to a file
    try:
        with open(result_file_path, 'w') as file:

            file.write(f"Retrieval results for feature size {features_size}\n\n")

            file.write(f'Mean Average Precision full:\n')
            file.write(f'{mean_ap.item()}\n\n')

            file.write(f'Mean Average Precision at top {top_k} (mAP@top_{top_k}):\n')
            file.write(f'{mean_ap_top_k.item()}\n\n')

            file.write('Recall:\n')
            for R in mean_recalls:
                file.write(f'R@{R}: {mean_recalls[R].item()}\n')

            file.write('\nPrecision:\n')
            for R in mean_precisions:
                file.write(f'P@{R}: {mean_precisions[R].item()}\n')
    except Exception as e:
        print(f"Exception: {e}")

    print(f'\nMean Average Precision full')
    print(mean_ap)

    print(f'\nMean Average Precision top_k {top_k}')
    print(mean_ap_top_k)

    print('Recall')
    for R in Rs:
        print(f'R@{R}', recalls[R].mean())
        
    print('Precision')
    for R in Rs:
        print(f'P@{R}', precisions[R].mean())


    return mean_ap, mean_ap_top_k, mean_recalls, mean_precisions



# # Helper function to calculate AP
# def calculate_ap(imatch):
#     Lx = torch.cumsum(imatch, dim=0)
#     Px = Lx.float() / torch.arange(1, len(imatch)+1, 1).to(Lx)

#     ranking = Px * imatch
#     AP = ranking.sum() / imatch.sum().clamp(min=1)  # Avoid division by zero
#     return AP


# def evaluate_retrieval_Kam_Woh(sorted_indices, query_labels, gallery_labels, dir_results, features_size):
#     # Basic input validation
#     assert isinstance(sorted_indices, torch.Tensor), "sorted_indices must be a PyTorch Tensor"
#     assert isinstance(query_labels, torch.Tensor), "query_labels must be a PyTorch Tensor"
#     assert isinstance(gallery_labels, torch.Tensor), "gallery_labels must be a PyTorch Tensor"
#     assert isinstance(dir_results, str), "dir_results must be a string"
#     # assert isinstance(features_size, int) and features_size > 0, "features_size must be a positive integer"

#     # Ensure the lengths of the sorted_indices and query_labels match
#     assert sorted_indices.shape[0] == len(query_labels), "Length of sorted_indices and query_labels must match"

#     # Prepare directory to save results
#     if not os.path.exists(dir_results):
#         os.makedirs(dir_results)
    
#     result_file_path = os.path.join(dir_results, f"retrieval_results_size_{features_size}.txt")

#     APs = []
#     Rs = [1, 5, 10, 20]
#     recalls = {R: [] for R in Rs}
#     precisions = {R: [] for R in Rs}

#     offset = 1  # ignore first one as the first one should be itself, but this depends on your evaluation, if query & gallery are completely different split, set this as 0

#     for qi in tqdm(range(len(query_labels))):

#         query_label = query_labels[qi]
#         query_retrieved_indices = sorted_indices[qi][offset:]  
#         retrieved_labels = gallery_labels[query_retrieved_indices]  # retrieve the label of the retrieved item
        
#         imatch = torch.eq(retrieved_labels, query_label)
#         imatch_sum = torch.sum(imatch)
#         Lx = torch.cumsum(imatch, dim=0)
#         Px = Lx.float() / torch.arange(1, len(imatch)+1, 1).to(Lx)
        
#         rel = torch.sum(imatch)  # number of relevant items
#         ranking = Px * imatch  # this is to obtain the score of the matched item
#         AP = ranking.sum() / rel.clamp(min=1)  # clamp is to avoid division by zero if no relevant item retrieved
#         # AP is the average precision formula
        
#         Lx_for_recall = (Lx >= 1).float()

#         for Ri, R in enumerate(Rs):  # R@1 R@5 R@10
#             rel_R = torch.sum(imatch[:R])
            
#             recalls[R].append(Lx_for_recall[R - 1])  # is the position at [R-1] 0 or 1? if 1, then we can retrieve an item within R, count as recall
#             precisions[R].append(rel_R / R)  # how accurate up to R items?
        
#         APs.append(AP)
        
#     APs = torch.tensor(APs)
#     recalls = {R: torch.tensor(recalls[R]) for R in Rs}
#     precisions = {R: torch.tensor(precisions[R]) for R in Rs}

#     mean_ap = APs.mean()
#     mean_recalls = {R: recalls[R].mean() for R in Rs}
#     mean_precisions = {R: precisions[R].mean() for R in Rs}

#     # Saving results to a file
#     with open(result_file_path, 'w') as file:
#         file.write(f"Retrieval results for feature size {features_size}\n\n")
#         file.write('Mean Average Precision feature_size:\n')
#         file.write(f'{mean_ap.item()}\n\n')
#         file.write('Recall:\n')
#         for R in mean_recalls:
#             file.write(f'R@{R}: {mean_recalls[R].item()}\n')
#         file.write('\nPrecision:\n')
#         for R in mean_precisions:
#             file.write(f'P@{R}: {mean_precisions[R].item()}\n')

#     print('\nMean Average Precision')
#     print(APs.mean())

#     print('Recall')
#     for R in Rs:
#         print(f'R@{R}', recalls[R].mean())
        
#     print('Precision')
#     for R in Rs:
#         print(f'P@{R}', precisions[R].mean())



#     return mean_ap, mean_recalls, mean_precisions

# Usage
# mean_ap, mean_recalls, mean_precisions = evaluate_retrieval(sorted_indices, query_labels, gallery_labels, k, "/path/to/dir_results", features_size)


