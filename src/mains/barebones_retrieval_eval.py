import os
import sys

print(f"####"*50)
ROOT="/home/alabutaleb/Desktop/myprojects/debugging_mvp"
sys.path.append(f"{ROOT}/src")
print(f"{sys.path=}")
print(f"{os.getcwd()=}")
print(f"####"*50)

import torch
import torch.nn as nn
import torch.nn.init as init
from helpers.features_unittest import TestFeatureSize

from datetime import datetime
import yaml
from loaders.cub200loader import DataLoaderCUB200
# from datasets.cub200 import Cub2011

import matplotlib.pyplot as plt
from retrieval.my_metrics import calculate_map_torchmetrics, calculate_map_single_query, calculate_recall_at_k, calculate_recall_at_k_single_query, mean_average_precision, average_precision, mean_recall_at_k
# from utils.debugging_functions import create_subset_data

from helpers.main_helpers import check_size, print_and_display_label, ensure_directory_exists,get_label_names_with_ids, plot_similarity_heatmap, save_tensors_to_directory, print_result_dict, save_retrieval_results
# from utils.helpers_function_kd import plot_performance
from helpers.retrieval_helpers import display_results_neatly, format_retrieved_indices, find_relevant_lists, plot_metrics, plot_metrics_logarithmic
from helpers.load_models import load_resnet50_convV2, load_resnet50_unmodifiedVanilla
from helpers.generate_save_features import generate_features, generate_single_image_features
from retrieval.retrieval_eval import retrieve, retrieve_one_image
from retrieval.kamoh_metrics import evaluate_retrieval_Kam_Woh_topK_mAP #, evaluate_retrieval_Kam_Woh
from retrieval.run_retrieval_evaluation import run_retrieval_evaluation_baselines_models, run_retrieval_evaluation_Vanilla
import numpy as np
import random

# for debugging
from torch.utils.data import Subset
import pickle




def main():
    # Set seed for reproducibility
    seed_value = 42  # can be any integer value
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

    #### set device #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Default to '0' if not set
    print(f"GPU ID: {gpu_id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    #####################
    ###### prep directories ######
    data_root = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb1/data/cub200"
    load_dir_baseline = f"/home/alabutaleb/Desktop/confirmation/baselines_allsizes/weights"   
    load_dir_vanilla = f"/home/alabutaleb/Desktop/confirmation"    # "resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth"
    dir_debug = "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_baselines_experiment_gpu_0/debugging/from_new_repo"

    # os.makedirs(dir_debug, exist_ok=True)
    ensure_directory_exists(dir_debug)

    print(f"Vanilla weights will be loaded from: {load_dir_vanilla}")
    print(f"Baseline weights will be loaded from: {load_dir_baseline}")
    #####################
    dataset_name = "cub200"

    #####################
    print("###"*30)
    print(f"You are curently using {device} device")
    print("###"*30)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    #### hyperparameters #####
    batch_size  = 256
    warmup_epochs=20 
  
    lr=0.00007 # best for baseline experiments
    weight_decay = 2e-05



    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    #### hyperparameters #####
    batch_size  = 256

    # use_early_stopping=True
    use_early_stopping=False

    # dir_retrieval = "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_baselines_experiment_gpu_0/retrieval_results_baselines_and_vanilla"
    dir_retrieval  = "/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_bl_vanilla_retrieval"

    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    _, testloader_cub200 = dataloadercub200.get_dataloaders()
    num_classes_cub200, _, label_to_name_test = dataloadercub200.get_number_of_classes()
    _, test_image_ids = dataloadercub200.get_unique_ids()
    # get_label_names_with_ids() will return a list of tupels of the form [(unique id, label, "label_name")]
    # This is useful for the purpose of visualising the results later on.
    test_image_ids_labels_names = get_label_names_with_ids(test_image_ids, data_root)





    # # ##############################################################################
    dataset_names="cub200"
    batch_size = 256
    lr = 7e-05
    top_k = 1000
    # feature_size = "vanilla"

    # vanilla finetuned resnet50
    feature_size_unmodifed = 2048
    vanilla_model = load_resnet50_unmodifiedVanilla(num_classes_cub200, feature_size_unmodifed, dataset_name, batch_size, lr, load_dir_vanilla)
    results_vanilla =  run_retrieval_evaluation_Vanilla(vanilla_model, testloader_cub200, dir_retrieval, batch_size, top_k, device)

    print_result_dict(results_vanilla, "vanilla")
    save_retrieval_results(results_vanilla, f"{dir_retrieval}/results_dic_vanilla_resnet50.pkl")

    # No plotting for vanilla model since it's only one model
    
    # exit(f"########END########")  
    # ##############################################################################    
    feature_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    dataset_names="cub200"
    batch_size = 256
    lr = 7e-05
    top_k = 1000
    results = run_retrieval_evaluation_baselines_models(testloader_cub200, feature_sizes, dir_retrieval, load_dir_baseline, dataset_names, num_classes_cub200, batch_size, lr, top_k, device)
    save_retrieval_results(results, f"{dir_retrieval}/results_dic_all_baselines_retrieval_eval.pkl")

    plot_metrics(results, top_k, feature_sizes, dir_retrieval)
    plot_metrics_logarithmic(results, top_k, feature_sizes, dir_retrieval)

    exit(f"########END########")

    ##############################################################################
    # Assuming test_image_ids_labels_names is obtained as before
    # unique_id = test_image_ids_labels_names[687][0]  # First element's unique ID
    # single_query_features, single_query_label = generate_single_image_features(vanilla_model, testloader_cub200, unique_id, device)
    ##############################################################################
    # gallery_features, gallery_labels = generate_features(vanilla_model, testloader_cub200, device)
    # query_features = gallery_features.clone().detach()
    # query_labels = gallery_labels.clone().detach()    
    ##############################################################################
    # print_and_display_label(dir_debug, test_image_ids_labels_names, unique_id, "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb1/data/cub200/CUB_200_2011")
    ##############################################################################
    # matched_tuple = next((item for item in test_image_ids_labels_names if item[0] == unique_id), None)

    # if matched_tuple is not None:
    #     print(f"Found tuple: {matched_tuple}\n")
    # else:
    #     print(f"No tuple found for unique_id {unique_id}\n")

    # print(f"unique_id: {unique_id}")
    # print(f"test_image_ids_labels_names[{unique_id}]:  {matched_tuple[2]}\n")


    # plot_similarity_heatmap(similarity_matrix, "heat map of similarity map", "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_baselines_experiment_gpu_0/debugging")
    # ##############################################################################

        ##############################################################################


    # insert code from the proxy paper

    ##############################################################################



    # relevant_lists =  find_relevant_lists(similarity_matrix, gallery_labels, query_labels)
    # retrieved_lists = format_retrieved_indices(sorted_indices) # reformatted_sorted_indices
    # map_score = mean_average_precision(retrieved_lists, relevant_lists)
    # r_at_1 = mean_recall_at_k(retrieved_lists, relevant_lists, k=1)
    # r_at_5 = mean_recall_at_k(retrieved_lists, relevant_lists, k=5)
    # r_at_10 = mean_recall_at_k(retrieved_lists, relevant_lists, k=10)   

# def mean_recall_at_k(predictions, retrieval_solution, k):
#     """Computes mean recall at K for retrieval prediction.
#     Args:
#         predictions: Dict mapping test image ID to a list of strings corresponding
#             to index image IDs.
#         retrieval_solution: Dict mapping test image ID to list of ground-truth image
#             IDs.
#         k: The number of top predictions to consider for calculating recall.
#     Returns:
#         mean_recall: Mean recall at K score (float).
#     Raises:
#         ValueError: If a test image in `predictions` is not included in
#             `retrieval_solution`."""

#     print(f"mAP using code from github: {map_score}")

    # ##############################################################################
    # similarity_matrix, sorted_similarity_matrix, sorted_indices = retrieve(gallery_features, query_features, batch_size, device=device)
    # ##############################################################################
    # map_score = calculate_map_torchmetrics(similarity_matrix, sorted_similarity_matrix, sorted_indices, gallery_labels, query_labels)
    # r_at_1 = calculate_recall_at_k(similarity_matrix, sorted_similarity_matrix, sorted_indices, gallery_labels, query_labels, k=1)
    # r_at_5 = calculate_recall_at_k(similarity_matrix, sorted_similarity_matrix, sorted_indices, gallery_labels, query_labels, k=5)
    # r_at_10 = calculate_recall_at_k(similarity_matrix, sorted_similarity_matrix, sorted_indices, gallery_labels, query_labels, k=10)

    # # ##############################################################################
    # display_results_neatly(map_score, r_at_1, r_at_5, r_at_10)
    # ##############################################################################
    # similarity_vector, sorted_scores, sorted_indices = retrieve_one_image(gallery_features, single_query_features, batch_size, device=device)
    # #############################################################################
    # map_score = calculate_map_single_query(similarity_vector, sorted_scores, sorted_indices, gallery_labels, single_query_label)
    # r_at_1 = calculate_recall_at_k_single_query(similarity_vector, gallery_labels, sorted_indices, single_query_label, k=1)
    # r_at_5 = calculate_recall_at_k_single_query(similarity_vector, gallery_labels, sorted_indices, single_query_label, k=5)  
    # r_at_10 = calculate_recall_at_k_single_query(similarity_vector, gallery_labels, sorted_indices, single_query_label, k=10)
    # ##############################################################################
    # display_results_neatly(map_score, r_at_1, r_at_5, r_at_10)
    # exit(f"########END########")


if __name__ == '__main__':
    main()