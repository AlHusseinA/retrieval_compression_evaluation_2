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
from retrieval.run_retrieval_evaluation import run_retrieval_evaluation_LDA
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

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


    # dir_debug = "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_baselines_experiment_gpu_0/debugging/from_new_repo"



    #####################
    dataset_name="cub200"
    batch_size = 256
    lr = 7e-05
    top_k = 1000
    #####################
    print("###"*30)
    print(f"You are curently using {device} device")
    print("###"*30)



    dir_retrieval_lda = "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_LDA_evaluation"
    ensure_directory_exists(dir_retrieval_lda)


    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()
    num_classes_cub200, _, label_to_name_test = dataloadercub200.get_number_of_classes()
    _, test_image_ids = dataloadercub200.get_unique_ids()
    # get_label_names_with_ids() will return a list of tupels of the form [(unique id, label, "label_name")]
    # This is useful for the purpose of visualising the results later on.
    test_image_ids_labels_names = get_label_names_with_ids(test_image_ids, data_root)

    # # ##############################################################################

    # feature_size = "vanilla"

    # # vanilla finetuned resnet50
    feature_size_unmodifed = 2048
    vanilla_model = load_resnet50_unmodifiedVanilla(num_classes_cub200, feature_size_unmodifed, dataset_name, batch_size, lr, load_dir_vanilla)
    vanilla_model.feature_extractor_mode()
    vanilla_model.to(device)

    
    # ##############################################################################    
    COMPRESSED_VECTOR_DIMS_LDA = [128, 64, 32, 16, 8]               # compress to a maximum of number of classes - 1 


    ####################################################################################
    results_PCA = run_retrieval_evaluation_LDA(vanilla_model, trainloader_cub200, testloader_cub200, COMPRESSED_VECTOR_DIMS_LDA, dir_retrieval_lda, dataset_name, num_classes_cub200, batch_size, lr, top_k, device)

    save_retrieval_results(results_PCA, f"{dir_retrieval_lda}/results_dic_all_LDA_retrieval_eval.pkl")
    plot_metrics(results_PCA, top_k, COMPRESSED_VECTOR_DIMS_LDA, dir_retrieval_lda)
    plot_metrics_logarithmic(results_PCA, top_k, COMPRESSED_VECTOR_DIMS_LDA, dir_retrieval_lda)
    save_retrieval_results(results_PCA, f"{dir_retrieval_lda}/results_dic_LDA_resnet50.pkl")



    exit(f"########END########")




if __name__ == '__main__':
    main()