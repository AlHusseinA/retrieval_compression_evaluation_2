
# import os
# import sys
# print(f"####"*50)
# ROOT="/home/alabutaleb/Desktop/myprojects/debugging_mvp"
# sys.path.append(f"{ROOT}/src")
# print(f"{sys.path=}")
# print(f"{os.getcwd()=}")
# print(f"####"*50)
# import re
# import torch
# from retrieval.kamoh_metrics import evaluate_retrieval_Kam_Woh_topK_mAP
# from helpers.generate_save_features import generate_features
# from retrieval.retrieval_eval import retrieve
# from loaders.cub200loader import DataLoaderCUB200
# from helpers.main_helpers import save_retrieval_results
# from helpers.load_models import load_resnet50_convV2_KD


# def load_resnet50_models(testloader_cub200, directory, dir_retrieval_debug):

#     num_classes_cub200 = 200
#     dataset_name="cub200"
#     batch_size = 256
#     lr = 7e-05
#     top_k = 1000
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Extract temperature values from folder names and sort them
#     temperatures = []
#     for folder_name in os.listdir(directory):

#         if folder_name.startswith("temperature_"):
#             try:
#                 temperature = float(folder_name.split("_")[1])
#                 temperatures.append((temperature, folder_name))

#             except ValueError:
#                 print(f"Invalid folder name format: {folder_name}")
    
#     temperatures.sort(key=lambda x: x[0])  # Sort by temperature value
#     print(f"\n Temperatures found in the folders: {temperatures}\n")
#     print(f"Length of temperatures: {len(temperatures)}")

#     # exit(f"#### END ####")
#     results_kd = {}

#     for temperature, folder_name in temperatures:
#         folder_path = os.path.join(directory, folder_name)
#         print(f"\nProcessing folder: {folder_name}")
#         print(f"Folder path: {folder_path}\n")
#         # exit(f"#### END ####")
#         # Extract models and sort by feature_size
#         models = [model for model in os.listdir(folder_path) if model.endswith('.pth')]
#         models.sort(key=lambda x: int(re.search("feature_size_(\d+)", x).group(1)))


#         print(f"\n Models found in folder: {folder_name}:\n {models}\n")
        
#         # Initialize an empty dictionary for this temperature
#         if temperature not in results_kd:
#             results_kd[temperature] = {}

#         for model_name in models:
#             # cnt+=1
#             # Extract parameters from the model filename
#             feature_size = int(re.search("feature_size_(\d+)", model_name).group(1))
#             batch_size = int(re.search("batchsize_(\d+)", model_name).group(1))
#             lr = float(re.search("lr_([0-9e\-]+(?:\.[0-9e\-]+)?)", model_name).group(1))

#             # Construct the full path to the model file
#             model_path = os.path.join(folder_path, model_name)
#             print(f"Model path: {model_path}")
#             print(f"\nLoading model {model_name} with feature size {feature_size}, batch size {batch_size}, and learning rate {lr}")

#             # Load the model
#             model_cub200 = load_resnet50_convV2_KD(num_classes_cub200, feature_size, dataset_name, batch_size, temperature, lr, model_path)
#             print(f"Loaded model {model_name} with feature size {feature_size}, batch size {batch_size}, and learning rate {lr}")

#             gallery_features, gallery_labels = generate_features(model_cub200, testloader_cub200, device)
#             query_features = gallery_features.clone().detach()
#             query_labels = gallery_labels.clone().detach()     
#             _, _, sorted_indices = retrieve(gallery_features, query_features, batch_size, device=device)
#             mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices, query_labels, gallery_labels, dir_retrieval_debug, feature_size, top_k, kd=True, temperature=temperature, pca=None)
#             print(f"\nResults for model {model_name} with feature size {feature_size}, batch size {batch_size}, and learning rate {lr}")
#             print(f"Mean AP: {mean_ap}, Mean AP top {top_k}: {mean_ap_top_k}, Mean recalls: {mean_recalls}, Mean precisions: {mean_precisions}")
#             print(f"Saving results in: {dir_retrieval_debug}\n")

#             # Save the results for this model in the nested dictionary
#             results_kd[temperature][feature_size] = {
#                 'mean_ap': mean_ap,
#                 'mean_ap_top_k': mean_ap_top_k,
#                 'mean_recalls': mean_recalls,
#                 'mean_precisions': mean_precisions
#             }
#     save_retrieval_results(results_kd, f"{dir_retrieval_debug}/results_dic_KD_resnet50.pkl")

#         # if cnt== len(temperatures):
#         #     exit(f"#### END ####")


# data_root = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb1/data/cub200"

# load_dir_kd = f"/home/alabutaleb/Desktop/confirmation/kd_weights"   
# dir_retrieval_debug = "/home/alabutaleb/Desktop/confirmation/debug_test"
# dataloadercub200 = DataLoaderCUB200(data_root, batch_size=256, num_workers=10)
# _, testloader_cub200 = dataloadercub200.get_dataloaders()
# load_resnet50_models(testloader_cub200, load_dir_kd, dir_retrieval_debug)
