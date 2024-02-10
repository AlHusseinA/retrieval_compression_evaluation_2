import os
import re
import torch

from torchvision.models import ResNet50_Weights
from helpers.main_helpers import print_result_dict, save_retrieval_results
from helpers.load_models import load_resnet50_convV2, load_resnet50_convV2_KD
from helpers.features_unittest import TestFeatureSize
from helpers.generate_save_features import generate_features
from retrieval.kamoh_metrics import evaluate_retrieval_Kam_Woh_topK_mAP#, evaluate_retrieval_Kam_Woh
from retrieval.retrieval_eval import retrieve
from compression.pca_fit_features import fit_pca_features
from compression.lda_fit_features import fit_lda_features
from compression.kpca_fit_features import fit_kpca_features







###########################################################################################################################################################
def run_retrieval_evaluation_LDA(vanilla_model, trainloader_cub200, testloader_cub200, COMPRESSED_VECTOR_DIMS_LDA, dir_retrieval_lda, dataset_name, num_classes_cub200, batch_size, lr, top_k, device):
    # receive data
    # generate features via vanilla models
    # fit lca on training features and labels
    # compress test features via lda
    # cycle
    results_LDA = {}


    # create train and test features once
    gallery_features_train, gallery_labels_train = generate_features(vanilla_model, trainloader_cub200, device)       # fit_lda_features() will send it to cpu before fitting
    gallery_features_test, gallery_labels_test = generate_features(vanilla_model, testloader_cub200, device)
    
    #####################################################################################
    # queries are the same as gallery
    query_features_test = gallery_features_test.clone().detach()
    gallery_features_test  = gallery_features_test.cpu().data.numpy() # send to cpu so lda can work on it

    query_labels_test = gallery_labels_test.clone().detach() 

    query_features_test = query_features_test.cpu().data.numpy() # send to cpu so lda can work on it

    # gallery_labels_train is moved to cpu since it'll be used with gallery_features_train to fit the LDA model
    gallery_labels_train = gallery_labels_train.cpu().data.numpy()
    #####################################################################################
                        
    for compressed_size in COMPRESSED_VECTOR_DIMS_LDA: # remember that compression has to be less than the number of classes! So we'll only compress to 128, 64, 32, 16, 8

        # compressed_size will determine the compression of the 2048 features, compressed_size is the final size
        # so, we need to compress both gallery_test and query_test features
        lda_compressor, _ = fit_lda_features(gallery_features_train, gallery_labels_train, compressed_size) # fit on training features and return lda model
        # compress query and gallery features from test split. We only need to compress one since both are equal

        gallery_features_test_compressed = lda_compressor.transform(gallery_features_test)
        gallery_features_test_compressed = torch.from_numpy(gallery_features_test_compressed).float()
        query_features_test_compressed = gallery_features_test_compressed.clone().detach()

        # assert size of compressed features == compressed_size

        assert gallery_features_test_compressed.shape[1] == compressed_size, "The compressed features are not of the expected size"
        
        _, _, sorted_indices_compressed = retrieve(gallery_features_test_compressed, query_features_test_compressed, batch_size, device=device)

         # compressed_size = feature_size, remember that original uncompressed size will always be 2048 in this script - for the vanilla model
        mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices_compressed, query_labels_test, gallery_labels_test, dir_retrieval_lda, compressed_size, top_k, kd=None, temperature=None, pca=None, lda=True)


        results_LDA[compressed_size] = {        
            'mAP': mean_ap,
            f'mAP at top k {top_k}': mean_ap_top_k,
            'mean_recalls': mean_recalls,
            'mean_precisions': mean_precisions
        }

        print_result_dict(results_LDA, compressed_size)


    return results_LDA



###########################################################################################################################################################
def run_retrieval_evaluation_kPCA(vanilla_model, trainloader_cub200, testloader_cub200, COMPRESSED_VECTOR_DIMS, dir_retrieval_kpca, dataset_name, num_classes_cub200, batch_size, lr, top_k, device):
    # receive data
    # generate features via vanilla models
    # fit kpca on training features
    # compress test features via kpca
    # cycle
    results_kPCA = {}


    # create train and test features once
    gallery_features_train, _ = generate_features(vanilla_model, trainloader_cub200, device)       # fit_kpca_features() will send it to cpu before fitting
    gallery_features_test, gallery_labels_test = generate_features(vanilla_model, testloader_cub200, device)
    
    #####################################################################################
    # queries are the same as gallery
    query_features_test = gallery_features_test.clone().detach()
    gallery_features_test  = gallery_features_test.cpu().data.numpy() # send to cpu so kpca can work on it

    query_labels_test = gallery_labels_test.clone().detach() 
    query_features_test = query_features_test.cpu().data.numpy() # send to cpu so kpca can work on it

    #####################################################################################
                        
    for compressed_size in COMPRESSED_VECTOR_DIMS:

        # compressed_size will determine the compression of the 2048 features, compressed_size is the final size
        # so, we need to compress both gallery_test and query_test features
        kpca_compressor, _ = fit_kpca_features(gallery_features_train, compressed_size, whiten=False) # fit on training features and return kpca model
        # compress query and gallery features from test split. We only need to compress one since both are equal

        gallery_features_test_compressed = kpca_compressor.transform(gallery_features_test)
        gallery_features_test_compressed = torch.from_numpy(gallery_features_test_compressed).float()
        query_features_test_compressed = gallery_features_test_compressed.clone().detach()

        # assert size of compressed features == compressed_size

        assert gallery_features_test_compressed.shape[1] == compressed_size, "The compressed features are not of the expected size"
        
        _, _, sorted_indices_compressed = retrieve(gallery_features_test_compressed, query_features_test_compressed, batch_size, device=device)

         # compressed_size = feature_size, remember that original uncompressed size will always be 2048 in this script - for the vanilla model
        mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices_compressed, query_labels_test, gallery_labels_test, dir_retrieval_kpca, compressed_size, top_k, \
                                                                                                    kd=None, temperature=None, pca=None, kpca=True)


        results_kPCA[compressed_size] = {        
            'mAP': mean_ap,
            f'mAP at top k {top_k}': mean_ap_top_k,
            'mean_recalls': mean_recalls,
            'mean_precisions': mean_precisions
        }

        print_result_dict(results_kPCA, compressed_size)


    return results_kPCA

###########################################################################################################################################################



###########################################################################################################################################################
def run_retrieval_evaluation_PCA(vanilla_model, trainloader_cub200, testloader_cub200, COMPRESSED_VECTOR_DIMS, dir_retrieval_pca, dataset_name, num_classes_cub200, batch_size, lr, top_k, device):
    # receive data
    # generate features via vanilla models
    # fit pca on training features
    # compress test features via pca
    # cycle
    results_PCA = {}


    # create train and test features once
    gallery_features_train, _ = generate_features(vanilla_model, trainloader_cub200, device)       # fit_pca_features() will send it to cpu before fitting
    gallery_features_test, gallery_labels_test = generate_features(vanilla_model, testloader_cub200, device)
    
    #####################################################################################
    # queries are the same as gallery
    query_features_test = gallery_features_test.clone().detach()
    gallery_features_test  = gallery_features_test.cpu().data.numpy() # send to cpu so pca can work on it

    query_labels_test = gallery_labels_test.clone().detach() 
    query_features_test = query_features_test.cpu().data.numpy() # send to cpu so pca can work on it

    #####################################################################################
                        
    for compressed_size in COMPRESSED_VECTOR_DIMS:

        # compressed_size will determine the compression of the 2048 features, compressed_size is the final size
        # so, we need to compress both gallery_test and query_test features
        # pca_compressor, _ = fit_pca_features(gallery_features_train, compressed_size, whiten=False) # fit on training features and return pca model
        pca_compressor, _ = fit_pca_features(gallery_features_train, compressed_size, whiten=False) # fit on training features and return pca model

        # compress query and gallery features from test split. We only need to compress one since both are equal

        gallery_features_test_compressed = pca_compressor.transform(gallery_features_test)
        gallery_features_test_compressed = torch.from_numpy(gallery_features_test_compressed).float()
        query_features_test_compressed = gallery_features_test_compressed.clone().detach()

        # assert size of compressed features == compressed_size

        assert gallery_features_test_compressed.shape[1] == compressed_size, "The compressed features are not of the expected size"
        
        _, _, sorted_indices_compressed = retrieve(gallery_features_test_compressed, query_features_test_compressed, batch_size, device=device)

         # compressed_size = feature_size, remember that original uncompressed size will always be 2048 in this script - for the vanilla model
        mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices_compressed, query_labels_test, gallery_labels_test, dir_retrieval_pca, compressed_size, top_k, kd=None, temperature=None, pca=True)


        results_PCA[compressed_size] = {        
            'mAP': mean_ap,
            f'mAP at top k {top_k}': mean_ap_top_k,
            'mean_recalls': mean_recalls,
            'mean_precisions': mean_precisions
        }

        print_result_dict(results_PCA, compressed_size)


    return results_PCA

###########################################################################################################################################################




def run_retrieval_evaluation_KD(testloader_cub200, dir_kd_retrieval, load_dir_kd, dataset_name, num_classes_cub200, batch_size, lr, top_k, device):
    # Extract temperature values from folder names and sort them
    temperatures = []
    for folder_name in os.listdir(load_dir_kd):

        if folder_name.startswith("temperature_"):
            try:
                temperature = float(folder_name.split("_")[1])
                temperatures.append((temperature, folder_name))

            except ValueError:
                print(f"Invalid folder name format: {folder_name}")
    
    temperatures.sort(key=lambda x: x[0])  # Sort by temperature value


    results_kd = {}

    for temperature, folder_name in temperatures:
        folder_path = os.path.join(load_dir_kd, folder_name)
        print(f"\nProcessing folder: {folder_name}")
        print(f"Folder path: {folder_path}\n")
        # exit(f"#### END ####")
        # Extract models and sort by feature_size
        models = [model for model in os.listdir(folder_path) if model.endswith('.pth')]
        models.sort(key=lambda x: int(re.search("feature_size_(\d+)", x).group(1)))
        
        # Initialize an empty dictionary for this temperature
        if temperature not in results_kd:
            results_kd[temperature] = {}

        for model_name in models:
            # cnt+=1
            # Extract parameters from the model filename
            feature_size = int(re.search("feature_size_(\d+)", model_name).group(1))
            batch_size = int(re.search("batchsize_(\d+)", model_name).group(1))
            lr = float(re.search("lr_([0-9e\-]+(?:\.[0-9e\-]+)?)", model_name).group(1))

            # Construct the full path to the model file
            model_path = os.path.join(folder_path, model_name)

            # Load the model
            model_cub200 = load_resnet50_convV2_KD(num_classes_cub200, feature_size, dataset_name, batch_size, temperature, lr, model_path)

            gallery_features, gallery_labels = generate_features(model_cub200, testloader_cub200, device)
            query_features = gallery_features.clone().detach()
            query_labels = gallery_labels.clone().detach()     
            _, _, sorted_indices = retrieve(gallery_features, query_features, batch_size, device=device)
            mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices, query_labels, gallery_labels, dir_kd_retrieval, feature_size, top_k, kd=True, temperature=temperature, pca=None)

            # Save the results for this model in the nested dictionary
            results_kd[temperature][feature_size] = {
                'mean_ap': mean_ap,
                'mean_ap_top_k': mean_ap_top_k,
                'mean_recalls': mean_recalls,
                'mean_precisions': mean_precisions
            }

    save_retrieval_results(results_kd, f"{dir_kd_retrieval}/results_dic_all_KD_resnet50.pkl")

    return results_kd






###########################################################################################################################################################
def run_retrieval_evaluation_Vanilla(vanilla_model, testloader_cub200, dir_retrieval, batch_size, top_k, device):
    
    results = {}  # Initialize an empty dictionary to store results
    feature_size = "vanilla"
    vanilla_model = vanilla_model.to(device)
    vanilla_model.feature_extractor_mode()
    vanilla_model.eval()


    
    ##############################################################################
    gallery_features, gallery_labels = generate_features(vanilla_model, testloader_cub200, device)
    query_features = gallery_features.clone().detach()
    query_labels = gallery_labels.clone().detach()     
    _, _, sorted_indices = retrieve(gallery_features, query_features, batch_size, device=device)

    mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices, query_labels, gallery_labels, dir_retrieval, feature_size, top_k, kd=None, temperature=None, pca=None)


    results[feature_size] = {
    
        'mAP': mean_ap,
        f'mAP at top k {top_k}': mean_ap_top_k,
        'mean_recalls': mean_recalls,
        'mean_precisions': mean_precisions

    }

    print_result_dict(results, "vanilla")

    return results

#################################################################################################################################################

def run_retrieval_evaluation_baselines_models(testloader_cub200, feature_sizes, dir_retrieval, load_dir, dataset_name, num_classes_cub200, batch_size, lr, top_k, device):
    results = {}  # Initialize an empty dictionary to store results

    for feature_size in feature_sizes:
        model_cub200 = load_resnet50_convV2(num_classes_cub200, feature_size, dataset_name, batch_size, lr, load_dir)
        #unit test for feature size
        testing_size = TestFeatureSize(model_cub200, feature_size) # this will confirm that the feature size is correct

        model_cub200.feature_extractor_mode()
        model_cub200.eval()

        try:
            testing_size.test_feature_size()
            print(f"The model under evaluation is in indeed with {feature_size} feature size!")

        except AssertionError as e:
            # add an error message to the assertion error
            e.args += (f"Expected feature size {feature_size}, got {model_cub200.features_out.in_features}")   
            raise e # if the feature size is not correct, raise an error
        
        model_cub200 = model_cub200.to(device)
        ##############################################################################
        gallery_features, gallery_labels = generate_features(model_cub200, testloader_cub200, device)
        query_features = gallery_features.clone().detach()
        query_labels = gallery_labels.clone().detach()     
        _, _, sorted_indices = retrieve(gallery_features, query_features, batch_size, device=device)
        mean_ap, mean_ap_top_k, mean_recalls, mean_precisions = evaluate_retrieval_Kam_Woh_topK_mAP(sorted_indices, query_labels, gallery_labels, dir_retrieval, feature_size, top_k, kd=None, temperature=None, pca=None)


        results[feature_size] = {
        
            'mAP': mean_ap,
            f'mAP at top k {top_k}': mean_ap_top_k,
            'mean_recalls': mean_recalls,
            'mean_precisions': mean_precisions
        }

        print_result_dict(results, feature_size)


    return results














