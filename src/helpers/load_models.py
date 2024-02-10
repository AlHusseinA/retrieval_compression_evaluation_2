
import os
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from models.resnet50_conv_compressionV2 import ResNet50_convV2
from models.resnet50_vanilla import ResNet50_vanilla
from helpers.features_unittest import TestFeatureSize

from copy import deepcopy


def load_resnet50_convV2_KD(num_classes_cub200,feature_size, dataset_name, batch_size, temperature, lr, load_dir):
    model = ResNet50_convV2(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    # file name = resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e-05.pth
    # fine_tuned_weights = torch.load(f'{load_dir}/KD_student_resnet50_temperature_{temperature}_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
    fine_tuned_students_weights = torch.load(load_dir)
                                                # KD_student_resnet50_temperature_0.09_feature_size_8_cub200_batchsize_256_lr_7e-05.pth
    model.load_state_dict(fine_tuned_students_weights)

    #unit test for feature size
    test = deepcopy(model)
    testing_size = TestFeatureSize(test, feature_size) # this will confirm that the feature size is correct
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size! This is the correct vanilla size.")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {test.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    return model



def load_resnet50_convV2(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):
    model = ResNet50_convV2(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    # file name = resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e-05.pth
    fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
    model.load_state_dict(fine_tuned_weights)

    #unit test for feature size
    test = deepcopy(model)
    testing_size = TestFeatureSize(test, feature_size) # this will confirm that the feature size is correct
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size! This is the correct vanilla size.")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {test.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    return model

def load_resnet50_unmodifiedVanilla(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):

    model = ResNet50_vanilla(num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')
    # fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')

    print("loading ", model.load_state_dict(fine_tuned_weights))
    # model.load_state_dict(fine_tuned_weights)
    # model.feature_extractor_mode()
    #unit test for feature size
    test = deepcopy(model)
    testing_size = TestFeatureSize(test, feature_size) # this will confirm that the feature size is correct

    assert model is not None, "Failed to load the model"
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size!")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {test.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    

    return model