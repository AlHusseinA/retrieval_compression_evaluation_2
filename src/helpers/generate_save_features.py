import os
import torch
from PIL import Image

def generate_features(model, loader, device):
    """ Generate features and labels for all images in the loader using the provided model. """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_features = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            batch_features = model(images)
            all_features.append(batch_features)  # Accumulate features
            all_labels.append(labels)  # Accumulate labels            

    # Concatenate all batch features and labels into single tensors
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels





def generate_single_image_features(feature_extractor, loader, unique_id, device, data_dir="/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb1/data/cub200"):
    feature_extractor.to(device)
    feature_extractor.eval()  # Set the model to evaluation mode

    split_file = os.path.join(data_dir, "CUB_200_2011", "train_test_split.txt")

    # Create lists for train and test unique IDs
    train_unique_ids, test_unique_ids = [], []
    with open(split_file, 'r') as file:
        for line in file:
            id, flag = map(int, line.split())
            if flag == 1:
                train_unique_ids.append(id)
            else:
                test_unique_ids.append(id)

    # Determine the split of the image and its index
    if unique_id in train_unique_ids:
        is_train_image = True
        image_index = train_unique_ids.index(unique_id)
    elif unique_id in test_unique_ids:
        is_train_image = False
        image_index = test_unique_ids.index(unique_id)
    else:
        raise ValueError(f"Image with unique ID {unique_id} does not belong to the train or test dataset.")

    # Ensure loader matches the image split
    if loader.dataset.train != is_train_image:
        raise ValueError(f"Loader split does not match the image with unique ID {unique_id}.")

    # Find the image in the loader
    current_idx = 0
    with torch.inference_mode():  # Disable gradient computation
        for images, labels in loader:  # Assuming the loader provides (images, labels)
            batch_size = images.size(0)
            if current_idx + batch_size > image_index:
                # The desired index is in this batch
                image_idx_in_batch = image_index - current_idx
                image = images[image_idx_in_batch].unsqueeze(0).to(device)  # Add batch dimension
                label = labels[image_idx_in_batch]

                features = feature_extractor(image)
                return features.squeeze(0), label  # Remove batch dimension and return with label

            current_idx += batch_size

    raise RuntimeError(f"No image found with unique ID {unique_id}.")