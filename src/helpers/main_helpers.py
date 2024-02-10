
import torch
import os
from copy import deepcopy
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import pickle



def print_result_dict(results, feature_size):
    # Printing the content of results[feature_size] neatly
    print(f"Results for feature size '{feature_size}':\n")
    for key, value in results[feature_size].items():
        if isinstance(value, list):
            # Join list items into a string for neat display
            value_str = ', '.join([str(v) for v in value])
            print(f"{key}: [{value_str}]")
        else:
            print(f"{key}: {value}")
    print("\n")            

def save_retrieval_results(results, dir_file_name):
    with open(dir_file_name, 'wb') as file:
        pickle.dump(results, file)




def plot_similarity_heatmap(similarity_matrix, title, dir):
    # Ensure the directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Move the tensor to CPU and convert to numpy, if it's on GPU
    if similarity_matrix.is_cuda:
        similarity_matrix = similarity_matrix.cpu()

    similarity_matrix_np = similarity_matrix.numpy()

    # Set the size of the figure
    plt.figure(figsize=(30, 30))

    # Create the heatmap
    ax = sns.heatmap(similarity_matrix_np, annot=False, cmap='viridis')

    # Add title and labels as needed
    ax.set_title(title)
    ax.set_xlabel("Gallery Features")
    ax.set_ylabel("Query Features")

    # Save the plot to the specified directory
    plt.savefig(os.path.join(dir, f"{title.replace(' ', '_')}.png"))

    # Show the plot (optional, you can remove this line if you only want to save the file)
    plt.show()



def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")




def check_size(model, input_size=(3, 224, 224)):
    """
    Check the size of the model output
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if the model is wrapped in DataParallel and unwrap if necessary
    if isinstance(model, torch.nn.DataParallel):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    model_copy = deepcopy(unwrapped_model).to(device)
    model_copy.feature_extractor_mode()

    # Generate a random input tensor of the specified size and move it to the same device as the model
    input_tensor = torch.randn(1, *input_size).to(device)
    output_size = model_copy(input_tensor).size()[1]

    # print(f"Input size: {input_size}")
    # print(f"Output size: {output_size}")

    return output_size

#########################################################################################

def print_and_display_label(dir_debug, tuples_list, unique_id, images_txt_dir):
    """
    Saves an image based on the unique_id in the specified debug directory.

    Parameters:
    - dir_debug (str): Directory where the figure will be saved.
    - tuples_list (list of tuples): List of tuples in the format (unique_id, label, "label_name").
    - unique_id (int): Unique ID of the image to process.
    - images_txt_dir (str): Directory where the 'images.txt' file is located.
    """

    # Step 1: Find the relevant tuple
    tuple_found = next((item for item in tuples_list if item[0] == unique_id), None)
    if not tuple_found:
        raise ValueError(f"No tuple found for unique_id {unique_id}")

    label, label_name = tuple_found[1], tuple_found[2]
    print(f"Label: {label}, Label Name: {label_name}")
    # Step 2: Read from images.txt and find the image path
    images_txt_path = os.path.join(images_txt_dir, 'images.txt')

    with open(images_txt_path, 'r') as file:
        for line in file:
            line_unique_id, img_path = line.split()
            if int(line_unique_id) == unique_id:
                # Correct the path construction
                full_img_path = os.path.join(images_txt_dir, 'images', label_name, os.path.basename(img_path))
                break
        else:
            raise ValueError(f"No image found for unique_id {unique_id} in images.txt")


    # Step 3: Load and save the image
    image = Image.open(full_img_path)
    plt.imshow(image)
    plt.title(f"Label: {label}, Label Name: {label_name}")
    plt.axis('off')

    # Step 4: Save the figure
    save_path = os.path.join(dir_debug, f"{unique_id}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Image saved at {save_path}")


# Example usage
# dir_debug = "/path/to/debug/dir"
# tuples_list = [(440, 8, '009.Brewer_Blackbird'), (442, 8, '009.Brewer_Blackbird')]  # and so on
# unique_id = 440
# images_txt_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb1/data/cub200/CUB_200_2011"

# print_and_display_label(dir_debug, tuples_list, unique_id, images_txt_dir)



######################################################


def get_label_names_with_ids(id_label_tuples, root_path):
    label_name_mapping = {}
    with open(os.path.join(root_path, 'CUB_200_2011', 'classes.txt')) as file:
        for line in file:
            label_number, label_name = line.split(' ', 1)
            # Adjust label number to 0-based indexing
            label_name_mapping[int(label_number) - 1] = label_name.strip()

    # Use the adjusted label name mapping
    return [(img_id, label, label_name_mapping[label]) for img_id, label in id_label_tuples]



# Usage Example:



def save_tensors_to_directory(tensors, directory):
    """
    Save multiple PyTorch tensors to specified directory.

    Args:
    tensors (dict): A dictionary where keys are filenames and values are tensors.
    directory (str): The path to the directory where tensors will be saved.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterate over the dictionary and save each tensor
    for filename, tensor in tensors.items():
        torch.save(tensor, os.path.join(directory, f'{filename}.pt'))


    # Dictionary of tensors with desired filenames as keys
    # tensors_to_save = {
    #     'gallery_labels': gallery_labels,  # 'gallery_labels' is the key, and the value is the tensor
    #     'query_labels': query_labels,  # 'query_labels' is the key, and the value is the tensor
    #     'ground_truths': ground_truths,  # 'ground_truths' is the key, and the value is the tensor
    #     'similarity_matrix': similarity_matrix,
    #     'sorted_similarity_matrix': sorted_similarity_matrix,
    #     'sorted_indices': sorted_indices
    # }
    # # Directory where you want to save the tensors
    # directory = '/home/alabutaleb/Desktop/confirmation/Retrieval_eval_baselines_experiment_gpu_0/debugging'

    # Call the function
    # save_tensors_to_directory(tensors_to_save, directory)

#



