import os
import pickle

def print_dictionary_structure(directory):
    pkl_file_path = None
    # Find the pkl file inside the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_file_path = os.path.join(root, file)
                break  # Stop after finding the first .pkl file
        if pkl_file_path:
            break

    if not pkl_file_path:
        print("No pkl file found in the directory.")
        return

    # Load the dictionary from the pkl file
    with open(pkl_file_path, "rb") as f:
        dictionary = pickle.load(f)

    # Improved function to print the structure of the dictionary
    def print_structure(d, indent=0):
        for key, value in d.items():
            print(f"{' ' * indent}{key}: {type(value).__name__}")
            if isinstance(value, dict):
                print_structure(value, indent + 4)

    print_structure(dictionary)

# Example usage
directory_path = "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_kd_evaluation"
print_dictionary_structure(directory_path)
