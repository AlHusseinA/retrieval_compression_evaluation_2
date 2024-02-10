

#plot all baselines + vanilla
import matplotlib.pyplot as plt
import numpy as np
import json
import os



def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['val_acc']

def get_feature_size_from_filename(filename):
    parts = filename.split('feature_size_')
    if len(parts) > 1:
        feature_size = parts[1].split('_')[0].split('.')[0]  # Correctly handle file extension
        return feature_size  # Return 'vanilla' or the feature size directly
    return 'Unknown'


def sort_legend_entries(entries):
    # Explicitly handle 'Vanilla' entry and numeric entries separately
    vanilla_entry = [entry for entry in entries if 'Vanilla' in entry[0]]
    numeric_entries = [entry for entry in entries if 'Vanilla' not in entry[0]]

    # Correct the handling for sorting numeric entries
    # Ensure labels are correctly identified as numeric before conversion
    sorted_numeric_entries = sorted(numeric_entries, key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True)

    # Combine 'Vanilla' at the top and then the sorted numeric entries
    sorted_entries = vanilla_entry + sorted_numeric_entries
    return sorted_entries


def plot_val_accuracies(directory, title, xlabel, ylabel):
    plt.figure(figsize=(7, 5))
    lines = []  # Store (label, line object) tuples
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            val_acc = load_data(file_path)
            feature_size = get_feature_size_from_filename(filename)
            epochs = np.arange(len(val_acc))
            if feature_size == 'vanilla':
                line, = plt.plot(epochs, val_acc, label='Vanilla', linestyle='--', linewidth=2, color='red', marker='o', markersize=2)
            elif feature_size == '8':
                line, = plt.plot(epochs, val_acc, label='8', linestyle='--', linewidth=2, color='blue', marker='+', markersize=2)
            else:
                line, = plt.plot(epochs, val_acc, label=feature_size)
            lines.append((feature_size, line))
    
    # Sort lines according to the specified order and create the legend
    sorted_lines = sort_legend_entries(lines)
    plt.legend([line for _, line in sorted_lines], [label for label, _ in sorted_lines], title='Feature Size', title_fontsize='small', fontsize='x-small', loc='lower right')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    save_path = os.path.join(directory, 'validation_accuracy_vs_epochs_new_size.png')
    plt.savefig(save_path)



# Directory containing all the JSON files
directory = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_bl_vanilla'
# directory = "/home/alabutaleb/Desktop/confirmation/TEMP"

title = 'Validation Accuracy vs Epochs'
xlabel = 'Epochs'
ylabel = 'Accuracy'

plot_val_accuracies(directory, title, xlabel, ylabel)
