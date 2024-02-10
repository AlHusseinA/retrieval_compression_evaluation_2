# import matplotlib.pyplot as plt
# import numpy as np
# import json
# import os
##################################################################################
# this code works but the figures are little bit unreadable since all graphs are lumped together and there's no indation of temperature's effect

# def load_data(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data['val_acc']

# def get_feature_size_from_filename(filename):
#     # Handle 'student_size' for new files and identify the vanilla model
#     if 'student_size' in filename:
#         parts = filename.split('student_size_')
#         if len(parts) > 1:
#             feature_size = parts[1].split('_')[0]
#             return feature_size  # Returns the number indicating the student size
#     elif 'umodifiedVanilla' in filename:
#         return 'vanilla'  # Identifies the vanilla model
#     return 'Unknown'  # Fallback for filenames that don't match expected patterns


# def sort_legend_entries(entries):
#     # Explicitly handle 'Vanilla' entry and numeric entries separately
#     vanilla_entry = [entry for entry in entries if 'Vanilla' in entry[0]]
#     numeric_entries = [entry for entry in entries if 'Vanilla' not in entry[0]]

#     # Correct the handling for sorting numeric entries
#     # Ensure labels are correctly identified as numeric before conversion
#     sorted_numeric_entries = sorted(numeric_entries, key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True)

#     # Combine 'Vanilla' at the top and then the sorted numeric entries
#     sorted_entries = vanilla_entry + sorted_numeric_entries
#     return sorted_entries

# def plot_val_accuracies(directory, title, xlabel, ylabel):
#     # plt.figure(figsize=(7, 5))
#     plt.figure(figsize=(20, 15))

#     lines = []  # Store (label, line object) tuples
#     for filename in os.listdir(directory):
#         if filename.endswith('.json'):
#             file_path = os.path.join(directory, filename)
#             val_acc = load_data(file_path)
#             feature_size = get_feature_size_from_filename(filename)
#             epochs = np.arange(len(val_acc))
#             label = 'Vanilla' if feature_size == 'vanilla' else feature_size  # Adjust label for plotting
#             line, = plt.plot(epochs, val_acc, label=label, linestyle='--', linewidth=1, marker='o', markersize=1)
#             lines.append((label, line))
    
#     # Sort lines and create the legend
#     sorted_lines = sort_legend_entries(lines)
#     plt.legend([line for _, line in sorted_lines], [label for label, _ in sorted_lines], title='Feature Size', title_fontsize='small', fontsize='x-small', loc='lower right')

#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     save_path = os.path.join(directory, 'validation_accuracy_vs_epochs.png')  # Adjust the file name if needed
#     plt.savefig(save_path)

##################################################################################
# this works well but the graphs are still unreadable 

# import matplotlib.pyplot as plt
# import numpy as np
# import json
# import os
# from itertools import cycle



# def load_data(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data['val_acc']

def get_feature_size_from_filename(filename):
    parts = filename.split('student_size_')
    if len(parts) > 1:
        feature_size = parts[1].split('_')[0]
        return feature_size
    elif 'umodifiedVanilla' in filename:
        return 'vanilla'
    return 'Unknown'

def get_temperature_from_filename(filename):
    parts = filename.split('_T_')
    if len(parts) > 1:
        temp_part = parts[1].split('_')[0]
        return temp_part
    return None

# colors = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
# temperature_to_color = {}


# def plot_val_accuracies(directory, title, xlabel, ylabel):
#     plt.figure(figsize=(60, 42))  # Adjusted for better visibility
#     lines = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.json'):
#             file_path = os.path.join(directory, filename)
#             val_acc = load_data(file_path)
#             feature_size = get_feature_size_from_filename(filename)
#             temperature = get_temperature_from_filename(filename)
#             epochs = np.arange(len(val_acc))
#             if feature_size == 'vanilla':
#                 color = 'red'
#             else:
#                 if temperature not in temperature_to_color:
#                     temperature_to_color[temperature] = next(colors)
#                 color = temperature_to_color[temperature]
#             line, = plt.plot(epochs, val_acc, linestyle='--', linewidth=2, color=color, marker='o', markersize=2)
#             if feature_size != 'vanilla':
#                 plt.annotate(f'T={temperature}', xy=(epochs[-1], val_acc[-1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9)
#             lines.append((f'{feature_size} (T={temperature})', line))
    
#     sorted_lines = sort_legend_entries(lines)
#     plt.legend([line for _, line in sorted_lines], [label for label, _ in sorted_lines], title='Feature Size', title_fontsize='small', fontsize='x-small', loc='lower right')

#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     save_path = os.path.join(directory, 'validation_accuracy_vs_epochs_temperature.png')
#     plt.savefig(save_path)

##################################################################################


import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict
from itertools import cycle

def load_all_data(directory):
    data_by_temperature = defaultdict(lambda: defaultdict(list))
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            feature_size = get_feature_size_from_filename(filename)
            temperature = get_temperature_from_filename(filename)
            val_acc = data['val_acc']
            data_by_temperature[temperature][feature_size] = val_acc
    return data_by_temperature

# def get_feature_size_from_filename(filename):
#     # Your implementation to extract feature size or 'vanilla'
#     return 'feature_size'

# def get_temperature_from_filename(filename):
#     # Your implementation to extract temperature
#     return 'temperature'

def sort_legend_entries(entries):
    vanilla_entry = [entry for entry in entries if 'Vanilla' in entry[0]]
    numeric_entries = [entry for entry in entries if 'Vanilla' not in entry[0]]
    sorted_numeric_entries = sorted(numeric_entries, key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True)
    sorted_entries = vanilla_entry + sorted_numeric_entries
    return sorted_entries

def plot_data_by_temperature(data_by_temperature, directory):
    feature_sizes = set(fs for temps in data_by_temperature.values() for fs in temps)
    colors = cycle(plt.cm.tab10(np.linspace(0, 1, len(feature_sizes)+1)))  # +1 for vanilla
    feature_size_to_color = {'vanilla': 'red'}
    feature_size_to_color.update({fs: next(colors) for fs in feature_sizes if fs != 'vanilla'})

    num_temperatures = len(data_by_temperature)
    fig, axs = plt.subplots(num_temperatures, figsize=(10, 7 * num_temperatures), sharex=True, sharey=True)
    if num_temperatures == 1:  # If there's only one temperature, axs is not a list
        axs = [axs]
    
    for ax, (temperature, feature_sizes_data) in zip(axs, data_by_temperature.items()):
        lines = []
        for feature_size, val_acc in feature_sizes_data.items():
            epochs = np.arange(len(val_acc))
            label = 'Vanilla' if feature_size == 'vanilla' else f'Size {feature_size}'
            color = feature_size_to_color[feature_size]
            line, = ax.plot(epochs, val_acc, label=label, linestyle='--', linewidth=2, color=color, marker='o', markersize=2)
            lines.append((label, line))

        sorted_lines = sort_legend_entries([(lbl, ln) for lbl, ln in lines])
        ax.legend([line for _, line in sorted_lines], [label for label, _ in sorted_lines], title='Feature Size', title_fontsize='small', fontsize='x-small', loc='upper right')
        ax.set_title(f'Temperature {temperature}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Validation Accuracy')
        ax.set_ylim(0, 100)

        ax.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(directory, 'validation_accuracy_by_feature_size_across_temperatures.png')
    plt.savefig(save_path)
    # plt.show()  # Optionally display the plot







# Usage example:
directory = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_classification'
data_by_temperature = load_all_data(directory)

plot_data_by_temperature(data_by_temperature, directory)
