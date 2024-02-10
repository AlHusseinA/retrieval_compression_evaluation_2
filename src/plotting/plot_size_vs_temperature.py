import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def plot_max_accuracy_vs_feature_size(summary_file):
    feature_size_to_color = {}
    feature_size_to_label = {}
    data = []

    colors = plt.cm.get_cmap('tab10', 10)
    color_index = 0

    with open(summary_file, 'r') as file:
        for line in file:
            if line.startswith("Temperature"):
                parts = line.split(',')
                temperature = parts[0].split('=')[1].strip()
                feature_size = parts[1].split('=')[1].strip()
                max_accuracy = float(parts[3].split('=')[1].strip())

                if feature_size not in feature_size_to_color:
                    feature_size_to_color[feature_size] = colors(color_index)
                    feature_size_to_label[feature_size] = True
                    color_index = (color_index + 1) % 10
                
                if temperature != 'N/A':
                    data.append((float(temperature), max_accuracy, feature_size))

    plt.figure(figsize=(10, 6))
    for temp, acc, size in data:
        plt.scatter(temp, acc, color=feature_size_to_color[size])

    custom_legend = [plt.Line2D([0], [0], color=feature_size_to_color[fs], lw=4, label=fs) for fs in feature_size_to_color]
    plt.legend(handles=custom_legend, title='Feature Size')
    # Adjusted legend and plot settings...
    plt.xscale('log')
    plt.xticks([0.1, 1, 2, 3], ['0.1', '1', '2', '3'])
    plt.title('Max Validation Accuracy vs Temperature for Different Feature Sizes')
    plt.xlabel('Temperature')
    plt.ylabel('Max Validation Accuracy (%)')
    plt.ylim(0, 100) 
    plt.grid(True)
    # Save the first plot
    output_image_path = os.path.join(os.path.dirname(summary_file), 'max_accuracy_vs_temperature_scatter_no_labels.png')
    plt.savefig(output_image_path)
    plt.close()  # Close the plot to free up memory
 
    # plt.figure(figsize=(10, 6))
    # for temp, acc, size in data:
    #     plt.scatter(temp, acc, color=feature_size_to_color[size])
    #     plt.text(temp, acc, size, fontsize=9)

    # custom_legend = [plt.Line2D([0], [0], color=feature_size_to_color[fs], lw=4, label=fs) for fs in feature_size_to_color]
    # plt.legend(handles=custom_legend, title='Feature Size')

    # plt.title('Max Validation Accuracy vs Temperature for Different Feature Sizes')
    # plt.xlabel('Temperature')
    # plt.xscale('log')  # Set the x-axis to a log scale
    # # Explicitly set the ticks and labels on the x-axis
    # # Assuming you want to emphasize these specific points on a log scale
    # plt.xticks([0.1, 1, 2, 3], ['0.1', '1', '2', '3'])
    # plt.ylabel('Max Validation Accuracy (%)')
    # plt.ylim(0, 100)

    # plt.grid(True)

    # # Saving the plot in the same directory as the summary file
    # directory = os.path.dirname(summary_file)
    # output_image_path = os.path.join(directory, 'max_accuracy_vs_temperature.png')
    # plt.savefig(output_image_path)
    # print(f"Plot saved to {output_image_path}")
    # plt.close()  # Close the plot window to free resources



# Usage example:
summary_file = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_classification/summary_by_size.txt'
directory = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_classification'
plot_max_accuracy_vs_feature_size(summary_file)
