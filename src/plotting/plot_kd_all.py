
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re


def parse_results_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    if 'vanilla' in file_path:
        feature_size = 2048
        temperature = 'vanilla'  # Use a distinctive identifier for the vanilla model
        
        metrics = {
            'mAP': float(re.search(r'Mean Average Precision full:\n(\d+\.\d+)', content).group(1)),
            # Adjusted regex to match the provided content format
            'mAP@top 1000': float(re.search(r'Mean Average Precision at top 1000.*?:\n(\d+\.\d+)', content).group(1)),
            'R@1': float(re.search(r'R@1: (\d+\.\d+)', content).group(1)),
            'R@5': float(re.search(r'R@5: (\d+\.\d+)', content).group(1)),
            'R@10': float(re.search(r'R@10: (\d+\.\d+)', content).group(1)),
            'R@20': float(re.search(r'R@20: (\d+\.\d+)', content).group(1)),
            'P@1': float(re.search(r'P@1: (\d+\.\d+)', content).group(1)),
            'P@5': float(re.search(r'P@5: (\d+\.\d+)', content).group(1)),
            'P@10': float(re.search(r'P@10: (\d+\.\d+)', content).group(1)),
            'P@20': float(re.search(r'P@20: (\d+\.\d+)', content).group(1)),
        }
    else:
        temperature = re.search(r'temperature_(\d+\.?\d*)_retrieval', file_path).group(1)
        feature_size = re.search(r'retrieval_results_size_(\d+)_', file_path).group(1)
        feature_size = int(feature_size)

        metrics = {
            # Use the corrected 'mAP' key for consistency
            'mAP': float(re.search(r'Mean Average Precision full:\n(\d+\.\d+)', content).group(1)),
            # Use the new key 'mAP@top 1000' directly since it's been standardized
            'mAP@top 1000': float(re.search(r'Mean Average Precision at top 1000.*?:\n(\d+\.\d+)', content).group(1)),
            # Rest of the metrics remain unchanged
            'R@1': float(re.search(r'R@1: (\d+\.\d+)', content).group(1)),
            'R@5': float(re.search(r'R@5: (\d+\.\d+)', content).group(1)),
            'R@10': float(re.search(r'R@10: (\d+\.\d+)', content).group(1)),
            'R@20': float(re.search(r'R@20: (\d+\.\d+)', content).group(1)),
            'P@1': float(re.search(r'P@1: (\d+\.\d+)', content).group(1)),
            'P@5': float(re.search(r'P@5: (\d+\.\d+)', content).group(1)),
            'P@10': float(re.search(r'P@10: (\d+\.\d+)', content).group(1)),
            'P@20': float(re.search(r'P@20: (\d+\.\d+)', content).group(1)),
        }

    return {'Temperature': temperature, 'Feature Size': feature_size, **metrics}




# def plot_metric_with_fig_ax(data, metric, save_dir):
#     # Ensure the save directory exists
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     feature_sizes = sorted(data['Feature Size'].unique(), key=lambda x: str(x))
#     temperatures = sorted(data['Temperature'].unique(), key=lambda x: (str(x).lower(), x))
    
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for temperature in temperatures:
#         subset = data[data['Temperature'] == temperature].sort_values(by='Feature Size')
#         ax.plot(subset['Feature Size'], subset[metric] * 100, marker='o', linestyle='-', label=f'Temp {temperature}' if temperature != 'vanilla' else 'vanilla')
    
#     # ax.set_title(f'{metric} by Feature Size')
#     ax.set_xlabel('Feature Size (log scale)', fontsize=11)
#     ax.set_ylabel(f'{metric} (%)', fontsize=11)
#     ax.set_xscale('log', base=2)
#     ax.set_xticks(feature_sizes)
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
#     # ax.set_ylim(0, 100)

#     handles, labels = ax.get_legend_handles_labels()
#     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: (t[0].lower(), t[0])))
#     ax.legend(handles, labels, loc='lower right')
    
#     ax.grid(True)

#     fig.savefig(os.path.join(save_dir, f'{metric.replace("@", "at")}_log.png'))
#     plt.close(fig)


def generate_plots_with_fig_ax(directory, save_dir):
    data = []
    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        result = parse_results_file(file_path)
        data.append(result)
    
    df = pd.DataFrame(data)

    # Rename columns in the DataFrame to match the new metric names
    df.rename(columns={'Mean Average Precision full': 'mAP', 'mAP@top_1000': 'mAP@top 1000'}, inplace=True)

    metrics = ['mAP', 'mAP@top 1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    for metric in metrics:
        plot_metric_with_fig_ax(df, metric, save_dir)


# def plot_metric_with_fig_ax(data, metric, save_dir):
#     # Ensure the save directory exists
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     feature_sizes = sorted(data['Feature Size'].unique(), key=lambda x: str(x))
#     temperatures = sorted(data['Temperature'].unique(), key=lambda x: (str(x).lower(), x))
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for temperature in temperatures:
#         subset = data[data['Temperature'] == temperature].sort_values(by='Feature Size')
#         ax.plot(subset['Feature Size'], subset[metric] * 100, marker='o', linestyle='-', label=f'Temp {temperature}' if temperature != 'vanilla' else 'vanilla')
    
#     ax.set_xlabel('Feature Size (log scale)', fontsize=11)

#     if metric == 'mAP@top_1000' or metric == 'mAP@top 1000':
#         ax.set_ylabel(f'mAP', fontsize=11)
#     else:

#         ax.set_ylabel(f'{metric} (%)', fontsize=11)

#     ax.set_xscale('log', base=2)
#     ax.set_xticks(feature_sizes)
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

#     handles, labels = ax.get_legend_handles_labels()
#     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: (t[0].lower(), t[0])))
#     # Place legend outside the plot area to the right, and use multiple columns if necessary
#     ax.legend(handles, labels, loc='lower right')#, bbox_to_anchor=(1, 1), fontsize=10, ncol=1)
    
#     ax.grid(True)
#     fig.tight_layout(rect=[0,0,0.75,1])  # Adjust the rect parameter as needed to fit the plot and legend comfortably
#     ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=2, borderpad=0.5, labelspacing=0.5, handletextpad=0.5)

#     fig.savefig(os.path.join(save_dir, f'{metric.replace("@", "at")}_log.png'))
#     plt.close(fig)


def plot_metric_with_fig_ax(data, metric, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_sizes = sorted(data['Feature Size'].unique(), key=lambda x: str(x))
    temperatures = sorted(data['Temperature'].unique(), key=lambda x: (str(x).lower(), x))

    fig, ax = plt.subplots(figsize=(10, 6))
    for temperature in temperatures:
        subset = data[data['Temperature'] == temperature].sort_values(by='Feature Size')
        ax.plot(subset['Feature Size'], subset[metric] * 100, marker='o', linestyle='-', label=f'Temp {temperature}' if temperature != 'vanilla' else 'vanilla')

    ax.set_xlabel('Feature Size (log scale)', fontsize=13)
    # ax.set_ylabel(f'{metric}', fontsize=11 if metric not in ['mAP@top_1000', 'mAP@top 1000'] else 'mAP')

    if metric == 'mAP@top_1000' or metric == 'mAP@top 1000':
        ax.set_ylabel(f'mAP', fontsize=13)
    else:

        ax.set_ylabel(f'{metric}', fontsize=13)

    ax.set_xscale('log', base=2)
    ax.set_xticks(feature_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: (t[0].lower(), t[0])))

    # Adjust the legend to be inside the plot in the lower right corner
    ax.legend(handles, labels, loc='lower right', fontsize=11, ncol=1, borderpad=0.5, labelspacing=0.5, handletextpad=0.5)

    ax.grid(True)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, f'{metric.replace("@", "at")}_log.png'))
    plt.close(fig)






# Example usage
directory = '/home/alabutaleb/Desktop/confirmation/Retrieval_eval_kd_evaluation'
save_dir = '/home/alabutaleb/Desktop/confirmation/Retrieval_eval_kd_evaluation'
generate_plots_with_fig_ax(directory, save_dir)
