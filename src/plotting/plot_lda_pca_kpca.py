import matplotlib.pyplot as plt
import os
import re
import glob
import pandas as pd


def parse_results_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    feature_size = re.search(r'feature size (\w+)', content).group(1)
    if feature_size == 'vanilla':
        feature_size = 2048
    else:
        feature_size = int(feature_size)
    
    metrics = {
        'Mean Average Precision full': float(re.search(r'Mean Average Precision full:\n(\d+\.\d+)', content).group(1)),
        'mAP@top_1000': float(re.search(r'Mean Average Precision at top 1000 \(mAP@top_1000\):\n(\d+\.\d+)', content).group(1)),
        'R@1': float(re.search(r'R@1: (\d+\.\d+)', content).group(1)),
        'R@5': float(re.search(r'R@5: (\d+\.\d+)', content).group(1)),
        'R@10': float(re.search(r'R@10: (\d+\.\d+)', content).group(1))
    }
    
    method = 'Vanilla' if 'vanilla' in file_path else re.search(r'(LDA|PCA|KPCA)', file_path).group(1)
    
    return feature_size, method, metrics

def plot_metric(data, metric, save_dir):
    plt.figure(figsize=(10, 6))
    for method in data['Method'].unique():
        subset = data[data['Method'] == method]
        plt.plot(subset['Feature Size'], subset[metric], marker='o', label=method)
        
    plt.title(f'{metric}')
    plt.xlabel('Feature Size')
    plt.ylabel(metric)
    plt.ylim(0, 100)  # Commented out for debugging

    plt.xscale('log', base=2)
    plt.xticks(data['Feature Size'].unique(), data['Feature Size'].unique())
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{metric.replace("@", "at")}.png'))
    plt.close()

def generate_plots(directory, save_dir):
    data = []
    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        feature_size, method, metrics = parse_results_file(file_path)
        for metric, value in metrics.items():
            data.append({'Feature Size': feature_size, 'Method': method, metric: value*100})
    
    df = pd.DataFrame(data)
    df = df.pivot_table(index=['Feature Size', 'Method']).reset_index()

    metrics = ['Mean Average Precision full', 'mAP@top_1000', 'R@1', 'R@5', 'R@10']
    for metric in metrics:
        plot_metric(df, metric, save_dir)


# Example usage
dir = "/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_pca_lda_kpca"
generate_plots(dir, dir)
