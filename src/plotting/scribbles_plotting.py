import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_metrics_from_files(directory):
    file_name_pattern = re.compile(r'temperature_(.*?)_retrieval_results_size_(\d+)_topK_(\d+)_KD\.txt')
    metrics_pattern = re.compile(r'Mean Average Precision full:\n(.*?)\n.*?' +
                                 r'Mean Average Precision at top 1000 \(mAP@top_1000\):\n(.*?)\n.*?' +
                                 r'Recall:\nR@1: (.*?)\nR@5: (.*?)\nR@10: (.*?)\nR@20: (.*?)\n.*?' +
                                 r'Precision:\nP@1: (.*?)\nP@5: (.*?)\nP@10: (.*?)\nP@20: (.*?)\n', re.DOTALL)
    
    results = []
    
    for file_name in os.listdir(directory):
        match = file_name_pattern.search(file_name)
        if match:
            temperature, feature_size, top_k = match.groups()
            with open(os.path.join(directory, file_name), 'r') as file:
                content = file.read()
                metrics_match = metrics_pattern.search(content)
                if metrics_match:
                    mAP_full, mAP_top_1000, R1, R5, R10, R20, P1, P5, P10, P20 = metrics_match.groups()
                    results.append({
                        'temperature': float(temperature),
                        'feature_size': int(feature_size),
                        'mAP_full': float(mAP_full)*100,
                        'mAP_top_1000': float(mAP_top_1000)*100,
                        'R@1': float(R1)*100,
                        'R@5': float(R5)*100,
                        'R@10': float(R10)*100,
                        'R@20': float(R20)*100,
                        'P@1': float(P1)*100,
                        'P@5': float(P5)*100,
                        'P@10': float(P10)*100,
                        'P@20': float(P20)*100
                    })
    return pd.DataFrame(results)



def plot_metrics(df, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    feature_sizes = sorted(df['feature_size'].unique())
    temperatures = sorted(df['temperature'].unique())
    metrics = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for temperature in temperatures:
            subset = df[df['temperature'] == temperature]
            if not subset.empty:
                # Plot each temperature as a separate line
                plt.plot(subset['feature_size'], subset[metric], marker='o', label=f'Temp {temperature}')
        
        plt.title(f'{metric} vs Feature Size for Different Temperatures')
        plt.xlabel('Feature Size')
        plt.ylabel(metric)
        plt.ylim(0,100)
        plt.xticks(feature_sizes)  # Ensure x-axis ticks match feature sizes
        plt.legend(title='Temperature')
        plt.grid(True)
        
        file_name = f"{metric}_vs_Feature_Size.png"
        file_path = os.path.join(save_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def plot_metrics_log_scale(df, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    feature_sizes = sorted(df['feature_size'].unique())
    temperatures = sorted(df['temperature'].unique())
    metrics = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for temperature in temperatures:
            subset = df[df['temperature'] == temperature]
            if not subset.empty:
                # Plot each temperature as a separate line
                plt.plot(subset['feature_size'], subset[metric], marker='o', label=f'Temp {temperature}')
        
        plt.title(f'{metric} vs Feature Size for Different Temperatures (Log Scale)')
        plt.xlabel('Feature Size')
        plt.ylabel(metric)
        plt.ylim(0,100)
        plt.xscale('log')  # Set the x-axis to a log scale
        plt.xticks(feature_sizes, labels=feature_sizes)  # Set x-axis ticks to feature sizes
        plt.legend(title='Temperature')
        plt.grid(True, which="both", ls="--")  # Improve grid for log scale
        
        file_name = f"{metric}_vs_Feature_Size_log_scale.png"
        file_path = os.path.join(save_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def plot_metrics_sorted(df, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    temperatures = sorted(df['temperature'].unique())
    metrics = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for temperature in temperatures:
            subset = df[(df['temperature'] == temperature)].sort_values(by='feature_size')
            if not subset.empty:
                plt.plot(subset['feature_size'], subset[metric], marker='o', label=f'Temp {temperature}')
        
        plt.title(f'{metric} vs Feature Size for Different Temperatures')
        plt.xlabel('Feature Size')
        plt.ylabel(metric)
        plt.xscale('linear')  # Use 'linear' for normal plots and 'log' for log-scaled plots
        plt.legend(title='Temperature')
        plt.grid(True)
        
        file_name = f"{metric}_vs_Feature_Size_sorted.png"
        file_path = os.path.join(save_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def plot_metrics_log_scale_sorted(df, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    temperatures = sorted(df['temperature'].unique())
    metrics = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for temperature in temperatures:
            subset = df[(df['temperature'] == temperature)].sort_values(by='feature_size')
            if not subset.empty:
                plt.plot(subset['feature_size'], subset[metric], marker='o', label=f'Temp {temperature}')
        
        plt.title(f'{metric} vs Feature Size for Different Temperatures (Log Scale)')
        plt.xlabel('Feature Size')
        plt.ylabel(metric)
        plt.xscale('log')  # Set the x-axis to a log scale for this version
        plt.legend(title='Temperature')
        plt.grid(True, which="both", ls="--")
        
        file_name = f"{metric}_vs_Feature_Size_log_scale_sorted.png"
        file_path = os.path.join(save_directory, file_name)
        plt.savefig(file_path)
        plt.close()

def plot_metrics_adjusted(df, save_directory, legend_font_size='medium'):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    feature_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # Defined feature sizes
    temperatures = sorted(df['temperature'].unique())
    metrics = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for temperature in temperatures:
            subset = df[(df['temperature'] == temperature) & df['feature_size'].isin(feature_sizes)].sort_values(by='feature_size')
            if not subset.empty:
                plt.plot(subset['feature_size'], subset[metric], marker='o', label=f'Temp {temperature}')
        
        plt.title(f'{metric} vs Feature Size for Different Temperatures')
        plt.xlabel('Feature Size')
        plt.ylabel(metric)
        plt.ylim(0, 100)  # Set y-axis from 0 to 100
        # plt.xticks(feature_sizes, feature_sizes)  # Specify x-axis ticks to match feature sizes
        plt.xticks(df['feature_size'].unique(), df['feature_size'].unique())

        plt.xscale('log')  # Adjust according to your preference ('linear' or 'log')
        plt.legend(title='Temperature', fontsize=legend_font_size)
        plt.grid(True, which="both", ls="--")
        
        file_name = f"{metric}_vs_Feature_Size_adjusted.png"
        file_path = os.path.join(save_directory, file_name)
        plt.savefig(file_path)
        plt.close()

# Example usage
directory = '/home/alabutaleb/Desktop/confirmation/Retrieval_eval_kd_evaluation'
save_directory = '/home/alabutaleb/Desktop/confirmation/Retrieval_eval_kd_evaluation'
df = extract_metrics_from_files(directory)
plot_metrics_sorted(df, save_directory)
plot_metrics_adjusted(df, save_directory)



