import matplotlib.pyplot as plt
import numpy as np
import os
import re
from itertools import cycle
from scipy.optimize import curve_fit
import itertools


def linear_fit(x, a, b):
    return a * x + b

def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d



def extract_metrics_from_file(filepath):
    """Extract metrics from a given file, ensuring robustness in value extraction."""
    metrics = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if 'Mean Average Precision full:' in line:
                # Attempt to extract the numeric value following the colon
                try:
                    value = float(lines[i].split(':')[-1].strip())
                    metrics['mAP_full'] = value
                except ValueError:
                    # Handle potential multi-line value or formatting inconsistency
                    value = float(lines[i+1].strip())  # Assuming the value is on the next line
                    metrics['mAP_full'] = value

            elif 'Mean Average Precision at top 1000' in line:
                try:
                    value = float(lines[i].split(':')[-1].strip())
                    metrics['mAP_top_1000'] = value
                except ValueError:
                    value = float(lines[i+1].strip())
                    metrics['mAP_top_1000'] = value

            else:
                # For other metrics, use regex to ensure we only capture lines with numerical values
                match = re.match(r'(R@1|R@5|R@10|R@20|P@1|P@5|P@10|P@20):\s*([0-9.]+)$', line)
                if match:
                    metric, value = match.groups()
                    metrics[metric] = float(value)
    return metrics


def parse_filename(filename):
    """Parse the filename to extract temperature and feature size."""
    temp_match = re.search(r'temperature_([0-9.]+)', filename)
    size_match = re.search(r'size_([0-9]+|vanilla)', filename)
    temperature = temp_match.group(1) if temp_match else 'vanilla'
    feature_size = size_match.group(1) if size_match else 'unknown'
    return temperature, feature_size



def plot_with_fits(directory, results, color_map):
    """Plot metrics with separate figures for linear and polynomial fits."""
    for feature_size, metrics in results.items():
        for metric, values in metrics.items():
            sorted_values = sorted(values, key=lambda x: x[0])
            temperatures, metric_values = zip(*sorted_values)
            temperatures = np.array(temperatures)
            metric_values = np.array(metric_values)

            # Skip vanilla as it does not fit on a numerical temperature scale
            if feature_size == 'vanilla' or not temperatures.size:
                continue

            # Prepare data for fitting
            temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)

            # Linear fit
            linear_params, _ = curve_fit(linear_fit, temperatures, metric_values)
            plt.figure(figsize=(8, 5))
            plt.scatter(temperatures, metric_values, label=f'Feature size: {feature_size}', color=color_map[feature_size])
            plt.plot(temp_range, linear_fit(temp_range, *linear_params), label='Linear Fit', linestyle='--', color='red')


            plt.xlabel('Temperature', fontsize=13)  # Increase font size for x-axis label to 14 points
            # print(f"metric: {metric}")
            if metric == "mAP_top_1000":
                # exit(f"THIS IS NEVER GETTING TRIGGERED!")
                plt.ylabel("mAP", fontsize=13)  # Increase font size for y-axis label to 14 points        
            else:
                plt.ylabel(metric, fontsize=13)


            plt.legend()
            plt.grid(True)
            save_path_linear = os.path.join(directory, f'{metric}_vs_temperature_linear_fit_{feature_size}.png')
            plt.savefig(save_path_linear)
            plt.close()

            # Polynomial fit
            poly_params, _ = curve_fit(polynomial_fit, temperatures, metric_values, maxfev=10000)
            plt.figure(figsize=(8, 5))
            plt.scatter(temperatures, metric_values, label=f'Feature size: {feature_size}', color=color_map[feature_size])
            plt.plot(temp_range, polynomial_fit(temp_range, *poly_params), label='Polynomial Fit', linestyle='-.', color='green')
            # plt.title(f'{metric} across temperatures (Polynomial Fit, Feature size: {feature_size})')
            # plt.xlabel('Temperature')
            # plt.ylabel(metric)
            plt.xlabel('Temperature', fontsize=13)  # Increase font size for x-axis label to 14 points
            if metric == "mAP_top_1000":
                plt.ylabel("mAP", fontsize=13)  # Increase font size for y-axis label to 14 points        
            else:
                plt.ylabel(metric, fontsize=13)

            plt.legend()
            plt.grid(True)
            save_path_poly = os.path.join(directory, f'{metric}_vs_temperature_polynomial_fit_{feature_size}.png')
            plt.savefig(save_path_poly)
            plt.close()


def plot_aggregated_metrics_with_polynomial_fit(directory, results, color_map, metrics_to_plot):
    """Plot selected metrics with polynomial fits for all feature sizes on a single figure."""
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        for feature_size, metrics in results.items():
            if metric in metrics:
                sorted_values = sorted(metrics[metric], key=lambda x: x[0])
                temperatures, metric_values = zip(*sorted_values)
                temperatures = np.array(temperatures)
                metric_values = np.array(metric_values)

                # Skip if no temperatures or if feature size is 'vanilla'
                if not temperatures.size or feature_size == 'vanilla':
                    continue

                # Polynomial fit
                poly_params, _ = curve_fit(polynomial_fit, temperatures, metric_values, maxfev=20000)

                # Prepare data for plotting the fit
                temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
                plt.scatter(temperatures, metric_values, label=f'Feature size: {feature_size}', color=color_map[feature_size])
                plt.plot(temp_range, polynomial_fit(temp_range, *poly_params), linestyle='-.', color=color_map[feature_size])
                

        plt.xlabel('Temperature', fontsize=13)  # Increase font size for x-axis label to 14 points
        

        if metric == "mAP_top_1000":
            print(f"metric: {metric}")
            plt.ylabel("mAP", fontsize=13)  # Increase font size for y-axis label to 14 points        
        else:
            plt.ylabel(metric, fontsize=13)
            
        plt.legend(title="Feature sizes", loc = 'center right')
        # plt.ylim(0, 100)
        plt.grid(True)
        save_path = os.path.join(directory, f'aggregated_{metric}_vs_temperature_polynomial_fit.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Plot saved: {save_path}')



def plot_metrics(directory):

    results = {}
    metric_names = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']
    for filename in os.listdir(directory):
        if filename.startswith('temperature') or filename.startswith('retrieval'):
            filepath = os.path.join(directory, filename)
            temperature, feature_size = parse_filename(filename)
            metrics = extract_metrics_from_file(filepath)
            if feature_size not in results:
                results[feature_size] = {metric: [] for metric in metric_names}
            for metric in metric_names:
                if metric in metrics:
                    results[feature_size][metric].append((float(temperature) if temperature != 'vanilla' else 0, metrics[metric]))

    feature_sizes = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_sizes)))
    color_map = {size: color for size, color in zip(feature_sizes, colors)}


    # Plotting
    for metric in metric_names:
        has_data = any(results[fs][metric] for fs in feature_sizes if metric in results[fs])
        if not has_data:
            print(f"No data found for metric: {metric}")
            continue  # Skip plotting for this metric if no data is present

        plt.figure(figsize=(8, 5))
        for feature_size, metrics in results.items():
            if metric in metrics and metrics[metric]:  # Check if there are data points for this metric
                temperatures, metric_values = zip(*sorted(metrics[metric], key=lambda x: x[0]))
                plt.scatter(temperatures, metric_values, label=f'Feature size: {feature_size}', color=color_map[feature_size])
        # plt.title(f'{metric} across temperatures')
        # plt.xlabel('Temperature')
        # plt.ylabel(metric)
        plt.xlabel('Temperature', fontsize=13)  # Increase font size for x-axis label to 14 points

        if metric == "mAP_top_1000":
            plt.ylabel("mAP", fontsize=13)  # Increase font size for y-axis label to 14 points        
        else:
            plt.ylabel(metric, fontsize=13)

        plt.legend()
        plt.grid(True)
        save_path = os.path.join(directory, f'{metric}_vs_temperature.png')
        plt.savefig(save_path)
        plt.close()
    print("Plotting complete. Figures saved.")    


     # After plotting individual metrics, plot aggregated metrics with polynomial fits.
    metrics_to_plot = ['mAP_top_1000', 'R@1', 'R@5']
    plot_aggregated_metrics_with_polynomial_fit(directory, results, color_map, metrics_to_plot)
    print("Plotting complete. Original, fitted, and aggregated figures saved.")




# Example usage
directory_summary = "/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_retrieval"
plot_metrics(directory_summary)

