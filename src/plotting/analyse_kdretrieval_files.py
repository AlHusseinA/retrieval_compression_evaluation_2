import os
import re
# Define a list of metrics we are interested in
metrics_of_interest = ['mAP_full', 'mAP_top_1000', 'R@1', 'R@5', 'R@10', 'R@20', 'P@1', 'P@5', 'P@10', 'P@20']



def extract_info_from_filename(filename):
    """
    Extract temperature and feature size from the filename.
    For the vanilla case, both temperature and feature size are returned as 'vanilla'.
    """
    if 'vanilla' in filename:
        return 'vanilla', 'vanilla'
    else:
        try:
            # Example filename: temperature_0.06_retrieval_results_size_32_topK_1000_KD.txt
            temp_part = filename.split('temperature_')[1].split('_')[0]
            feature_size_part = filename.split('size_')[1].split('_')[0]
            return temp_part, feature_size_part
        except IndexError:
            print(f"Filename format is incorrect or unexpected: {filename}")
            return None, None


def parse_metrics_from_file(filepath):
    """
    Parse the file to extract metrics using regular expressions for more flexible matching.
    """
    # Define patterns with compiled regular expressions for each metric
    metrics_patterns = {
        'mAP_full': re.compile(r'Mean Average Precision full:\s*(\d+\.\d+)'),
        'mAP_top_1000': re.compile(r'Mean Average Precision at top 1000 \(mAP@top_1000\):\s*(\d+\.\d+)'),
        'R@1': re.compile(r'R@1:\s*(\d+\.\d+)'),
        'R@5': re.compile(r'R@5:\s*(\d+\.\d+)'),
        'R@10': re.compile(r'R@10:\s*(\d+\.\d+)'),
        'R@20': re.compile(r'R@20:\s*(\d+\.\d+)'),
        'P@1': re.compile(r'P@1:\s*(\d+\.\d+)'),
        'P@5': re.compile(r'P@5:\s*(\d+\.\d+)'),
        'P@10': re.compile(r'P@10:\s*(\d+\.\d+)'),
        'P@20': re.compile(r'P@20:\s*(\d+\.\d+)')
    }

    parsed_metrics = {metric: 0 for metric in metrics_patterns.keys()}

    with open(filepath, 'r') as file:
        content = file.read()
        for metric, pattern in metrics_patterns.items():
            match = pattern.search(content)
            if match:
                parsed_metrics[metric] = float(match.group(1))

    return parsed_metrics


def update_top_metrics(top_metrics, metric, new_value, temperature, feature_size, tolerance=0.009):
    """
    Update the list of top metrics with a new value, considering a tolerance for the second highest value.
    """
    current_top = top_metrics[metric]
    if not current_top:
        # If the list is empty, simply add the new value
        top_metrics[metric].append((new_value, temperature, feature_size))
    else:
        # If there's at least one value, check against the highest
        highest_value, highest_temp, highest_size = current_top[0]
        if abs(highest_value - new_value) < tolerance:
            # If the new value is within tolerance of the highest, consider it equal and do not add as second highest
            pass  # You might want to update if the new one is actually higher but within tolerance
        else:
            # If outside tolerance, add as a new entry
            top_metrics[metric].append((new_value, temperature, feature_size))
            # Sort and keep top 2
            top_metrics[metric].sort(key=lambda x: x[0], reverse=True)
            top_metrics[metric] = top_metrics[metric][:2]



def record_max_retrieval_metrics(directory):
    # Initialize dictionaries to hold the top two values for each metric and metrics for size 8 with different temperatures
    top_metrics = {metric: [] for metric in metrics_of_interest}
    size_8_metrics = {metric: [] for metric in metrics_of_interest}
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and 'retrieval_results' in filename:
            temperature, feature_size = extract_info_from_filename(filename)
            metrics = parse_metrics_from_file(os.path.join(directory, filename))
            
            for metric, value in metrics.items():
                update_top_metrics(top_metrics, metric, value, temperature, feature_size)
                if feature_size == '8':
                    size_8_metrics[metric].append((value, temperature))
    
    # Write to the summary file
    summary_file_path = os.path.join(directory, "retrieval_metrics_summary.txt")
    
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write("[Info regarding max metrics]\n")
        for metric in metrics_of_interest:
            if top_metrics[metric]:
                # Write max value info
                max_value_info = top_metrics[metric][0]  # The first tuple is the max value
                summary_file.write(f"max {metric} = {max_value_info[0]} at feature size {max_value_info[2]} and temperature {max_value_info[1]}\n")
        
        summary_file.write("\n[Info regarding second highest metrics]\n")
        for metric in metrics_of_interest:
            if len(top_metrics[metric]) > 1:
                # Write second highest value info if it exists
                second_highest_value_info = top_metrics[metric][1]  # The second tuple is the second highest value
                summary_file.write(f"second highest {metric} = {second_highest_value_info[0]} at feature size {second_highest_value_info[2]} and temperature {second_highest_value_info[1]}\n")
        
        summary_file.write("\n[Info regarding feature size 8 metrics]\n")
        for metric in metrics_of_interest:
            sorted_size_8_metrics = sorted(size_8_metrics[metric], key=lambda x: x[0], reverse=True)
            summary_file.write(f"{metric} (for all temperatures)\n")
            for value, temperature in sorted_size_8_metrics:
                summary_file.write(f"{metric} = {value} at temperature {temperature}\n")

    print(f"Summary written to {summary_file_path}")


#debugging
    


# Example usage:
directory = "/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_retrieval"
record_max_retrieval_metrics(directory)
