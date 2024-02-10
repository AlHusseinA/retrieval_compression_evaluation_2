import torch
import matplotlib.pyplot as plt

def display_results_neatly(map_score, r1, r5, r10):
    """
    Displays the retrieval performance metrics neatly on the terminal.

    Args:
    map_score (torch.Tensor): Tensor containing the mean average precision score.
    r1 (torch.Tensor): Tensor containing the recall at 1 score.
    r5 (torch.Tensor): Tensor containing the recall at 5 score.
    r10 (torch.Tensor): Tensor containing the recall at 10 score.
    """

    # Ensure the inputs are tensors and can be converted to items
    assert all(isinstance(metric, torch.Tensor) for metric in [map_score, r1, r5, r10]), \
        "All inputs must be torch.Tensor type"

    # Create a dictionary to hold the results
    results = {
        "Mean Average Precision (mAP)": map_score.item(),
        "Recall at 1": r1.item(),
        "Recall at 5": r5.item(),
        "Recall at 10": r10.item()
    }

    # Display the results
    print("Retrieval Performance Metrics:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")  # Formatting the score to 4 decimal places for readability


def find_relevant_lists(similarity_matrix, gallery_labels, query_labels):

    num_queries = similarity_matrix.shape[0]
    num_gallery_items = similarity_matrix.shape[1]

    # Create binary ground truth tensor for each query-gallery pair
    expanded_query_labels = query_labels.unsqueeze(1).expand(-1, num_gallery_items)  # Reshape for broadcasting
    expanded_gallery_labels = gallery_labels.unsqueeze(0).expand(num_queries, -1)  # Reshape for broadcasting
    # ground_truths = (query_labels == gallery_labels).int()
    ground_truths = (expanded_query_labels == expanded_gallery_labels).int()

    # Convert the ground_truths tensor to a list of lists
    relevant_lists = []
    for row in ground_truths:
        relevant_indices = torch.nonzero(row, as_tuple=False).squeeze(1).tolist()  # Find indices of relevant items
        relevant_lists.append(relevant_indices)

    return relevant_lists


def format_retrieved_indices(sorted_indices):
    """
    Convert the sorted indices tensor from the retrieve function to a list of lists
    suitable for the mAP and AP calculation functions.
    
    :param sorted_indices: Torch tensor of sorted indices from retrieve function.
    :return: List of lists, each containing retrieved item indices for a query.
    """
    # Convert the tensor to a list of numpy arrays (each array for one query)
    retrieved_lists = [indices.cpu().numpy() for indices in sorted_indices]

    return retrieved_lists



def parse_vanilla_results(file_path):
    vanilla_results = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if "Mean Average Precision full:" in line:
            # Extract the next line for the value if it's not on the same line
            vanilla_results['mAP'] = float(lines[i + 1].strip()) if line.strip().endswith(':') else float(line.split(':')[1].strip())
        elif "Mean Average Precision at top 1000 (mAP@top_1000):" in line:
            vanilla_results['mAP@top_1000'] = float(lines[i + 1].strip()) if line.strip().endswith(':') else float(line.split(':')[1].strip())
        elif "R@1:" in line:
            vanilla_results['R@1'] = float(line.split(':')[1].strip())
        elif "R@5:" in line:
            vanilla_results['R@5'] = float(line.split(':')[1].strip())
        elif "R@10:" in line:
            vanilla_results['R@10'] = float(line.split(':')[1].strip())
        elif "R@20:" in line:
            vanilla_results['R@20'] = float(line.split(':')[1].strip())

    return vanilla_results



def multiply_results_by_100(results, top_k):
    for feature_size in results:
        for key, value in results[feature_size].items():
            if key in ['mAP', f'mAP at top k {top_k}']:
                # Directly multiply if the value is a Tensor
                results[feature_size][key] = value * 100
            else:
                # Check if the value is a dictionary
                if isinstance(value, dict):
                    results[feature_size][key] = {k: v * 100 for k, v in value.items()}
                else:
                    # Handle non-dictionary, presumably Tensor, values
                    results[feature_size][key] = value * 100
    return results


def plot_metrics(results, top_k, feature_sizes, save_dir):

    # feature_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    feature_sizes.reverse()  
    results = multiply_results_by_100(results, top_k)

    if f'mAP at top k {top_k}' in results:
        results["mAP at top k {top_k}"] = results[f'mAP@_top_{top_k}']
        del results[f"mAP at top k {top_k}"]



    # Plot for mean_ap and mean_ap_top_k
    for metric in ['mAP', f'mAP at top k {top_k}']:
        display_metric = metric.replace('mAP at top k', f'mAP@_top_{top_k}') if 'top k' in metric else metric
        values = [results[size][metric] for size in feature_sizes]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(feature_sizes, values, marker='o', linestyle='-')
        ax.set_xlabel("Feature Size")
        ax.set_xticks(feature_sizes)
        ax.set_ylabel(metric.title())
        ax.autoscale(enable=False, axis='y', tight=None)
        ax.set_ylim([0, 100])
        ax.grid(True)
        fig.savefig(f"{save_dir}/{metric}.png")
        plt.close(fig)

    # Plot for mean_recalls at different ranks
    for rank in [1, 5, 10, 20]:
        values = [results[size]['mean_recalls'][rank] for size in feature_sizes]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(feature_sizes, values, marker='o', linestyle='-')
        ax.set_title(f"Recalls@{rank}")
        ax.set_xlabel("Feature Size")
        ax.set_xticks(feature_sizes)
        ax.set_ylabel(f"Recalls@{rank}")
        ax.autoscale(enable=False, axis='y', tight=None)
        ax.set_ylim([0, 100])
        ax.grid(True)
        fig.savefig(f"{save_dir}/mean_recalls_{rank}.png")
        plt.close(fig)

    # Additional plotting for R@1, R@5, R@10 in one figure
    fig, ax = plt.subplots(figsize=(10, 6))
    for rank, color in zip([1, 5, 10], ['r', 'g', 'b']):
        values = [results[size]['mean_recalls'][rank] for size in feature_sizes]
        ax.plot(feature_sizes, values, marker='o', linestyle='-', color=color, label=f'R@{rank}')
   

    
    
    ax.set_title("Recalls@1, 5, 10 ")
    ax.set_xlabel("Feature Size")

    all_feature_sizes = feature_sizes 
    ax.set_xticks(all_feature_sizes)  
    ax.set_ylabel("Recall")
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.legend()

    fig.savefig(f"{save_dir}/recalls_at_1_5_10.png")
    plt.close(fig)
         


def plot_metrics_logarithmic(results, top_k, feature_sizes, save_dir):
    feature_sizes = sorted(feature_sizes)

    def plot_single_metric(original_metric, values, feature_sizes, rank=None):
        # Handling special cases for display_metric based on whether it's a recall metric
        if rank is not None:
            # Directly use "Recalls@{rank}" for recall metrics
            display_metric = f"Recalls@{rank}"
        elif 'top k' in original_metric:
            # Format display_metric for mAP@top_k case
            display_metric = f'mAP@top_{top_k}'
        else:
            # Use the original_metric as is for other cases
            display_metric = original_metric

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(feature_sizes, values, marker='o', linestyle='-')

        ax.set_title(display_metric)
        ax.set_xlabel("Feature Size (log scale)")
        ax.set_xscale('log')
        ax.set_xticks(feature_sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylabel(display_metric)  # This now correctly sets ylabel based on metric type
        ax.autoscale(enable=False, axis='y', tight=None)
        ax.set_ylim(bottom=0, top=100)

        ax.grid(True)

        # Construct file_name, simplifying the logic
        # Use a simplified, lowercase, underscore-free version of display_metric for file names
        file_name = display_metric.replace('@', 'at').replace(' ', '_').lower()

        if rank is not None:
            # Append '_log' for consistency in file naming
            file_name += "_log"
        fig.savefig(f"{save_dir}/{file_name}.png")
        plt.close(fig)       

    # Original metrics plotting remains unchanged
    metrics_to_plot = ['mAP', f'mAP at top k {top_k}']
    for metric in metrics_to_plot:
        values = [results.get(size, {}).get(metric, 0) for size in feature_sizes]
        plot_single_metric(metric, values, feature_sizes)

    # Adjusted recall metrics plotting
    for rank in [1, 5, 10, 20]:
        values = [results.get(size, {}).get('mean_recalls', {}).get(rank, 0) for size in feature_sizes]
        plot_single_metric('mean_recalls', values, feature_sizes, rank=rank)





 # Additional plotting for R@1, R@5, R@10 in one figure, with logarithmic x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Ensure log scale is set for the x-axis
    ax.set_xscale('log')
    # Plot the combined recalls
    for rank, color in zip([1, 5, 10], ['r', 'g', 'b']):
        values = [results.get(size, {}).get('mean_recalls', {}).get(rank, 0) for size in feature_sizes]
        ax.plot(feature_sizes, values, marker='o', linestyle='-', color=color, label=f'Recalls@{rank}')
    

    ax.set_title("Recalls@1, 5, 10 ")
    ax.set_xlabel("Feature Size (log scale)")

    vanilla_feature_size = max(feature_sizes) + 5 

    ax.set_xticks(feature_sizes)  
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylabel("Recall")
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.legend()

    fig.savefig(f"{save_dir}/recalls_at_1_5_10_log.png")
    plt.close(fig)









        

