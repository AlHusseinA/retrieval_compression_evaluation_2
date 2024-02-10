import os
import json


def flatten_if_nested(values):
    """Flatten the list if it is nested."""
    if values and isinstance(values[0], list):
        return [item for sublist in values for item in sublist]
    return values


def analyze_experiments_results(directory):

    # Step 1: List all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            # Step 2: Read JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)
            
            # Initialize the analysis result string
            analysis_result = ""
            
            # Step 3: Analyze the data
            epochs = data["epochs"]
            for metric in ["train_acc", "val_acc", "train_loss", "val_loss"]:
                values = data[metric]
                
                # Ensure values are not nested
                values = flatten_if_nested(values)
                
                if metric in ["train_acc", "val_acc"]:
                    # For accuracy, find the maximum value and the value at the last epoch
                    max_value = max(values)
                    max_epoch = epochs[values.index(max_value)]
                    final_value = values[-1]
                    final_epoch = epochs[-1]
                    analysis_result += f"Max {metric} is {max_value}% at epoch {max_epoch}\n"
                    analysis_result += f"Final {metric} is {final_value}% at epoch {final_epoch}\n"
                else:
                    # For loss, find the minimum value and the value at the last epoch
                    min_value = min(values)
                    min_epoch = epochs[values.index(min_value)]
                    final_value = values[-1]
                    final_epoch = epochs[-1]
                    analysis_result += f"Min {metric} is {min_value} at epoch {min_epoch}\n"
                    analysis_result += f"Final {metric} is {final_value} at epoch {final_epoch}\n"
            
            # Step 4: Write the analysis to a text file
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_filepath = os.path.join(directory, text_filename)
            with open(text_filepath, 'w') as text_file:
                text_file.write(analysis_result)

# Example usage

# Example usage
vanilla_jsons = "/home/alabutaleb/Desktop/confirmation"
# baseline_jsons = "/home/alabutaleb/Desktop/confirmation/baselines_allsizes/logs_new"
# analyze_experiments_results(baseline_jsons)
analyze_experiments_results(vanilla_jsons)

