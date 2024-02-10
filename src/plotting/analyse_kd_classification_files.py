


import json
import os

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


def record_max_val_accuracy(directory):
    results = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
            max_val_acc = max(data['val_acc'])
            epoch = data['val_acc'].index(max_val_acc) + 1  # Adding 1 as epochs are usually counted from 1
            temperature = get_temperature_from_filename(filename)
            feature_size = get_feature_size_from_filename(filename)
            results.append({
                "temperature": temperature if temperature is not None else "N/A",
                "feature_size": feature_size,
                "epoch": epoch,
                "max_val_acc": max_val_acc,
                "record": f"Temperature = {temperature if temperature else 'N/A'}, Feature size = {feature_size}, Epoch = {epoch}, max validation accuracy = {max_val_acc:.4f}"
            })

    results_by_temp = sorted([r for r in results if r["temperature"] != "N/A"], key=lambda x: float(x["temperature"]))
    results_by_temp += [r for r in results if r["temperature"] == "N/A"]
    results_by_feature_size = sorted(results, key=lambda x: int(x["feature_size"]) if x["feature_size"].isdigit() else 0)
    results_by_val_acc = sorted(results, key=lambda x: x["max_val_acc"], reverse=True)

    # Write to the main summary file
    main_output_file = os.path.join(directory, "max_val_accuracy_summary_ordered.txt")
    with open(main_output_file, 'w') as file:
        write_summary(file, results_by_temp, results_by_feature_size, results_by_val_acc)
    
    # Write to a separate file for feature size = 8
    feature_size_8_file = os.path.join(directory, "feature_size_8_summary.txt")
    results_feature_size_8 = [r for r in results if r["feature_size"] == "8"]
    with open(feature_size_8_file, 'w') as file:
        write_summary(file, results_feature_size_8, results_feature_size_8, results_feature_size_8)

    print(f"Main summary written to {main_output_file}")
    print(f"Feature size 8 summary written to {feature_size_8_file}")

def write_summary(file, results_by_temp, results_by_feature_size, results_by_val_acc):
    file.write("Ordered by Temperature:\n")
    for result in results_by_temp:
        file.write(result["record"] + "\n")
    file.write("\nOrdered by Feature Size:\n")
    for result in results_by_feature_size:
        file.write(result["record"] + "\n")
    file.write("\nOrdered by Max Validation Accuracy:\n")
    for result in results_by_val_acc:
        file.write(result["record"] + "\n")
# Example usage:
directory = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_classification'
record_max_val_accuracy(directory)
