import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.optimize import curve_fit

def linear_fit(x, a, b):
    return a * x + b

def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def create_data_by_temperature_from_summary(summary_file):
    data_by_temperature = {}

    with open(summary_file, 'r') as file:
        for line in file:
            if line.startswith("Temperature"):
                parts = line.strip().split(',')
                temperature = parts[0].split('=')[1].strip()
                feature_size = parts[1].split('=')[1].strip()
                max_val_acc = float(parts[3].split('=')[1].strip())

                if temperature == 'N/A':  # Skip entries without a valid temperature
                    continue

                # Initialize nested dictionary if temperature is encountered for the first time
                if temperature not in data_by_temperature:
                    data_by_temperature[temperature] = {}

                # Store max validation accuracy for each feature size under its temperature
                data_by_temperature[temperature][feature_size] = max_val_acc

    return data_by_temperature

def plot_fitted_lines_with_data(data_by_temperature, directory):
    feature_sizes = set(fs for temps in data_by_temperature.values() for fs in temps)
    colors = cycle(plt.cm.tab10(np.linspace(0, 1, len(feature_sizes) + 1)))
    feature_size_to_color = {'vanilla': 'red'}
    feature_size_to_color.update({fs: next(colors) for fs in feature_sizes if fs != 'vanilla'})

    fit_functions = {
        'linear': (linear_fit, ['a', 'b']),
        'polynomial': (polynomial_fit, ['a', 'b', 'c', 'd'])
    }

    for fit_type, (fit_func, params) in fit_functions.items():
        plt.figure(figsize=(8, 5))

        for feature_size in feature_sizes:
            temperatures = []
            max_accuracies = []

            for temperature, sizes_data in data_by_temperature.items():
                if feature_size in sizes_data and temperature != 'N/A':
                    temperatures.append(float(temperature))
                    max_accuracies.append(sizes_data[feature_size])

            if temperatures and max_accuracies:
                temperatures = np.array(temperatures)
                max_accuracies = np.array(max_accuracies)

                # Plot the data points
                plt.scatter(temperatures, max_accuracies, color=feature_size_to_color[feature_size], label=f'Size {feature_size}')

                if len(temperatures) > 1:
                    # Fit and plot the fitted line
                    popt, _ = curve_fit(fit_func, np.log(temperatures), max_accuracies, maxfev=10000)
                    fitted_temps = np.linspace(min(temperatures), max(temperatures), 100)
                    fitted_accs = fit_func(np.log(fitted_temps), *popt) if fit_type in ['linear', 'polynomial'] else fit_func(fitted_temps, *popt)
                    plt.plot(fitted_temps, fitted_accs, color=feature_size_to_color[feature_size], linestyle='--')

        plt.xscale('log')
        plt.xticks([0.1, 1, 2, 3], ['0.1', '1', '2', '3'])
        # plt.xlabel('Temperature')
        # plt.ylabel('Max Validation Accuracy (%)')
        plt.xlabel('Temperature', fontsize=11)  # Increase font size for x-axis label to 14 points
        plt.ylabel('Accuracy (%)', fontsize=11)  # Increase font size for y-axis label to 14 points        
        # plt.ylim(0, 100)
        # plt.title(f'Fitted Max Validation Accuracy vs Temperature ({fit_type} fit)')
        plt.legend(title='Feature Size', loc='best')
        plt.grid(True)

        output_image_path = os.path.join(directory, f'max_accuracy_vs_temperature_{fit_type}_fit_with_data.png')
        plt.savefig(output_image_path)
        plt.close()
        print(f"{fit_type.capitalize()} fit plot with data saved to {output_image_path}")


def plot_fitted_lines(data_by_temperature, directory):
    feature_sizes = set(fs for temps in data_by_temperature.values() for fs in temps)
    colors = cycle(plt.cm.tab10(np.linspace(0, 1, len(feature_sizes) + 1)))
    feature_size_to_color = {'vanilla': 'red'}
    feature_size_to_color.update({fs: next(colors) for fs in feature_sizes if fs != 'vanilla'})

    fit_functions = {
        'linear': (linear_fit, ['a', 'b']),
        # 'exponential': (exponential_fit, ['a', 'b', 'c']),
        'polynomial': (polynomial_fit, ['a', 'b', 'c', 'd'])
    }
    for fit_type, (fit_func, params) in fit_functions.items():
        plt.figure(figsize=(8, 5))

        for feature_size in feature_sizes:
            temperatures = []
            max_accuracies = []

            for temperature, sizes_data in data_by_temperature.items():
                if feature_size in sizes_data and temperature != 'N/A':
                    temperatures.append(float(temperature))
                    max_accuracies.append(sizes_data[feature_size])

            if temperatures and max_accuracies:
                temperatures = np.array(temperatures)
                max_accuracies = np.array(max_accuracies)

                if len(temperatures) > 1:
                    popt, _ = curve_fit(fit_func, np.log(temperatures), max_accuracies, maxfev=10000)
                    fitted_temps = np.linspace(min(temperatures), max(temperatures), 100)
                    if fit_type == 'linear' or fit_type == 'polynomial':
                        fitted_accs = fit_func(np.log(fitted_temps), *popt)
                    plt.plot(fitted_temps, fitted_accs, label=f'Size {feature_size}', color=feature_size_to_color[feature_size])
        
        # Automatic y-axis scaling is ensured here by not setting plt.ylim()
        plt.xscale('log')
        plt.xticks([0.1, 1, 2, 3], ['0.1', '1', '2', '3'])
        plt.xlabel('Temperature', fontsize=11)  # Increase font size for x-axis label to 14 points
        plt.ylabel('Accuracy (%)', fontsize=11)  # Increase font size for y-axis label to 14 points

        # plt.ylabel('Max Validation Accuracy (%)')

        # plt.title(f'Fitted Max Validation Accuracy vs Temperature ({fit_type} fit)')
        plt.legend(title='Feature Size', loc='best')
        plt.grid(True)

        output_image_path = os.path.join(directory, f'max_accuracy_vs_temperature_{fit_type}_fit.png')
        plt.savefig(output_image_path)
        plt.close()
        print(f"{fit_type.capitalize()} fit plot saved to {output_image_path}")

    # for fit_type, (fit_func, params) in fit_functions.items():
        # plt.figure(figsize=(8, 5))

    #     for feature_size in feature_sizes:
    #         temperatures = []
    #         max_accuracies = []

    #         for temperature, sizes_data in data_by_temperature.items():
    #             if feature_size in sizes_data and temperature != 'N/A':
    #                 temperatures.append(float(temperature))
    #                 max_accuracies.append(sizes_data[feature_size])

    #         if temperatures and max_accuracies:
    #             temperatures = np.array(temperatures)
    #             max_accuracies = np.array(max_accuracies)

    #             if len(temperatures) > 1:
    #                 popt, _ = curve_fit(fit_func, np.log(temperatures), max_accuracies, maxfev=10000)
    #                 fitted_temps = np.linspace(min(temperatures), max(temperatures), 100)
    #                 if fit_type == 'linear' or fit_type == 'polynomial':
    #                     fitted_accs = fit_func(np.log(fitted_temps), *popt)
    #                 else:  # Handle exponential fit separately due to its nature
    #                     fitted_accs = fit_func(fitted_temps, *popt)
    #                 plt.plot(fitted_temps, fitted_accs, label=f'Size {feature_size}', color=feature_size_to_color[feature_size])
        
    #     plt.xscale('log')
    #     plt.xticks([0.1, 1, 2, 3], ['0.1', '1', '2', '3'])
    #     plt.xlabel('Temperature')
    #     plt.ylabel('Max Validation Accuracy (%)')
    #     # plt.ylim(0, 100) 

    #     plt.title(f'Fitted Max Validation Accuracy vs Temperature ({fit_type} fit)')
    #     plt.legend(title='Feature Size', loc='best')
    #     plt.grid(True)

    #     output_image_path = os.path.join(directory, f'max_accuracy_vs_temperature_{fit_type}_fit.png')
    #     plt.savefig(output_image_path)
    #     plt.close()
    #     print(f"{fit_type.capitalize()} fit plot saved to {output_image_path}")



# Usage example:
    

summary_file = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_classification/summary_by_size.txt'
# directory = '/home/alabutaleb/Desktop/confirmation/plot_all/plot_all_kd_classification'
data_by_temperature = create_data_by_temperature_from_summary(summary_file)
directory = os.path.dirname(summary_file)  # Use the same directory as the summary file

# To use the plot_fitted_lines function, ensure you have prepared the `data` list and `feature_size_to_color` dictionary as before
plot_fitted_lines(data_by_temperature, directory)
plot_fitted_lines_with_data(data_by_temperature, directory)
