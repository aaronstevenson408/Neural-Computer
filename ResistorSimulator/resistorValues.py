import csv
import random
from datetime import datetime

def generate_resistor_values(e_series, magnitudes):
    all_resistor_values = []
    for magnitude in magnitudes:
        resistor_values = [round(value * magnitude, 4) for value in e_series]
        all_resistor_values.extend(resistor_values)
    return all_resistor_values

def update_e_series(e_series, magnitudes):
    updated_series = []
    for series in e_series:
        series_name = series['name']
        series_tolerance = series['tolerance']
        series_values = generate_resistor_values(series['values'], magnitudes)
        
        updated_series.append({
            'name': series_name,
            'tolerance': series_tolerance,
            'values': series_values
        })
    
    return updated_series

def generate_random_value(ideal_ohm, tolerance_percentage):
    min_value = ideal_ohm - (ideal_ohm * tolerance_percentage / 100)
    max_value = ideal_ohm + (ideal_ohm * tolerance_percentage / 100)
    return round(random.uniform(min_value, max_value), 4)  # Round to 4 decimal places

def generate_csv(selected_series, num_simulated_components):
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"resistor_simulation_{date}_{','.join(selected_series)}.csv"

    with open(filename, mode='w', newline='') as file:
        fieldnames = ["Series", "Tolerance Percentage", "Ideal Ohm"] + [f"Sim_{i}" for i in range(1, num_simulated_components + 1)]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for series_name in selected_series:
            for series in updated_e_series:
                if series['name'] == series_name:
                    for value in series['values']:
                        row_data = {
                            "Series": series_name,
                            "Tolerance Percentage": series['tolerance'],
                            "Ideal Ohm": value
                        }
                        for i in range(1, num_simulated_components + 1):
                            simulated_value = generate_random_value(value, series['tolerance'])
                            row_data[f"Sim_{i}"] = simulated_value
                        writer.writerow(row_data)

# Define the E-Series and magnitudes
e_series = [
    {"name": "E3", "tolerance": 0.4, "values": [1.0, 2.2, 4.7]},
    {"name": "E6", "tolerance": 0.2, "values": [1.0, 1.5, 2.2, 3.3, 4.7, 6.8]},
    {"name": "E12", "tolerance": 0.1, "values": [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]},
    {"name": "E24", "tolerance": 0.05, "values": [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]},
    {"name": "E48", "tolerance": 0.02, "values": [1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69, 1.78, 1.87, 1.96, 2.05, 2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01, 3.16, 3.32, 3.48, 3.65, 3.83, 4.02, 4.22, 4.42, 4.64, 4.87, 5.11, 5.36, 5.62, 5.90, 6.19, 6.49, 6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53]},
    {"name": "E96", "tolerance": 0.01, "values": [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76]},
    {"name": "E192", "tolerance": 0.005, "values": [1.00, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.11, 1.13, 1.14, 1.15, 1.17, 1.18, 1.20, 1.21, 1.23, 1.24, 1.26, 1.27, 1.29, 1.30, 1.32, 1.33, 1.35, 1.37, 1.38, 1.40, 1.42, 1.43, 1.45, 1.47, 1.49, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.65, 1.67, 1.69, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.87, 1.89, 1.91, 1.93, 1.96, 1.98, 2.00, 2.03, 2.05, 2.08, 2.10, 2.13, 2.15, 2.18, 2.21, 2.23, 2.26, 2.29, 2.32, 2.34, 2.37, 2.40, 2.43, 2.46, 2.49, 2.52, 2.55, 2.58, 2.61, 2.64, 2.67, 2.71, 2.74, 2.77, 2.80, 2.84, 2.87, 2.91, 2.94, 2.98, 3.01, 3.05, 3.09, 3.12, 3.16, 3.20, 3.24, 3.28, 3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.57, 3.61, 3.65, 3.70, 3.74, 3.79, 3.83, 3.88, 3.92, 3.97, 4.02, 4.07, 4.12, 4.17, 4.22, 4.27, 4.32, 4.37, 4.42, 4.48, 4.53, 4.59, 4.64, 4.70, 4.75, 4.81, 4.87, 4.93, 4.99, 5.05, 5.11, 5.17, 5.23, 5.30, 5.36, 5.42, 5.49, 5.56, 5.62, 5.69, 5.76, 5.83, 5.90, 5.97, 6.04, 6.12, 6.19, 6.26, 6.34, 6.42, 6.49, 6.57, 6.65, 6.73, 6.81, 6.90, 6.98, 7.06, 7.15, 7.23, 7.32, 7.41, 7.50, 7.59, 7.68, 7.77, 7.87, 7.96, 8.06, 8.16, 8.25, 8.35, 8.45, 8.56, 8.66, 8.76, 8.87, 8.98, 9.09, 9.20, 9.31, 9.42, 9.53, 9.65, 9.76, 9.88]}
]
magnitudes = [0.01, 0.1, 1, 100, 1000, 1000000]

# Update the E-Series dictionary
updated_e_series = update_e_series(e_series, magnitudes)

# Ask the user for input
selected_series = input("Enter the series you want to simulate (comma-separated): ").split(',')
num_simulated_components = int(input("Enter the number of simulated components per series: "))

# Generate the CSV file
generate_csv(selected_series, num_simulated_components)
