import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
import os 
import datetime
import pandas as pd

# Define the NOR gate model
class NORModel(nn.Module):
    def __init__(self):
        super(NORModel, self).__init__()
        self.input_layer = nn.Linear(2, 1)  # Two input nodes, one output node

    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        return x
    
# Generate NOR gate dataset
def generate_nor_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        input1 = np.random.randint(2)  # Generate random 0 or 1
        input2 = np.random.randint(2)  # Generate random 0 or 1
        nor_output = int(not (input1 or input2))  # Compute NOR gate output
        dataset.append((input1, input2, nor_output))

    return dataset

def train_nor_model(model, dataset, num_epochs=100, learning_rate=0.1):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training_losses = []

    if len(dataset) == 0:
        return training_losses  # Return an empty list if the dataset is empty

    for epoch in range(num_epochs):
        total_loss = 0.0
        for input1, input2, target in dataset:  # Unpack (input1, input2, target) tuples
            inputs = torch.tensor([[input1, input2]], dtype=torch.float32)
            target = torch.tensor([[target]], dtype=torch.float32)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        training_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}')

    return training_losses


# Function to evaluate the model and save pass/fail results to a CSV file
def evaluate_nor_model_and_save_results(model, dataset, pass_fail_file):
    criterion = nn.BCELoss()
    
    # Initialize variables to track results
    test_loss = 0.0
    correct_predictions = 0

    # Open the pass/fail CSV file and write results
    os.makedirs(os.path.dirname(pass_fail_file), exist_ok=True)
    with open(pass_fail_file, 'w', newline='') as pass_fail_csv:
        pass_fail_writer = csv.writer(pass_fail_csv)
        pass_fail_writer.writerow(['Input1', 'Input2', 'Target', 'Prediction', 'Pass/Fail'])

        for input1, input2, target in dataset:
            inputs = torch.tensor([[input1, input2]], dtype=torch.float32)
            target = torch.tensor([[target]], dtype=torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            test_loss += loss.item()

            # Check if the model's prediction matches the target
            prediction = (outputs >= 0.5).float()
            correct = (prediction == target).all().item()

            # Write results to the pass/fail CSV file
            pass_fail_writer.writerow([input1, input2, target.item(), prediction.item(), 'Pass' if correct else 'Fail'])

            correct_predictions += correct

    # Calculate and print test loss and accuracy
    test_loss /= len(dataset)
    test_accuracy = (correct_predictions / len(dataset)) * 100

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy

def plot_losses_and_save(training_losses, labels=None, title=None, save_file=None):
    plt.figure(figsize=(8, 6))
    plt.plot(training_losses, label=labels[0] if labels else 'Training Loss', linewidth=2)
    
    if labels:
        plt.legend()
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if title:
        plt.title(title)
    
    if save_file:
        # Ensure the directory exists before saving the file
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, format='png')
    
    plt.close()

def calculate_efficiency_score(test_accuracy, learning_rate, training_loss, test_loss, num_train_samples, num_epochs):
    # Calculate the efficiency score based on your criteria
    efficiency_score = (
        test_accuracy +
        (1 / (1 + learning_rate)) +
        (1 / (1 + training_loss)) +
        (1 / (1 + test_loss)) +
        (1 / (1 + num_train_samples)) +
        (1 / (1 + num_epochs))
    )
    return efficiency_score

def setup_and_run_experiment(output_file, num_test_samples, num_train_samples_range, num_epochs_range, learning_rate_range,current_time):

    # Add the timestamp to the output file name
    output_file = f'{output_file}_{current_time}.csv'
    
    # Open the file and write the header if the file doesn't exist
    with open(output_file, 'w', newline='') as results_file:
        results_writer = csv.writer(results_file)
        header = ['Num_Train_Samples', 'Num_Epochs', 'Learning_Rate', 'Training_Loss', 'Test_Loss', 'Test_Accuracy', 'Efficiency_Score']
        results_writer.writerow(header)
            
    for num_train_samples in num_train_samples_range:
        for num_epochs in num_epochs_range:
            for learning_rate in learning_rate_range:
                # Construct the save file names based on the parameters
                pass_fail_file = f'passFail/pass_fail_{num_train_samples}_{num_epochs}_{learning_rate}.csv'
                plot_file = f'plot/plot_{num_train_samples}_{num_epochs}_{learning_rate}.png'

                # Call the main function with the correct values, including pass_fail_file and plot_file
                training_loss, test_loss, test_accuracy = main(num_train_samples, num_test_samples, num_epochs, learning_rate, output_file,  pass_fail_file, plot_file)

                # Calculate the efficiency score
                efficiency_score = calculate_efficiency_score(test_accuracy, learning_rate, training_loss, test_loss, num_train_samples, num_epochs)

                # Open the file again and write the results, including the efficiency score
                with open(output_file, 'a', newline='') as results_file:
                    results_writer = csv.writer(results_file)
                    results_writer.writerow([num_train_samples, num_epochs, learning_rate, training_loss, test_loss, test_accuracy, efficiency_score])

    # Main function
def sort_models(data, accuracy_threshold=100.0):
    # Filter out models with test accuracy below the threshold
    filtered_data = data[data['Test_Accuracy'] >= accuracy_threshold]
    
    # Sort the filtered DataFrame based on the efficiency score in descending order
    sorted_data = filtered_data.sort_values(by='Efficiency_Score', ascending=False)
    
    return sorted_data

## TODO: fix the stress testing  
def stress_test_models(data, num_stress_tests):
    # Create a new DataFrame to store the stress test results
    stress_test_results = pd.DataFrame(columns=data.columns)

    for index, row in data.iterrows():
        # Perform stress tests for each model
        is_stress_test_passed = True
        for _ in range(num_stress_tests):
            # Call the main function with the model's parameters
            num_train_samples = row['Num_Train_Samples']
            num_epochs = row['Num_Epochs']
            learning_rate = row['Learning_Rate']
            output_file = 'stress_test_results.csv'  # Adjust the output file name
            pass_fail_file = 'stress_test_pass_fail.csv'  # Adjust the pass/fail file name
            plot_file = 'stress_test_plot.png'  # Adjust the plot file name

            training_loss, test_loss, test_accuracy = main(num_train_samples, num_test_samples, num_epochs, learning_rate, output_file, pass_fail_file, plot_file)

            # If the model is not 100% accurate in any stress test, mark it as failed
            if test_accuracy < 100.0:
                is_stress_test_passed = False
                break
        
        # If the model passed all stress tests, record its data in the results DataFrame
        if is_stress_test_passed:
            stress_test_results = stress_test_results.append(row, ignore_index=True)
    
    return stress_test_results
def main(num_train_samples, num_test_samples, num_epochs, learning_rate, output_file=None, pass_fail_file=None, plot_file=None):    
    # Generate the training dataset
    train_dataset = generate_nor_dataset(num_train_samples)

    # Generate the test dataset
    test_dataset = generate_nor_dataset(num_test_samples)

    # Create the NORModel
    model = NORModel()

    # Train the model and get the training losses
    training_losses = train_nor_model(model, train_dataset, num_epochs=num_epochs, learning_rate=learning_rate)

    # Evaluate the model on the test dataset and save results to a file
    test_loss, test_accuracy = evaluate_nor_model_and_save_results(model, test_dataset,pass_fail_file)

    #print(f'Test Loss: {test_loss:.4f}')
    #print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Plot only the training losses with a custom title and save as PNG
    plot_losses_and_save(training_losses, labels=['Training Loss'], title=f'Training Loss (Samples: {num_train_samples}, Epochs: {num_epochs}, LR: {learning_rate})', save_file=plot_file)
    
    # Return the test loss and test accuracy
    return training_losses[-1], test_loss, test_accuracy

if __name__ == "__main__":
    num_test_samples = 100  # Constant for the number of test samples
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Define ranges for training samples, epochs, and learning rates
    num_train_samples_range = range(1, 101, 20)
    num_epochs_range = range(1, 10, 2)
    learning_rate_range = [round(0.1 * i, 2) for i in range(1, 9)]  # [0.1, 0.2, 0.3, 0.4, 0.5]

    # Specify the output file name
    output_file_name = 'results'

    setup_and_run_experiment(output_file_name, num_test_samples, num_train_samples_range, num_epochs_range, learning_rate_range,current_time)

    # Load the CSV data into a DataFrame
    data = pd.read_csv(f'{output_file_name}_{current_time}.csv')

    # Add the Efficiency_Score column to the DataFrame
    data['Efficiency_Score'] = calculate_efficiency_score(data['Test_Accuracy'], data['Learning_Rate'], data['Training_Loss'], data['Test_Loss'], data['Num_Train_Samples'], data['Num_Epochs'])

    # Sort the models based on efficiency score
    sorted_data = sort_models(data)
    print(sorted_data)
    #TODO: fix stress testing
    # Stress test the models and get the results
    #stress_test_results = stress_test_models(sorted_data, num_stress_tests=100)

    # Print or access the stress test results
    # print("Stress Test Results:")
    # print(stress_test_results)