import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
import os 

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

# Train the NOR gate model
def train_nor_model(model, dataset, num_epochs=100, learning_rate=0.1):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training_losses = []

    if len(dataset) == 0:
        return training_losses  # Return an empty list if the dataset is empty

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataset:
            inputs = torch.tensor([[inputs[0], inputs[1]]], dtype=torch.float32)  # Add an extra dimension
            targets = torch.tensor([[targets]], dtype=torch.float32)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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


# Main function
def main(num_train_samples, num_test_samples, num_epochs=100, learning_rate=0.1, output_file = None, pass_fail_file=None, plot_file=None):
    num_epochs = int(num_epochs)
    
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

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Plot only the training losses with a custom title and save as PNG
    plot_losses_and_save(training_losses, labels=['Training Loss'], title=f'Training Loss (Samples: {num_train_samples}, Epochs: {num_epochs}, LR: {learning_rate})', save_file=plot_file)
    
    # Return the test loss and test accuracy
    return test_loss, test_accuracy

if __name__ == "__main__":
    num_test_samples = 100  # Constant for the number of test samples

    # Define ranges for training samples, epochs, and learning rates
    num_train_samples_range = range(0, 1001, 10)
    num_epochs_range = range(0, 101, 10)
    learning_rate_range = [0.1 * i for i in range(1, 6)]  # [0.1, 0.2, 0.3, 0.4, 0.5]

    
    output_file = f'results.csv'
    # Open the file and write the header if the file doesn't exist
    with open(output_file, 'w', newline='') as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow(['Num_Train_Samples', 'Num_Epochs', 'Learning_Rate', 'Test_Loss', 'Test_Accuracy'])
    
    # Close the file before calling main
    results_file.close()
    
    for num_train_samples in num_train_samples_range:
        for num_epochs in num_epochs_range:
            for learning_rate in learning_rate_range:
                # Construct the save file names based on the parameters
                pass_fail_file = f'passFail/pass_fail_{num_train_samples}_{num_epochs}_{learning_rate}.csv'
                plot_file = f'plot/plot_{num_train_samples}_{num_epochs}_{learning_rate}.png'

                # Call the main function with the correct values, including pass_fail_file and plot_file
                test_loss, test_accuracy = main(num_train_samples, num_test_samples, num_epochs, learning_rate, output_file,  pass_fail_file, plot_file)


                # Open the file again and write the results
                with open(output_file, 'a', newline='') as results_file:
                    results_writer = csv.writer(results_file)
                    results_writer.writerow([num_train_samples, num_epochs, learning_rate, test_loss, test_accuracy])