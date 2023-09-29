import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

# Define the neural network architecture
class StrictNORModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(StrictNORModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the custom dataset class
class NORGateDataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_values, output_val = self.data[idx]
        return torch.tensor(input_values, dtype=torch.float32), torch.tensor(output_val, dtype=torch.float32)

    def load_data(self):
        # Load data from a file
        with open(self.data_file, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data

# Function to list .pkl files in the current directory
def list_pkl_files():
    pkl_files = [f for f in os.listdir() if f.endswith(".pkl")]
    return pkl_files

# Function to train the model
def train_model(model, dataloader, num_epochs, criterion, optimizer):
    losses = []  # List to store loss values for plotting

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize loss for this epoch
        for inputs, targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Ensure the target labels are of data type long
            targets = targets.long()

            # Calculate the loss
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()  # Accumulate loss for this epoch

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate the average loss for this epoch
        avg_epoch_loss = epoch_loss / len(dataloader)

        # Append the average epoch loss to the list for plotting
        losses.append(avg_epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_epoch_loss:.4f}")

    return losses



# Main training script
if __name__ == "__main__":
    # List .pkl files in the current directory
    pkl_files = list_pkl_files()

    # Check if there are any .pkl files to choose from
    if not pkl_files:
        print("No .pkl files found in the current directory.")
    else:
        # Prompt the user to choose a .pkl file
        print("Choose a .pkl file to use as the dataset:")
        for i, pkl_file in enumerate(pkl_files):
            print(f"{i + 1}: {pkl_file}")

        choice = input("Enter the number corresponding to your choice: ")
        choice = int(choice)
        if 1 <= choice <= len(pkl_files):
            data_file = pkl_files[choice - 1]

            # Load the chosen dataset
            custom_dataset = NORGateDataset(data_file)

            # Display the number of training data points
            num_data_points = len(custom_dataset)
            print(f"Number of training data points: {num_data_points}")

            # Define batch size and create DataLoader
            batch_size = 32
            dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

            # Instantiate the model
            input_size = 2  # Number of input features (2 for NOR gate)
            hidden_size = 64  # Number of hidden units in the model
            num_classes = 3  # Number of output classes (0, 1, -1)
            model = StrictNORModel(input_size, hidden_size, num_classes)


            # Define the loss function (Binary Cross-Entropy Loss) and optimizer (Adam)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training parameters
            num_epochs = 10  # Number of training epochs

            # Call the training loop function
            losses = train_model(model, dataloader, num_epochs, criterion, optimizer)

            # Plot the training loss
            plt.figure()
            plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('Training Loss Over Epochs')
            plt.grid(True)

            # Save the training loss plot with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            loss_plot_filename = f"training_loss_{timestamp}.png"
            plt.savefig(loss_plot_filename)
            plt.close()

            # Save the trained model with a timestamp
            model_filename = f"strict_nor_model_{timestamp}.pth"
            torch.save(model.state_dict(), model_filename)

            print(f"Training loss plot saved as {loss_plot_filename}")
            print(f"Trained model saved as {model_filename}")
        else:
            print("Invalid choice.")

