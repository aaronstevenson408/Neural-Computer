import random
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define the number of data points you want to generate
num_data_points = 1000
batch_size = 32
data_file = "nor_dataset.pkl"

# Create a custom dataset class
class NORGateDataset(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points
        self.data = self.load_data() if os.path.exists(data_file) else self.generate_data()

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.data[idx]

    def generate_data(self):
        data = []

        # Generate random examples
        for _ in range(self.num_points // 2):
            input1 = random.uniform(0, 1)
            input2 = random.uniform(0, 1)

            if input1 == 0 and input2 == 0:
                output_val = 1
            elif (input1 == 0 or input1 == 1) and (input2 == 0 or input2 == 1):
                output_val = 0
            else:
                output_val = -1  # Use -1 to represent errors

            data.append(((input1, input2), output_val))

        # Generate known true examples
        for _ in range(self.num_points // 4):
            data.append(((0, 0), 1))

        # Generate known false examples
        for _ in range(self.num_points // 4):
            data.append(((0, 1), 0))
            data.append(((1, 0), 0))
            data.append(((1, 1), 0))

        # Shuffle the data to mix examples
        random.shuffle(data)

        # Save the generated data to a file
        with open(data_file, 'wb') as file:
            pickle.dump(data, file)

        return data

    def load_data(self):
        # Load data from a file
        with open(data_file, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data

# Define the neural network architecture
class StrictNORModel(nn.Module):
    def __init__(self, input_size):
        super(StrictNORModel, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 1)  # Output size is 1 for binary classification

    def forward(self, x):
        # Define the forward pass
        out = self.fc1(x)
        return out

# Create a custom collate function for the DataLoader
def custom_collate(batch):
    inputs, outputs = zip(*batch)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)  # Use dtype=float32 for -1

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)

            # Convert predictions to binary (0 or 1) using a threshold (e.g., 0.5)
            predictions = (outputs >= 0.5).float()

            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.tolist())

    accuracy = accuracy_score(all_targets, all_predictions)
    return accuracy

# Function to plot training loss
def plot_training_loss(losses):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()

# Training loop function
def train_model(model, dataloader, num_epochs, criterion, optimizer):
    losses = []  # List to store loss values for plotting

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize loss for this epoch
        for inputs, targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets.view(-1, 1))  # Reshape targets for binary classification
            epoch_loss += loss.item()  # Accumulate loss for this epoch

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate the average loss for this epoch
        avg_epoch_loss = epoch_loss / len(dataloader)

        # Append the average epoch loss to the list for plotting
        losses.append(avg_epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_epoch_loss:.4f}")

    return losses

# Create the custom dataset
custom_dataset = NORGateDataset(num_data_points)

# Verification: Check if data was loaded or generated
if os.path.exists(data_file):
    print("Data loaded from file.")
else:
    print("Data generated and saved to file.")

# Create a DataLoader with the custom collate function
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=custom_collate)

# Instantiate the model
input_size = 2  # Number of input features (2 for NOR gate)
model = StrictNORModel(input_size)

# Define the loss function (Binary Cross-Entropy Loss) and optimizer (Adam)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 10  # Number of training epochs

# Call the training loop function
losses = train_model(model, dataloader, num_epochs, criterion, optimizer)

# Plot the training loss
plot_training_loss(losses)

# Call the evaluation function
accuracy = evaluate_model(model, dataloader)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "strict_nor_model.pth")

# Now, you have a complete script with all functions and dataloader creation.
