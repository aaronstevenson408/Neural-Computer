import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def generate_nor_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        input1 = np.random.randint(2)  # Generate random 0 or 1
        input2 = np.random.randint(2)  # Generate random 0 or 1
        nor_output = int(not (input1 or input2))  # Compute NOR gate output
        dataset.append((input1, input2, nor_output))

    return dataset

class NORModel(nn.Module):
    def __init__(self):
        super(NORModel, self).__init__()
        self.input_layer = nn.Linear(2, 1)  # Two input nodes, one output node

    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        return x

if __name__ == "__main__":
    num_samples = 100  # You can change this to your desired number of samples
    dataset = generate_nor_dataset(num_samples)
    print(dataset)
    
    # Create an instance of the NORModel
    model = NORModel()
    
    # Define the loss function (Binary Cross-Entropy Loss)
    criterion = nn.BCELoss()
    
    # Define the optimizer (SGD with a learning rate)
    learning_rate = 0.1  # You can adjust this as needed
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Create an empty list to store the loss values
    losses = []

    # Define the number of training epochs
    num_epochs = 1000  # You can adjust this as needed

    for epoch in range(num_epochs):
        total_loss = 0.0
        for input1, input2, target in dataset:
            # Convert inputs and targets to PyTorch tensors
            inputs = torch.tensor([[input1, input2]], dtype=torch.float32)  # Add an extra dimension
            targets = torch.tensor([[target]], dtype=torch.float32)  # Add an extra dimension

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate the average loss for this epoch
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}')

    # Plot the training loss
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
