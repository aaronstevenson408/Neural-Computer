import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os

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

# Function to list .pth files in the current directory
def list_pth_files():
    pth_files = [f for f in os.listdir() if f.endswith(".pth")]
    return pth_files

if __name__ == "__main__":
    # List .pkl files in the current directory
    pkl_files = list_pkl_files()

    # List .pth files in the current directory
    pth_files = list_pth_files()

    # Check if there are any .pkl files to choose from
    if not pkl_files:
        print("No .pkl files found in the current directory.")
    else:
        # Prompt the user to choose a .pkl file
        print("Choose a .pkl file to use as the dataset:")
        for i, pkl_file in enumerate(pkl_files):
            print(f"{i + 1}: {pkl_file}")

        choice_data = input("Enter the number corresponding to your choice: ")

        try:
            choice_data = int(choice_data)
            if 1 <= choice_data <= len(pkl_files):
                data_file = pkl_files[choice_data - 1]

                # Check if there are any .pth files to choose from
                if not pth_files:
                    print("No .pth files found in the current directory.")
                else:
                    # Prompt the user to choose a .pth file
                    print("Choose a .pth file to use as the model:")
                    for i, pth_file in enumerate(pth_files):
                        print(f"{i + 1}: {pth_file}")

                    choice_model = input("Enter the number corresponding to your choice: ")

                    try:
                        choice_model = int(choice_model)
                        if 1 <= choice_model <= len(pth_files):
                            model_file = pth_files[choice_model - 1]

                            # Load the chosen dataset
                            custom_dataset = NORGateDataset(data_file)

                            # Define batch size and create DataLoader
                            batch_size = 32
                            dataloader = DataLoader(custom_dataset, batch_size=batch_size)

                            # Instantiate the model
                            input_size = 2  # Number of input features (2 for NOR gate)
                            model = StrictNORModel(input_size)

                            # Load the trained model weights
                            model.load_state_dict(torch.load(model_file))

                            # Evaluation loop
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

                            # Calculate and print accuracy
                            accuracy = accuracy_score(all_targets, all_predictions)
                            print(f"Accuracy: {accuracy * 100:.2f}%")
                        else:
                            print("Invalid choice for the model.")
                    except ValueError:
                        print("Invalid choice for the model.")
            else:
                print("Invalid choice for the dataset.")
        except ValueError:
            print("Invalid choice for the dataset.")
