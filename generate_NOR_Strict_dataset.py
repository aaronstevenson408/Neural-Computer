import random
import torch
import pickle
import os
import datetime

# Function to generate the NOR Strict dataset
def generate_NOR_Strict_dataset(num_data_points):
    data = []

    # Generate random examples
    for _ in range(num_data_points // 2):
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
    for _ in range(num_data_points // 4):
        data.append(((0, 0), 1))

    # Generate known false examples
    for _ in range(num_data_points // 4):
        data.append(((0, 1), 0))
        data.append(((1, 0), 0))
        data.append(((1, 1), 0))

    # Shuffle the data to mix examples
    random.shuffle(data)

    return data

def save_dataset(data, data_file):
    # Save the dataset to a file
    with open(data_file, 'wb') as file:
        pickle.dump(data, file)

def main():
    data_file = "NOR_Strict_dataset.pkl"

    # Check if the dataset file already exists
    if os.path.exists(data_file):
        update_dataset = input("Dataset file already exists. Do you want to update it? (yes/no): ").lower()
        if update_dataset in ('yes', 'y'):
            num_data_points = int(input("Enter the number of data points to generate or update with: "))
            # Generate or update the dataset
            data = generate_NOR_Strict_dataset(num_data_points)
            # Save the dataset
            save_dataset(data, data_file)
            print(f"Dataset updated and saved as {data_file}")
        else:
            print("Dataset not updated.")
    else:
        generate_new = input("Dataset file doesn't exist. Do you want to generate a new dataset? (yes/no): ").lower()
        if generate_new in ('yes', 'y'):
            num_data_points = int(input("Enter the number of data points to generate: "))
            # Generate the dataset
            data = generate_NOR_Strict_dataset(num_data_points)
            # Save the dataset
            save_dataset(data, data_file)
            print(f"New dataset generated and saved as {data_file}")
        else:
            custom_file = input("Enter the path to an existing dataset file or leave blank to exit: ").strip()
            if custom_file:
                data_file = custom_file
                if not os.path.exists(data_file):
                    print("File does not exist. Exiting.")
                    return
                else:
                    print(f"Using existing dataset file: {data_file}")
            else:
                print("Exiting without generating or using a dataset.")

if __name__ == "__main__":
    main()
