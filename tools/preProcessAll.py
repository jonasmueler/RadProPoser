import os
import torch

# Define the preprocessing function
def preProcess(x: torch.Tensor):
    """
    Preprocess the tensor by centering and applying sequential FFTs.

    Args:
        x (torch.Tensor): Input tensor to preprocess.

    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    x = x.unsqueeze(dim = 0)
    x = x - torch.mean(x, dim=-1, keepdim=True)  # Center the tensor
    # Apply sequential FFTs
    x = torch.fft.fft(torch.fft.fft(torch.fft.fft(torch.fft.fft(x, dim=-1, norm="forward"), dim=-2, norm="forward"), dim=-3, norm="forward"), dim=-4, norm="forward")
    x = torch.fft.fftshift(x, dim=(-4, -3, -2, -1))
    x = x.permute(0, 1, 5, 2, 3, 4)  # Permute dimensions
    return x

# Script to process all tensors in a directory
def process_tensors(input_dir, output_dir):
    """
    Apply preprocessing to all tensor files in a directory and save the results in order.

    Args:
        input_dir (str): Path to the input directory containing .pth tensor files.
        output_dir (str): Path to the output directory where preprocessed tensors will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get and sort the file names numerically
    file_names = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".pth")],
        key=lambda x: int(os.path.splitext(x)[0])  # Sort by numeric value of file name
    )

    counter = 0
    # Iterate through sorted files
    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        #print(f"Processing {file_name}...")

        # Load the tensor
        tensor = torch.load(input_path)

        # Apply preprocessing
        processed_tensor = preProcess(tensor)

        # Save the processed tensor
        torch.save(processed_tensor, output_path)
        #print(f"Saved preprocessed tensor to {output_path}")

        counter += 1

        if counter % 100 == 0:
            print(counter)

# Example usage
if __name__ == "__main__":
    input_directory = "/home/jonas/data/radarPose/radar_raw_data/train/radar"  # Replace with the path to your input directory
    output_directory = "/home/jonas/data/radarPose/radar_raw_data/train/radar_fft"  # Replace with the path to your output directory
    process_tensors(input_directory, output_directory)