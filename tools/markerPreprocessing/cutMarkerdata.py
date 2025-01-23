import os
import numpy as np
import pandas as pd
import ezc3d

def get_all_files_in_directory(directory_path):
    """
    Retrieves all file names in the specified directory.

    Parameters:
        directory_path (str): Path to the directory.

    Returns:
        list: List of file names in the directory.
    """
    try:
        file_names = [entry for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))]
        return file_names
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing {directory_path}.")
        return []

def cut_marker_data(file_path, start_frame, end_frame):
    """
    Cuts marker data from a .c3d file based on the specified frame range.

    Parameters:
        file_path (str): Path to the .c3d file.
        start_frame (int): The starting frame for the cut.
        end_frame (int): The ending frame for the cut.

    Returns:
        np.ndarray: Cut marker data as a numpy array (frames, markers, xyz).
    """
    try:
        c3d_data = ezc3d.c3d(file_path)
        markers = c3d_data['data']['points'][:3, :, start_frame:end_frame] * 100
        markers = np.transpose(markers, axes=(2, 1, 0))  # (frames, markers, xyz)

        # check if tehy only contain one position set 
        if check_static_markers(markers):
            print("markers are static!")

        return markers
    except Exception as e:
        print(f"Failed to cut marker data from {file_path}: {e}")
        return None

def validate_array(arr):
    r = True
    # Check for NaN values
    if np.isnan(arr).any():
        r = False
    
    # Check shape
    if arr.ndim != 3 or arr.shape[1:] != (39, 3):
        r = False
    
    return r

def check_static_markers(marker_data):
    """
    Checks if marker positions are identical across time.

    Parameters:
        marker_data (numpy array): Shape (time, markers, coords).

    Returns:
        bool: True if all marker positions are identical across time, False otherwise.
    """
    return np.all(marker_data == marker_data[0])

def process_marker_files(csv_path, input_dir, output_dir):
    """
    Processes marker files by cutting data based on the frame range specified in a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file containing file names and frame ranges.
        input_dir (str): Directory containing the input .c3d files.
        output_dir (str): Directory to save the processed marker data as .npy files.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV file
    try:
        frame_data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV file {csv_path}: {e}")
        return

    for _, row in frame_data.iterrows():
        file_name = row.get('Filename')
        if file_name:
            file_name = os.path.splitext(file_name)[0] + ".c3d"  # Ensure file has .c3d extension

        start_frame = int(row.get('Start frame', 0))
        end_frame = int(row.get('End frame', -1))

        if not file_name:
            print("Missing filename in CSV row.")
            continue

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.npy")

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        # Process the marker data
        markers = cut_marker_data(input_path, start_frame, end_frame)
        if validate_array(markers) == False:
            continue
        if markers is not None:
            np.save(output_path, markers)
            print(f"Processed and saved: {output_path}")

# Example usage
if __name__ == "__main__":
    csv_file_path = "/home/jonas/code/bioMechRadar/mt_rawradar_humanpose/Code/Dataset_Scripts/data_handling/omc_frames_4_0.csv"
    input_directory = "/home/jonas/data/radarPose/optitrack_recordings"
    output_directory = "/home/jonas/data/radarPose/optitrack_data_cut"

    process_marker_files(csv_file_path, input_directory, output_directory)
