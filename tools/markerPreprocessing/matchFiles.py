import os
import numpy as np
import pandas as pd
import ezc3d
import re
import os
import numpy as np
import open3d as o3d

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

def find_matching_markers(skeleton_file, marker_dir):
    """
    Finds all matching marker files for a given skeleton file.

    Parameters:
        skeleton_file (str): Name of the skeleton file (e.g., skeleton_p1_an0_ac0_r0.npy).
        marker_dir (str): Directory containing marker files.

    Returns:
        list: List of matching marker file names. Returns an empty list if no matches are found.
    """
    import re
    import os

    match = re.search(r"skeleton_p(\d+)_an(\d+)_ac(\d+)_r", skeleton_file)
    if not match:
        print(f"Invalid skeleton file format: {skeleton_file}")
        return []

    participant, angle, action = match.groups()
    marker_files = [f for f in os.listdir(marker_dir) if f.endswith(".npy")]

    matching_files = [
        marker_file for marker_file in marker_files
        if re.search(rf"s{participant}a{action}_{angle}\.npy", marker_file)
    ]

    return matching_files

def extract_without_skeleton(skeleton_file):

    """

    Extracts everything from the skeleton file name except the word 'skeleton'.



    Parameters:

        skeleton_file (str): Name or path of the skeleton file (e.g., skeleton_p1_an0_ac0_r1.npy).



    Returns:

        str: The extracted part of the name without 'skeleton', or None if 'skeleton' is not in the name.

    """

    match = re.search(r"skeleton_(.+)", skeleton_file)

    if match:

        return match.group(1)

    return None

def temporal_icp(keypoints, markers):
    aligned_markers = []
    prev_transformation = np.eye(4)  # Identity matrix as initial guess

    for t in range(keypoints.shape[0]):
        kp_frame = keypoints[t]  # Shape: (26, 3)
        markers_frame = markers[t]  # Shape: (39, 3)

        # Convert to Open3D point clouds
        kp_cloud = o3d.geometry.PointCloud()
        kp_cloud.points = o3d.utility.Vector3dVector(kp_frame)
        markers_cloud = o3d.geometry.PointCloud()
        markers_cloud.points = o3d.utility.Vector3dVector(markers_frame)

        # Perform ICP with the previous transformation as the initial guess
        threshold = 0.00002
        reg_result = o3d.pipelines.registration.registration_icp(
            markers_cloud, kp_cloud, threshold,
            init=prev_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Apply transformation to markers
        transformation = reg_result.transformation
        prev_transformation = transformation  # Update for next frame
        markers_cloud.transform(transformation)
        aligned_markers.append(np.asarray(markers_cloud.points))

    return np.array(aligned_markers)

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

def find_matches(skeleton_directory="/home/jonas/data/radarPose/radar_raw_data/skeletons",
                 marker_directory="/home/jonas/data/radarPose/optitrack_data_cut"):
    """
    Finds and processes matches for skeleton files in the skeleton directory.
    Compares the shape of each skeleton file with its matching marker file(s).
    """

    skeletons = get_all_files_in_directory(skeleton_directory)
    print("Skeleton files:", skeletons)

    for skeleton in skeletons:
        print(f"Processing skeleton file: {skeleton}")

        # Find matching marker files
        matches = find_matching_markers(skeleton, marker_directory)
        print(f"Matching marker files for {skeleton}: {matches}")

        if not matches:
            print(f"No matches found for skeleton file: {skeleton}")
            continue  # Skip to the next skeleton file

        # Load the skeleton data
        skeleton_path = os.path.join(skeleton_directory, skeleton)
        try:
            skeleton_data = np.load(skeleton_path) * 100
            skeleton_data = np.reshape(skeleton_data, (skeleton_data.shape[0], 26, 3))
            print(f"Loaded skeleton data shape: {skeleton_data.shape}")
        except Exception as e:
            print(f"Error loading skeleton file {skeleton}: {e}")
            continue

        # Process each matching marker file
        for match in matches:
            marker_path = os.path.join(marker_directory, match)
            
            marker_data = np.load(marker_path)
            print(f"Loaded marker data shape: {marker_data.shape}")

            if check_static_markers(marker_data):
                print("static")
            

            # Check shape compatibility
            if marker_data.shape[0] != skeleton_data.shape[0]:
                print(f"Shape mismatch: Skeleton {skeleton_data.shape}, Marker {marker_data.shape}")
                continue

            # Align markers to skeleton
            aligned_markers = temporal_icp(skeleton_data, marker_data)
            print(f"Aligned markers shape: {aligned_markers.shape}")
            #aligned_markers = marker_data
            

            # validate res 
            if validate_array(aligned_markers) == False:
                continue
            

            # Save results
            np.save(f"/home/jonas/data/radarPose/radar_raw_data/matched_skeletons/{skeleton}", skeleton_data)
            np.save(f"/home/jonas/data/radarPose/radar_raw_data/matched_markers/markers_{extract_without_skeleton(skeleton)}", aligned_markers)
            print(f"Saved aligned markers for {skeleton}.")
find_matches()

# Example usage
#if __name__ == "__main__":
#    skeleton_directory = "/home/jonas/data/radarPose/radar_raw_data/skeletons"
#    marker_directory = "/home/jonas/data/radarPose/optitrack_data_cut"#

#    check_skeletons_against_markers(skeleton_directory, marker_directory)