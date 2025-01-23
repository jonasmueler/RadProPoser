import os
import re
import matplotlib.pyplot as plt
from collections import Counter

def analyze_directory_and_save_plot(directory_path, save_path="distribution_plot.png"):
    """
    Analyzes files in the given directory and saves distributions of actions and angles as a plot.

    Parameters:
        directory_path (str): Path to the directory containing the files.
        save_path (str): Path to save the plot. Default is "distribution_plot.png".
    """
    # Regex to extract participant, angle, action, and repetition
    pattern = r"markers_p(?P<participant>\d+)_an(?P<angle>\d+)_ac(?P<action>\d+)_r(?P<repetition>\d+).npy"
    
    # Initialize counters
    participants = []
    angles = []
    actions = []
    repetitions = []
    
    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        match = re.match(pattern, file_name)
        if match:
            participants.append(match.group("participant"))
            angles.append(match.group("angle"))
            actions.append(match.group("action"))
            repetitions.append(match.group("repetition"))
    
    # Count occurrences
    angle_counts = Counter(angles)
    action_counts = Counter(actions)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Action distribution
    axes[0].bar(action_counts.keys(), action_counts.values(), color='blue', alpha=0.7)
    axes[0].set_title("Action Distribution")
    axes[0].set_xlabel("Actions")
    axes[0].set_ylabel("Frequency")
    
    # Angle distribution
    axes[1].bar(angle_counts.keys(), angle_counts.values(), color='green', alpha=0.7)
    axes[1].set_title("Angle Distribution")
    axes[1].set_xlabel("Angles")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# Example usage
directory_path = "/home/jonas/data/radarPose/radar_raw_data/matched_markers"  # Replace with the path to your directory
save_path = "distribution_plot.png"  # Replace with your desired save path
analyze_directory_and_save_plot(directory_path, save_path)