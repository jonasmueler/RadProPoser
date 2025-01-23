import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ezc3d # Path to your C3D file 
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import os


def plot_skeleton_3d_with_points(joint_data, overlay_points=None, overlay_color='red'):
    """
    Plots a 3D skeleton for a single frame and overlays additional points in a different color.
    
    Parameters:
        joint_data (numpy array): Shape (joints=26, coords=3) representing joint positions in 3D space.
        overlay_points (numpy array, optional): Shape (num_points, 3) representing additional points to plot. Default is None.
        overlay_color (str): Color for the overlay points. Default is 'red'.
    """
    # Define joint connections (bones)
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),   # Right leg to toes
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),  # Left leg to toes
        (0, 11), (11, 12), (12, 23), (23, 24), (24, 25),  # Spine to neck and head
        (23, 13), (13, 14), (14, 15), (15, 16), (16, 17),  # Right arm
        (23, 18), (18, 19), (19, 20), (20, 21), (21, 22)   # Left arm
    ]

    # Initialize the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set limits for better visualization
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-1, 200)
    ax.set_title('3D Skeleton with Overlay Points')

    # Plot the skeleton
    for bone in bones:
        joint1, joint2 = bone
        x_values = [joint_data[joint1][0], joint_data[joint2][0]]
        y_values = [joint_data[joint1][1], joint_data[joint2][1]]
        z_values = [joint_data[joint1][2], joint_data[joint2][2]]
        ax.plot(x_values, y_values, z_values, 'bo-', linewidth=2, markersize=4)  # Blue lines and points

    # Overlay additional points if provided
    if overlay_points is not None:
        ax.scatter(overlay_points[:, 0], overlay_points[:, 1], overlay_points[:, 2],
                   c=overlay_color, label='Overlay Points', s=5)

    # Add a legend if overlay points are plotted
    if overlay_points is not None:
        ax.legend()

    # Show the plot
    plt.show()


def plot_skeleton_3d_with_points_time(joint_data, overlay_points=None, overlay_color='red', interval=100, save_path=None):
    """
    Plots a 3D skeleton for a sequence of frames and overlays additional points in a different color.

    Parameters:
        joint_data (numpy array): Shape (time, joints=26, coords=3) representing joint positions in 3D space over time.
        overlay_points (numpy array, optional): Shape (time, num_points, 3) representing additional points to plot. Default is None.
        overlay_color (str): Color for the overlay points. Default is 'red'.
        interval (int): Interval between frames in milliseconds. Default is 100 ms.
        save_path (str, optional): Path to save the animation as a video file. If None, the video is not saved.
    """
    # Define joint connections (bones)
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),   # Right leg to toes
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),  # Left leg to toes
        (0, 11), (11, 12), (12, 23), (23, 24), (24, 25),  # Spine to neck and head
        (23, 13), (13, 14), (14, 15), (15, 16), (16, 17),  # Right arm
        (23, 18), (18, 19), (19, 20), (20, 21), (21, 22)   # Left arm
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-1, 200)
    ax.set_title('3D Skeleton with Overlay Points')

    skeleton_lines = []
    for _ in bones:
        line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=4)
        skeleton_lines.append(line)

    overlay_scatter = ax.scatter([], [], [], c=overlay_color, label='Overlay Points', s=5) if overlay_points is not None else None

    def update(frame):
        frame_joint_data = joint_data[frame]
        for line, bone in zip(skeleton_lines, bones):
            joint1, joint2 = bone
            x_values = [frame_joint_data[joint1][0], frame_joint_data[joint2][0]]
            y_values = [frame_joint_data[joint1][1], frame_joint_data[joint2][1]]
            z_values = [frame_joint_data[joint1][2], frame_joint_data[joint2][2]]
            line.set_data(x_values, y_values)
            line.set_3d_properties(z_values)

        if overlay_points is not None:
            frame_overlay_points = overlay_points[frame]
            overlay_scatter._offsets3d = (
                frame_overlay_points[:, 0],
                frame_overlay_points[:, 1],
                frame_overlay_points[:, 2]
            )

        return skeleton_lines + ([overlay_scatter] if overlay_scatter is not None else [])

    ani = FuncAnimation(fig, update, frames=joint_data.shape[0], interval=interval, blit=True)

    if save_path is not None:
        # Save the animation as a video file
        ani.save(save_path, writer=animation.FFMpegWriter(fps=1000 // interval))
        print(f"Video saved to {save_path}")

    plt.show()

    # Create the animation
    num_frames = joint_data.shape[0]
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    # Show the plot
    plt.show()


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



if __name__ == "__main__":
    # load marker
    markers = np.load("/home/jonas/data/radarPose/radar_raw_data/matched_markers/markers_p1_an0_ac6_r0.npy")

    # load keypoints
    keypoints = np.load("/home/jonas/data/radarPose/radar_raw_data/matched_skeletons/skeleton_p1_an0_ac6_r0.npy")


    print(markers.shape)
    print(keypoints.shape)

    # Example usage
    plot_skeleton_3d_with_points_time(keypoints, markers, save_path = os.getcwd() + "/test.mp4")
