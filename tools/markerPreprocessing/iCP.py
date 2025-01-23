import open3d as o3d
import numpy as np

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

# Example input tensors
keypoints = np.random.rand(100, 26, 3)  # Shape: (time, 26, 3)
markers = np.random.rand(100, 39, 3)    # Shape: (time, 39, 3)

# Perform temporal ICP
aligned_markers = temporal_icp(keypoints, markers)
print(aligned_markers.shape)  # Shape: (time, 39, 3)
