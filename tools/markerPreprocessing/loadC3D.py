import ezc3d # Path to your C3D file 
file_path = "/home/jonas/data/radarPose/optitrack_recordings/2024-02-23-09-35-10_s1a0_0.c3d"# Load the C3D file 
c3d_data = ezc3d.c3d(file_path) # Display general information about the file
#print(c3d_data)
markers = c3d_data['data']['points'][:3, ...] * 100
print(markers)
print(markers.shape)