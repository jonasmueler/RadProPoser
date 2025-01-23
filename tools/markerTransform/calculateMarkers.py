import numpy as np
import os
import visualCheck

def extract_pattern_number(filename):
    try:
        parts = filename.split('_')
        # Find the parts that match the pattern
        p = next(int(part[1:]) for part in parts if part.startswith('p'))
        an = next(int(part[2:]) for part in parts if part.startswith('an'))
        ac = next(int(part[2:]) for part in parts if part.startswith('ac'))
        r = next(int(part[1:]) for part in parts if part.startswith('r'))
        return (p, an, ac, r)
    except (ValueError, StopIteration):
        # Fallback to alphabetical sorting if pattern can't be parsed
        return filename

def load_and_concat_npy_files(directory, act = "ac0", ang = "an1"):
    """
    Load all .npy files from a directory and concatenate them along the first dimension.

    Args:
        directory (str): Path to the directory containing .npy files.

    Returns:
        numpy.ndarray: Concatenated array from all .npy files.
    """
    # List all .npy files in the directory that contain 'ac0' in their name
    npy_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy') and act in f and ang in f],
        key=extract_pattern_number
    )
    
    if not npy_files:
        raise ValueError("No .npy files found in the directory.")
    
    # Load each .npy file into a list of arrays
    arrays = [np.load(file) for file in npy_files]
    
    # Concatenate all arrays along the first dimension
    concatenated_array = np.concatenate(arrays, axis=0)
    
    return concatenated_array


def calculate_offset_per_marker_per_frame(X, y):
    """
    X: skeleton, 
    y: marker

    """
    res = {}
    for m in range(y.shape[0]):
        diff = np.linalg.norm(X - y[m, :], axis = 1)
        referenceKeypoint = np.argmin(diff)
        res[str(m)] = [referenceKeypoint, X[referenceKeypoint] - y[m]]

    return res


def get_distances(ang = "an0"):
    # load data
    X = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_skeletons", "ac0", ang)
    y = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_markers", "ac0", ang)
    
    # calculate marker position
    d = {}
    for t in range(X.shape[0]):
        res = calculate_offset_per_marker_per_frame(X[t,...], y[t, ...])
        if t == 0:
            d = res.copy()
            
        # check if nearest centroid is the same
        elif t > 0:
            for i in range(len(X[t, :, 0])):
                # check if same index found as closest 
                if d[str(i)][0] == res[str(i)][0]:
                    # update distance
                    d[str(i)][1] = (d[str(i)][1] + res[str(i)][1])/2
                
                else: 
                    #print("index different!")
                    pass
            
    return d

def get_marker_from_keys(d, keys):
    # iterate over markers
    markers = [] 
    for k in d:
        key = keys[d[k][0],...]
        marker = key +  d[k][1]
        markers.append(marker)
    
    return np.stack(markers, axis = 0)

def get_markers_time(d, keysTime):
    markersTime = []
    for t in range(keysTime.shape[0]):
        markers = get_marker_from_keys(d, keysTime[t])
        markersTime.append(markers)

    out = np.stack(markersTime, axis = 0)

    return out



if __name__ == "__main__":
    d = get_distances(ang = "an0")

    # test on new keys
    X = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_skeletons", "ac8", "an1")
    y = load_and_concat_npy_files("/home/jonas/data/radarPose/radar_raw_data/matched_markers", "ac8", "an1")

    # get new markers
    yNew = get_markers_time(d, X)

    # plot 
    visualCheck.plot_skeleton_3d_with_points_time(X.reshape(X.shape[0], 26, 3), yNew.reshape(yNew.shape[0], 39, 3), save_path = os.getcwd() + "/test.mp4")


    

