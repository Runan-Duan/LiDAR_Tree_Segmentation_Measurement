import argparse
import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def laz_to_pcd(laz_file, pcd_file):
    # Read the .laz file
    las = laspy.read(laz_file)
    # Extract the point coordinates (X, Y, Z)
    points = np.vstack([las.x, las.y, las.z]).transpose()
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(pcd_file, pcd)
    print(f"Converted {laz_file} to {pcd_file}")

# Downsample the point cloud
def downsample_point_cloud(pcd, voxel_size=0.1):
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    return pcd_downsampled

# Remove statistical outliers
def remove_statistical_outliers(pcd, nb_neighbors=50, std_ratio=1.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_cleaned = pcd.select_by_index(ind)
    return pcd_cleaned

# Ground Removal
def remove_ground(pcd, distance_threshold=0.01):
    # Perform plane segmentation (RANSAC)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, 
                                             ransac_n=3, 
                                             num_iterations=1000)
    # Extract the ground plane
    cloud_no_ground = pcd.select_by_index(inliers, invert=True)
    return cloud_no_ground

# Tree Segmentation (Using DBSCAN)
def segment_trees(pcd, eps=0.2, min_points=10):
    # Convert to numpy for DBSCAN
    points = np.asarray(pcd.points)
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    # Create a list of segmented clusters
    labels = clustering.labels_
    unique_labels = set(labels)
    clusters = []
    
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_indices = np.where(labels == label)[0]
            cluster_points = points[cluster_indices]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            clusters.append(cluster_pcd)
    return clusters

# Measure Tree Height
def measure_tree_height(tree_pcd):
    # Estimate the ground plane using RANSAC
    plane_model, inliers = tree_pcd.segment_plane(distance_threshold=0.2, 
                                                  ransac_n=3, 
                                                  num_iterations=1000)
    # Extract points in the tree cluster
    tree_points = np.asarray(tree_pcd.points)
    # The ground plane is defined by the equation ax + by + cz + d = 0
    # The height is measured as the max Z value minus the ground plane Z
    a, b, c, d = plane_model
    ground_z = -(d) / c  # Ground plane Z at x = 0, y = 0
    # Measure the maximum height in the cluster
    max_height = np.max(tree_points[:, 2]) - ground_z
    return max_height

# Save Tree Heights to a File
def save_heights_to_file(tree_heights, filename="tree_heights.txt"):
    with open(filename, "w") as f:
        for i, height in enumerate(tree_heights):
            f.write(f"Tree {i+1} height: {height:.2f} meters\n")
    print(f"Tree heights saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Tree Analysis Workflow ')
    parser.add_argument('--input_file(laz)', type=str,
                        help='Automatic detection of trees in a 3D point cloud and derivation of individual heights')
    parser.add_argument('--output_file(pcd)', type=str,
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    
    # Convert files
    laz_to_pcd("input_file.laz", "output_file.pcd")
    # Load the point cloud
    pcd = o3d.io.read_point_cloud("output_file.pcd")
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    # Preprocessing (downsampling, outlier removal)
    pcd_downsampled = downsample_point_cloud(pcd, voxel_size=0.1)
    o3d.visualization.draw_geometries([pcd_downsampled])
    pcd_cleaned = remove_statistical_outliers(pcd_downsampled)
    o3d.visualization.draw_geometries([pcd_cleaned])
    # Ground Removal
    pcd_no_ground = remove_ground(pcd_cleaned)
    o3d.visualization.draw_geometries([pcd_no_ground])
    # Segment trees using DBSCAN
    tree_clusters = segment_trees(pcd_no_ground)
    o3d.visualization.draw_geometries(tree_clusters)
    # Measure tree height
    tree_heights = []
    for i, tree_cluster in enumerate(tree_clusters):
        height = measure_tree_height(tree_cluster)
        tree_heights.append(height)
        print(f"Tree {i+1} height: {height:.2f} meters")
    # Save files
    save_heights_to_file(tree_heights)


if __name__ == '__main__':
    main()
