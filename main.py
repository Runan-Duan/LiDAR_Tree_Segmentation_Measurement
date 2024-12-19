import argparse
import torch
import cuml
import cudf
from cuml.cluster import DBSCAN
import cupy as cp
import laspy
import numpy as np
import open3d as o3d


def laz_to_pcd(laz_file, pcd_file):
    """Convert file"""
    las = laspy.read(laz_file)  # Read the .laz file
    points = np.vstack([las.x, las.y, las.z]).transpose()  # Extract the point coordinates (X, Y, Z)
    pcd = o3d.geometry.PointCloud()  # Create an Open3D PointCloud object
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(pcd_file, pcd)  # Save the point cloud to a PCD file
    print(f"Converted {laz_file} to {pcd_file}")


def segment_trees(pcd, eps=0.2, min_points=10):
    """Tree Segmentation (Using DBSCAN)"""
    points = np.asarray(pcd.points)  # Convert to numpy for DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)  # Apply DBSCAN clustering
    labels = clustering.labels_  
    unique_labels = set(labels)
    clusters = []  # Create a list of segmented clusters
    
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_indices = np.where(labels == label)[0]
            cluster_points = points[cluster_indices]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            clusters.append(cluster_pcd)
    return clusters


def measure_tree_height(tree_pcd):
    """Measure Tree Height:
    - The ground plane Z: ax + by + cz + d = 0
    - The height = the max Z value - the ground plane Z
    """
    # Estimate the ground plane using RANSAC
    plane_model, inliers = tree_pcd.segment_plane(distance_threshold=0.2, 
                                                  ransac_n=3, 
                                                  num_iterations=1000)
    # Extract points in the tree cluster
    tree_points = np.asarray(tree_pcd.points)

    a, b, c, d = plane_model # Extract weights
    ground_z = -(d) / c  # Ground plane Z at x = 0, y = 0

    # Measure the maximum height in the cluster
    max_height = np.max(tree_points[:, 2]) - ground_z
    return max_height


def save_heights_to_file(tree_heights, filename="tree_heights.txt"):
    """Save Tree Heights to a File"""
    with open(filename, "w") as f:
        for i, height in enumerate(tree_heights):
            f.write(f"Tree {i+1} height: {height:.2f} meters\n")
    print(f"Tree heights saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Tree Analysis Workflow: Automatic detection of trees in a 3D point cloud and derivation of individual heights')
    parser.add_argument('--input_file_laz', type=str,
                        help='provides an input laz file')
    parser.add_argument('--input_file_pcd', type=str,
                        help='provides an input pcd file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA processing')
    parser.add_argument('--voxel_size', type=float, default=0.05, 
                        help='downsampling rate (default: 0.05)')
    parser.add_argument('--nb_neighbors', type=int, default=20, 
                        help='the number of neighbors to consider for each point for statistical outlier removal (SOR) (default: 20)')
    parser.add_argument('--std_ratio', type=float, default=2.0, 
                        help='the threshold for identifying outliers. A higher value removes fewer points (more strict), vice versa (default: 2.0)')   
    args = parser.parse_args()

    # Use GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Support files conversion
    # laz_to_pcd("input_file.laz", "input_file.pcd")

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(args.input_file_pcd)
    o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud

    # Preprocessing (downsampling, outlier removal)
    pcd_downsampled = pcd.voxel_down_sample(args.voxel_size)
    # Statistical Outlier Removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    pcd_cleaned = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_cleaned])
    
    # Transfer to GPU for speed-up calculation
    if not args.no_cuda:
        points = cp.asarray(pcd.points)  #Convert Open3D point cloud to numpy array on GPU
    
    # Perform plane segmentation (RANSAC) to remove ground
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, 
                                             ransac_n=3, 
                                             num_iterations=1000)
    # Extract the ground plane
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)  
    o3d.visualization.draw_geometries([pcd_no_ground])
    
    # Segment trees using DBSCAN
    tree_clusters = segment_trees(pcd_no_ground)


    # Apply DBSCAN (GPU-accelerated)
    db = DBSCAN(eps=0.2, min_samples=10)
    labels_gpu = db.fit_predict(points_gpu)
    # Convert the labels back to numpy and assign to clusters
    labels_cpu = cp.asnumpy(labels_gpu)
    # Visualize clusters (this can be further customized)
    pcd_clusters = []
    unique_labels = np.unique(labels_cpu)
    for label in unique_labels:
    if label != -1:  # Skip noise points
    cluster_points = points[labels_cpu == label]
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    pcd_clusters.append(cluster_pcd)
    # Visualize the clusters
    o3d.visualization.draw_geometries(pcd_clusters)


    o3d.visualization.draw_geometries(tree_clusters)
    # Measure tree height
    tree_heights = []
    for i, tree_cluster in enumerate(tree_clusters):
        height = measure_tree_height(tree_cluster)
        tree_heights.append(height)
        print(f"Tree {i+1} height: {height:.2f} meters")

    # Calculate distances on GPU
    dist_matrix_gpu = calculate_distances_gpu(points_tensor_gpu)

    # Optionally, move back to CPU if needed
    dist_matrix_cpu = dist_matrix_gpu.cpu()
    print(dist_matrix_cpu)

    # Save files
    save_heights_to_file(tree_heights)


if __name__ == '__main__':
    main()





# Perform operations on the GPU 
# Distance calculations
def calculate_distances_gpu(points_tensor):
    # Calculate the Euclidean distance between points (on GPU)
    dist_matrix = torch.cdist(points_tensor, points_tensor)  # GPU-based distance calculation
    return dist_matrix

def remove_statistical_outliers(pcd, nb_neighbors=50, std_ratio=1.0):
    """Remove statistical outliers"""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_cleaned = pcd.select_by_index(ind)
    return pcd_cleaned