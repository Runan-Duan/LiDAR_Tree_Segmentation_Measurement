#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

/*
This workflow takes a PCD point cloud (converted from LAZ), segments the trees, removes the ground, 
and measures the height of each tree using the Point Cloud Library (PCL). 

The process involves:
- Loading the point cloud.
- Removing the ground plane using RANSAC.
- Segmenting individual trees using Euclidean Cluster Extraction.
- Measuring the height of each segmented tree.
You can adapt these code to your specific use case, depending on the characteristics of your data 
(e.g., point density, tree size).
*/

pcl::PointCloud<pcl::PointXYZ>::Ptr remove_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Create the segmentation object for ground segmentation
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);  // Set ground detection threshold

    // Objects to hold the segmentation result
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Perform segmentation
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    // Extract ground points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);  // Extract non-ground points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*cloud_no_ground);

    return cloud_no_ground;
}


std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_trees(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> tree_clusters;

    // Create a KDTree for search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // Extract clusters using Euclidean Cluster Extraction
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);  // Set the distance threshold
    ec.setMinClusterSize(100);    // Set the minimum number of points in a cluster
    ec.setMaxClusterSize(10000);  // Set the maximum number of points in a cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    // Store the individual tree clusters
    for (const auto& indices : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tree(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& index : indices.indices)
            tree->points.push_back(cloud->points[index]);
        tree->width = tree->points.size();
        tree->height = 1;
        tree->is_dense = true;

        tree_clusters.push_back(tree);
    }

    return tree_clusters;
}

float measure_tree_height(pcl::PointCloud<pcl::PointXYZ>::Ptr tree)
{
    float min_z = FLT_MAX;
    float max_z = -FLT_MAX;

    for (const auto& point : tree->points)
    {
        if (point.z < min_z)
            min_z = point.z;
        if (point.z > max_z)
            max_z = point.z;
    }

    return max_z - min_z;  // Height of the tree
}

int main()
{
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("output.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file output.pcd \n");
        return -1;
    }

    // Remove ground points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground = remove_ground(cloud);

    // Segment trees
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> trees = segment_trees(cloud_no_ground);

    // Measure height for each tree
    for (size_t i = 0; i < trees.size(); ++i)
    {
        float tree_height = measure_tree_height(trees[i]);
        std::cout << "Tree " << i+1 << " height: " << tree_height << " meters" << std::endl;
    }

    return 0;
}
