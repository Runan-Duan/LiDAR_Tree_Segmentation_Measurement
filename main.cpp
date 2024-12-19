#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing.h>
using namespace std;

/*
This workflow takes a PCD point cloud (converted from LAZ), segments the trees, removes the ground, 
and measures the height of each tree using the Point Cloud Library (PCL). 

The process involves:
1. Preprocessing
- Loading the point cloud.
- Downsampling
- Removing the ground plane using RANSAC
- Progressive morphological filtering (PMF)

2. Segmenting trees
- Region-growing: segment overlapping trees

3. Measuring tree height
- Calculating a localized ground plane for each tree cluster using KD-Tree
- Using RANSAC to fit the local ground plane under each tree
- Computing height relative to local ground plane

4. Visual Inspection and Validation
- Highlighting detected tree tops and bases, and labeling heights for validation
- Export clusters to files 

5. Smoothing and Cleanup
- Using MLS (Moving Least Squares) to smooth the tree cluster surfaces, especially for crowns
*/

pcl::PointCloud<pcl::PointXYZ>::Ptr remove_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_trees(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
float measure_tree_height(pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cluster);

int main()
{
    std::cout << "Start running, using PCL version: " << PCL_VERSION << std::endl;

    // Assign variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Load point cloud
    std::string infile = "input_small.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(infile, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s\n", infile.c_str());
        return -1;
    }
    else {
        std::cout << "File " << infile << " loaded successfully!" << std::endl;
        std::cerr << "PointCloud before downsampling: " << cloud->width * cloud->height << " data points (" <<
        pcl::getFieldsList(*cloud) << ")." << std::endl;
    }

    // Preprocessing
    // 1. Downsampling
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);  // Leaf size 10cm
    voxel_filter.filter(*cloud_filtered);
    std::cerr << "PointCloud after downsampling: " << cloud_filtered->width * cloud_filtered->height
              << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

    // 2. Remove ground using RANSAC
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground = remove_ground(cloud_filtered);

    std::cout << "Cloud after ground removal contains " << cloud_no_ground->points.size() << " points." << std::endl;

    // 3. Noise removal (Statistical Outlier Removal)
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_no_ground);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_no_ground);

    // 4. Segment trees
    std::cout << "Segmenting trees..." << std::endl;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> trees = segment_trees(cloud_no_ground);
    std::cout << "Number of tree clusters: " << trees.size() << std::endl;

    // 5. Measure height for each tree and save to file
    std::ofstream outfile("tree_heights.txt");
    if (outfile.is_open()) {
        for (size_t i = 0; i < trees.size(); ++i)
        {
            float tree_height = measure_tree_height(trees[i]);
            outfile << "Tree " << i + 1 << " height: " << tree_height << " meters" << std::endl;
        }
        outfile.close();
        std::cout << "Tree heights saved to tree_heights.txt" << std::endl;
    }
    else {
        std::cerr << "Unable to open output file." << std::endl;
    }

    return 0;
}

// Remove ground points using RANSAC
pcl::PointCloud<pcl::PointXYZ>::Ptr remove_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*cloud_no_ground);

    return cloud_no_ground;
}

// Segment trees using region growing
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_trees(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setSearchMethod(tree);
    reg.setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    reg.extract(cluster_indices);

    // Extract each tree cluster
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> trees;
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int index : indices.indices) {
            tree_cluster->points.push_back(cloud->points[index]);
        }
        trees.push_back(tree_cluster);
    }
    return trees;
}

// Measure tree height based on local ground plane
float measure_tree_height(pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cluster)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setInputCloud(tree_cluster);
    seg.segment(*inliers, *coefficients);

    // Calculate height by finding the highest point in the tree cluster
    float max_height = -std::numeric_limits<float>::infinity();
    for (const auto& point : tree_cluster->points) {
        max_height = std::max(max_height, point.z);
    }

    return max_height - coefficients->values[3]; // Local ground plane height adjustment
}
