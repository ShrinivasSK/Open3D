#include <open3d/Open3D.h>

#include <iostream>
#include <memory>
#include <thread>

int main() {
    auto cloud_ptr = std::make_shared<geometry::PointCloud>();
    auto down_pcd = std::make_shared<geometry::PointCloud>();
    if (io::ReadPointCloud("../fragment.ply", *cloud_ptr)) {
        utility::LogInfo("Successfully read {}\n", argv[2]);
    } else {
        utility::LogError("Failed to read {}\n\n", argv[2]);
        return 1;
    }
    cloud_ptr->NormalizeNormals();
    // down_pcd=cloud_ptr->VoxelDownSample(0.05);
    visualization::DrawGeometries({down_pcd}, "PointCloud",
                                                  1600, 900);
    return 0;
}