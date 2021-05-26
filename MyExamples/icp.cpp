#include <open3d/Open3D.h>

#include <iostream>
#include <memory>
#include <thread>

using namespace open3d;
using namespace std;

void DrawTensorPointCloud(t::geometry::PointCloud p) {
    geometry::PointCloud p_ = p.ToLegacyPointCloud();

    auto cloud_ptr = std::make_shared<geometry::PointCloud>(p_);

    visualization::DrawGeometries({cloud_ptr}, "PointCloud",
                                                  1600, 900);
}

void DrawRegResultTensor(t::geometry::PointCloud source,
                         t::geometry::PointCloud target,
                         core::Tensor transform) {
    source.Transform(transform.To(core::Dtype::Float32));
    auto source_temp =  std::make_shared<geometry::PointCloud>(source.ToLegacyPointCloud());
    auto target_temp =  std::make_shared<geometry::PointCloud>(target.ToLegacyPointCloud());

    source_temp->PaintUniformColor({1, 0.706, 0});
    target_temp->PaintUniformColor({0, 0.651, 0.929});

    visualization::DrawGeometries({source_temp, target_temp});
}

void draw_registration_result(std::shared_ptr<geometry::PointCloud> source,
                              std::shared_ptr<geometry::PointCloud> target,
                              Eigen::Matrix4d transformation) {
    auto source_temp = std::make_shared<geometry::PointCloud>(*source);
    auto target_temp = std::make_shared<geometry::PointCloud>(*target);
    source_temp->PaintUniformColor({1, 0.706, 0});
    target_temp->PaintUniformColor({0, 0.651, 0.929});
    source_temp->Transform(transformation);
    visualization::DrawGeometries({source_temp, target_temp});
}

int main() {
    auto source = std::make_shared<geometry::PointCloud>();
    auto target = std::make_shared<geometry::PointCloud>();

    io::ReadPointCloud("../cloud_bin_0.pcd", *source);
    io::ReadPointCloud("../cloud_bin_1.pcd", *target);

    auto source_t = t::geometry::PointCloud::FromLegacyPointCloud(*source);
    auto target_t = t::geometry::PointCloud::FromLegacyPointCloud(*target);

    double thresh = 0.02;

    core::Device gpu = core::Device("CPU:0");
    core::Dtype dtype = core::Dtype::Float64;

    std::vector<double> init_trans_vec{
            0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
            0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

    core::Tensor init_trans(init_trans_vec, {4, 4}, dtype, gpu);

    // DrawRegResultTensor(source_t, target_t, init_trans);

    auto res = t::pipelines::registration::RegistrationICP(
            source_t, target_t, thresh, init_trans,
            t::pipelines::registration::TransformationEstimationPointToPoint());

    // auto
    // res=pipelines::registration::EvaluateRegistration(*source,*target,thresh,init_trans);
    cout << res.inlier_rmse_<< "\n";

    DrawRegResultTensor(source_t,target_t,res.transformation_);

    // visualization::DrawGeometries({source, target});

    return 0;
}