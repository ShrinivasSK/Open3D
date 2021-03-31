// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/t/io/TriangleMeshIO.h"

#include "open3d/io/TriangleMeshIO.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshIO, CreateMeshFromFile) {
    auto mesh = t::io::CreateMeshFromFile(TEST_DATA_DIR "/knot.ply");
    EXPECT_EQ(mesh->GetTriangles().GetLength(), 2880);
    EXPECT_EQ(mesh->GetVertices().GetLength(), 1440);
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshPLY) {
    t::geometry::TriangleMesh mesh, mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(TEST_DATA_DIR "/knot.ply", mesh));
    std::string file_name = std::string(TEST_DATA_DIR) + "/test_mesh.ply";
    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name, mesh));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name, mesh_read));
    EXPECT_TRUE(mesh.GetTriangles().AllClose(mesh_read.GetTriangles()));
    EXPECT_TRUE(mesh.GetVertices().AllClose(mesh_read.GetVertices()));
    EXPECT_EQ(std::remove(file_name.c_str()), 0);
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshOBJ) {
    t::geometry::TriangleMesh mesh, mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(
            TEST_DATA_DIR "/open3d_downloads/tests/cube.obj", mesh));
    std::string file_name = std::string(TEST_DATA_DIR) + "/test_mesh.obj";
    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name, mesh));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name, mesh_read));
    EXPECT_TRUE(mesh.GetTriangles().AllClose(mesh_read.GetTriangles()));
    EXPECT_TRUE(mesh.GetVertices().AllClose(mesh_read.GetVertices()));
    EXPECT_EQ(std::remove(file_name.c_str()), 0);
}

// TODO: Add tests for triangle_uvs, materials, triangle_material_ids and
// textures once these are supported.
TEST(TriangleMeshIO, TriangleMeshLegecyCompatibility) {
    t::geometry::TriangleMesh mesh_tensor, mesh_tensor_read;
    geometry::TriangleMesh mesh_legacy, mesh_legacy_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(TEST_DATA_DIR "/monkey/monkey.obj",
                                        mesh_tensor));
    EXPECT_TRUE(io::ReadTriangleMesh(TEST_DATA_DIR "/monkey/monkey.obj",
                                     mesh_legacy));

    EXPECT_EQ(mesh_tensor.GetTriangles().GetLength(),
              mesh_legacy.triangles_.size());
    EXPECT_EQ(mesh_tensor.GetVertices().GetLength(),
              mesh_legacy.vertices_.size());
    EXPECT_EQ(mesh_tensor.GetVertexNormals().GetLength(),
              mesh_legacy.vertex_normals_.size());

    std::string file_name_tensor =
            std::string(TEST_DATA_DIR) + "/test_mesh_tensor.obj";
    std::string file_name_legacy =
            std::string(TEST_DATA_DIR) + "/test_mesh_legacy.obj";

    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name_tensor, mesh_tensor));
    EXPECT_TRUE(io::WriteTriangleMesh(file_name_legacy, mesh_legacy));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name_tensor, mesh_tensor_read));
    EXPECT_TRUE(io::ReadTriangleMesh(file_name_legacy, mesh_legacy_read));

    EXPECT_EQ(mesh_tensor_read.GetTriangles().GetLength(),
              mesh_legacy_read.triangles_.size());
    EXPECT_EQ(mesh_tensor_read.GetVertices().GetLength(),
              mesh_legacy_read.vertices_.size());
    EXPECT_EQ(mesh_tensor_read.GetVertexNormals().GetLength(),
              mesh_legacy_read.vertex_normals_.size());
    EXPECT_EQ(std::remove(file_name_tensor.c_str()), 0);
    EXPECT_EQ(std::remove(file_name_legacy.c_str()), 0);
}

}  // namespace tests
}  // namespace open3d
