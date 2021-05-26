#include <open3d/Open3D.h>

#include <iostream>
#include <memory>
#include <thread>

using namespace open3d;
using namespace std;

core::Tensor Initialise(bool print = false) {
    core::Device gpu = core::Device("CUDA:0");
    core::Device cpu = core::Device("CPU:0");
    core::Dtype dtype = core::Dtype::Float32;

    // Initialising
    // 1. With Init Value
    auto t1 = core::Tensor::Init<int>({1, 2, 3, 4}, gpu);

    // Of a given shape. Initial value random
    if (print) {
        auto t2 = core::Tensor({1, 2, 3}, dtype, gpu);

        // 2. With Vector
        std::vector<float> t_vec{0.862,  0.011, -0.507, 0.5,   -0.139, 0.967,
                                 -0.215, 0.7,   0.487,  0.255, 0.835,  -1.4,
                                 0.0,    0.0,   0.0,    1.0};

        core::Tensor t3(t_vec, {4, 4}, dtype, gpu);

        cout << "T1: \n" << t1.ToString() << "\n";
        cout << "T2: \n" << t2.ToString() << "\n";
        cout << "T3: \n" << t3.ToString() << "\n";
    }

    return t1;
}

void PrintProperties(core::Tensor t) {
    // Properties
    cout << "T: \n" << t.ToString() << "\n";
    cout << "Shape: " << t.GetShape().ToString() << "\n";
    cout << "Device: " << t.GetDevice().ToString() << "\n";
    cout << "Dtype: " << t.GetDtype().ToString() << "\n";
}

void DeviceTransfer(core::Tensor t) {
    core::Device cpu = core::Device("CPU:0");
    // Device Transfer
    auto t_cpu = t.To(cpu);
    cout << "T in CPU\n";
    cout << t_cpu.ToString() << "\n";
}

void TypeCasting(core::Tensor t) {
    // Type Casting
    auto t_int = t.To(core::Dtype::Int32);
    cout << "T typecasted\n";
    cout << t_int.ToString() << "\n";
}

void BinaryOperations(){
    core::Device gpu = core::Device("CUDA:0");
    core::Dtype dtype = core::Dtype::Int32;

    core::Tensor a = core::Tensor::Init<int>({1, 2, 3, 4}, gpu);
    core::Tensor b = core::Tensor::Init<int>({-1, 1, -1, 1}, gpu);

    cout<<"A: \n"<<a.ToString()<<"\n";
    cout<<"B: \n"<<b.ToString()<<"\n";

    // Binary Operations
    cout<<"Binary Operations: +,-,/,*\n";
    cout<<(a+b).ToString()<<"\n";
    cout<<(a-b).ToString()<<"\n";
    cout<<(a/b).ToString()<<"\n";
    cout<<(a*b).ToString()<<"\n";

    // Broadcasting
    auto t1 = core::Tensor({2, 3}, dtype, gpu);
    auto t2 = core::Tensor({2,1}, dtype, gpu);

    cout<<"A: \n"<<t1.ToString()<<"\n";
    cout<<"B: \n"<<t2.ToString()<<"\n";

    cout<<"BroadCasting: \n";
    cout << (t1 + t2).ToString() << "\n";
}

void UnaryOperations(){
    core::Device gpu = core::Device("CUDA:0");

    core::Tensor a = core::Tensor::Init<float>({4, 9, 16}, gpu);

    cout<<"A: \n"<<a.ToString()<<"\n";

    // Unary Operations
    cout<<"Unary Operations: cos, sin, sqrt\n";
    cout<<(a.Cos()).ToString()<<"\n";
    cout<<(a.Sin()).ToString()<<"\n";
    cout<<(a.Sqrt()).ToString()<<"\n";
}

void LogicalOperations(){
    core::Device gpu = core::Device("CUDA:0");

    core::Tensor a = core::Tensor::Init<bool>({true, false, true}, gpu);
    core::Tensor b = core::Tensor::Init<bool>({true, false, true}, gpu);

    cout<<"A: \n"<<a.ToString()<<"\n";
    cout<<"B: \n"<<b.ToString()<<"\n";

    // Logical Operations
    cout<<"Logical Operations: and, or, xor, not\n";
    cout<<(a.LogicalAnd(b)).ToString()<<"\n";
    cout<<(a.LogicalOr(b)).ToString()<<"\n";
    cout<<(a.LogicalXor(b)).ToString()<<"\n";
    cout<<(a.LogicalNot()).ToString()<<"\n";
}

void ComparisionOperations(){
    core::Device gpu = core::Device("CUDA:0");
    core::Dtype dtype = core::Dtype::Int32;

    core::Tensor a = core::Tensor::Init<int>({0, 1, -1}, gpu);
    core::Tensor b = core::Tensor::Init<int>({0,0,0,}, gpu);

    cout<<"A: \n"<<a.ToString()<<"\n";
    cout<<"B: \n"<<b.ToString()<<"\n";

    // Comparision Operations
    cout<<"Comparision Operations: >,<,==,!=\n";
    cout<<(a>b).ToString()<<"\n";
    cout<<(a<b).ToString()<<"\n";
    cout<<(a==b).ToString()<<"\n";
    cout<<(a!=b).ToString()<<"\n";
    cout<<"Others like <=,>= also exist\n";
}

void Reduction(){
    core::Device gpu = core::Device("CUDA:0");

    core::Tensor a = core::Tensor::Init<float>({4, 9, 16}, gpu);

    cout<<"A: \n"<<a.ToString()<<"\n";

    // Reduction
    cout<<"Reduction: sum, mean, prod, max, argmax\n";
    cout<<(a.Sum({0})).ToString()<<"\n";
    cout<<(a.Mean({0})).ToString()<<"\n";
    cout<<(a.Prod({0})).ToString()<<"\n";
    cout<<(a.Max({0})).ToString()<<"\n";
    cout<<(a.ArgMax({0})).ToString()<<"\n";
}

void Indexing(){
    core::Device gpu = core::Device("CUDA:0");
    core::Dtype dtype = core::Dtype::Float32;


    core::Tensor a = core::Tensor({2, 6, 4}, dtype,gpu);

    cout<<"A: \n"<<a.ToString()<<"\n";

    // Simple Indexing
    cout<<"Indexing: a[1] \n";
    cout<<(a[1]).ToString()<<"\n";

    // Slicing
    // Dimension:Start:Stop:Step
    cout<<"Slicing: a[0:2:1]\n";
    cout<<(a.Slice(0,0,1,2)).ToString()<<"\n";
    cout<<"Slicing: a[0:2:1][:,0:3:2,:]\n";
    cout<<(a.Slice(0,0,1,2).Slice(1,0,3,2)).ToString()<<"\n";
}

int main() {
    // auto t = Initialise();

    // PrintProperties(t);

    // DeviceTransfer(t);

    // TypeCasting(t)

    // Shared Memory: Not Shared by default
    // Probably no such option

    // BinaryOperations();

    // UnaryOperations();

    // LogicalOperations();

    // ComparisionOperations();

    // Reduction();

    // Indexing();

    cout<<"Uncomment something to run!\n";

    return 0;
}