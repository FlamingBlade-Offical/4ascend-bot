// lilith.h
#ifndef LILITH_H
#define LILITH_H

#include <vector>
#include <cassert>
#include "game_state.h"

class Matrix {
public:
    int rows, cols;
    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c);
    float& at(int i, int j);
    float  at(int i, int j) const;

    // 激活函数（之后会加）
    void relu();
    void add_bias(const Matrix& bias);
    void softmax();  // 把矩阵的某一行（1×cols）转为概率分布
    void tanh_();    // 逐元素 tanh
    void relu_backward(const Matrix& forward_input);
    void save(std::ofstream& out) const;
    static Matrix load(std::ifstream& in);

    // 矩阵乘法
    Matrix operator*(const Matrix& other) const;
    Matrix transpose() const;
};
class Linear {
public:
    Matrix W;  // 权重 (fan_in × fan_out)
    Matrix b;  // 偏置 (1 × fan_out)
    Matrix X_cached;
    Linear(int in_features, int out_features);
    Linear() = default;   // 需要默认构造函数（用于加载）
    void save(std::ofstream& out) const;
    static Linear load(std::ifstream& in);
    // 前向传播：输入 → 输出
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& dY,float lr);
    Matrix policy_loss_gradient(const Matrix& p, const Matrix& pi);
};
class Network {
public:
    Linear layer1, layer2;
    Linear policy_head, value_head;

    Network();
    void save(const std::string& filename) const;
    static Network load(const std::string& filename);
    // 前向传播：输入特征矩阵 (1×648)，返回 (policy矩阵, value原始值)
    void forward(const Matrix& input, Matrix& policy, float& value);
    float train(const Matrix& input, const std::vector<float>& pi, float z, float lr);
private:
    // 缓存正向传播的中间值（仅用于训练）
    Matrix h1_pre, h2_pre;
    Matrix policy_raw, value_raw;
};
// 编码函数声明
std::vector<float> encode(const GameState& state);  // 需要提前包含 game_state.h

#endif