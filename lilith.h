// lilith.h

#ifndef LILITH_H
#define LILITH_H

#include <vector>
#include <cassert>
#include <fstream>
#include <string>

#include "game_state.h"

// ======================================================
// Matrix
// ======================================================

class Matrix {
public:

    int rows, cols;

    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c);

    float& at(int i, int j);

    float at(int i, int j) const;

    // activation
    void relu();

    void tanh_();

    void relu_backward(const Matrix& forward_input);

    void softmax();

    void add_bias(const Matrix& bias);

    // io
    void save(std::ofstream& out) const;

    static Matrix load(std::ifstream& in);

    // math
    Matrix operator*(const Matrix& other) const;

    Matrix transpose() const;
};

// ======================================================
// Adam Optimizer Config
// ======================================================

struct AdamConfig {

    float lr;

    float beta1;

    float beta2;

    float eps;

    float grad_clip;

    AdamConfig(
        float lr_ = 1e-3f,
        float b1_ = 0.9f,
        float b2_ = 0.999f,
        float eps_ = 1e-8f,
        float clip_ = 1.0f
    );
};

// ======================================================
// Linear Layer
// ======================================================

class Linear {
public:

    // parameters
    Matrix W;

    Matrix b;

    // cache
    Matrix X_cached;

    // Adam states
    Matrix mW, vW;

    Matrix mb, vb;

    int t;

    Linear(int in_features, int out_features);

    Linear() = default;

    Matrix forward(const Matrix& input);

    Matrix backward(
        const Matrix& dY,
        const AdamConfig& opt
    );

    void save(std::ofstream& out) const;

    static Linear load(std::ifstream& in);
};

// ======================================================
// Network
// ======================================================

class Network {
public:

    Linear layer1;
    Linear layer2;

    Linear policy_head;
    Linear value_head;

    Network();

    void forward(
        const Matrix& input,
        Matrix& policy,
        float& value
    );

    float train(
        const Matrix& input,
        const std::vector<float>& pi,
        float z,
        const AdamConfig& opt
    );

    void save(const std::string& filename) const;

    static Network load(const std::string& filename);

    // training stability
    void set_policy_smoothing(float eps);

    void set_value_weight(float w);

private:

    // forward cache
    Matrix h1_pre;
    Matrix h2_pre;

    Matrix policy_raw;

    Matrix value_raw;

    // stabilization
    float policy_smooth_eps;

    float value_weight;
};

// ======================================================
// Encode
// ======================================================

std::vector<float> encode(const GameState& state);

#endif