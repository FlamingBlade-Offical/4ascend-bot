// lilith.cpp
#include "lilith.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cassert>

// ================= Matrix 实现 =================
Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

float& Matrix::at(int i, int j) { return data[i * cols + j]; }
float  Matrix::at(int i, int j) const { return data[i * cols + j]; }

Matrix Matrix::operator*(const Matrix& other) const {
    assert(cols == other.rows);
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < cols; ++k) {
            float a_ik = at(i, k);
            for (int j = 0; j < other.cols; ++j)
                result.at(i, j) += a_ik * other.at(k, j);
        }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix res(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            res.at(j, i) = at(i, j);
    return res;
}

void Matrix::relu() {
    for (auto& v : data)
        if (v < 0) v = 0;
}

void Matrix::add_bias(const Matrix& bias) {
    assert(bias.rows == 1 && bias.cols == cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            at(i, j) += bias.at(0, j);
}

void Matrix::relu_backward(const Matrix& forward_input) {
    // forward_input 是正向传播时 ReLU 之前的输入
    // 当前矩阵（*this）是上游传回的梯度 dY
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (forward_input.at(i, j) <= 0.0f)
                at(i, j) = 0.0f;
}

void Matrix::softmax() {
    if (rows != 1) return;
    float max_val = data[0];
    for (int i = 1; i < cols; ++i)
        if (data[i] > max_val) max_val = data[i];

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    float inv_sum = 1.0f / std::max(sum, 1e-12f);
    for (int i = 0; i < cols; ++i)
        data[i] *= inv_sum;
}

void Matrix::tanh_() {
    for (auto& v : data)
        v = std::tanh(v);
}

// ================= Loss gradients =================

// 计算策略损失关于 softmax 前原始分数的梯度 (p - π)
static Matrix policy_loss_gradient(const Matrix& p, const Matrix& pi) {
    assert(p.rows == 1 && pi.rows == 1 && p.cols == pi.cols);
    Matrix grad(1, p.cols);
    for (int j = 0; j < p.cols; ++j)
        grad.at(0, j) = p.at(0, j) - pi.at(0, j);
    return grad;
}

// ================= encode 实现（9 通道，共 648 维） =================
std::vector<float> encode(const GameState& state) {
    std::vector<float> features(648, 0.0f);
    int offset;

    // 通道0：我的棋子（当前玩家）
    offset = 0 * 81;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (state.board[i][j] == state.player_turn)
                features[offset + i * 9 + j] = 1.0f;

    // 通道1：对手棋子
    offset = 1 * 81;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (state.board[i][j] == 3 - state.player_turn)
                features[offset + i * 9 + j] = 1.0f;

    // 通道2：魔力植物（用原始数值）
    offset = 2 * 81;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (state.plants[i][j] > 0)
                features[offset + i * 9 + j] = (float)state.plants[i][j];

    // 通道3：是否 ASCEND 阶段（全局填充）
    float ascend_flag = (state.ascend_turn != 0) ? 1.0f : 0.0f;
    offset = 3 * 81;
    for (int k = offset; k < offset + 81; ++k)
        features[k] = ascend_flag;

    // 通道4：我的血量（归一化）
    float my_hp = state.hp[state.player_turn] / 6.0f;
    offset = 4 * 81;
    for (int k = offset; k < offset + 81; ++k)
        features[k] = my_hp;

    // 通道5：对手血量
    float opp_hp = state.hp[3 - state.player_turn] / 6.0f;
    offset = 5 * 81;
    for (int k = offset; k < offset + 81; ++k)
        features[k] = opp_hp;

    // 通道6：空白位置
    offset = 6 * 81;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (state.board[i][j] == 0)
                features[offset + i * 9 + j] = 1.0f;

    // 通道7：ASCEND 槽位（对方攻击棋子的位置）
    offset = 7 * 81;
    for (int k = offset; k < offset + 81; ++k)
        features[k] = 0.0f;
    int opponent = 3 - state.player_turn;
    for (auto& [cx, cy] : state.ascend_player[opponent].slots)
        features[offset + cx * 9 + cy] = 1.0f;

    // 通道8：你原文件里没写（保持全0）
    // 如果你未来想加：比如 turn_number/81 等全局特征，也可放这里

    return features;
}

// ================= Adam config =================
AdamConfig::AdamConfig(float lr_, float b1_, float b2_, float eps_, float clip_)
    : lr(lr_), beta1(b1_), beta2(b2_), eps(eps_), grad_clip(clip_) {}

// ================= Linear 实现（带 Adam） =================
Linear::Linear(int in_features, int out_features)
    : W(in_features, out_features), b(1, out_features),
      mW(in_features, out_features), vW(in_features, out_features),
      mb(1, out_features), vb(1, out_features),
      t(0)
{
    // Xavier 初始化权重
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::uniform_real_distribution<float> init_dist(-limit, limit);
    for (int i = 0; i < W.rows; ++i)
        for (int j = 0; j < W.cols; ++j)
            W.at(i, j) = init_dist(rng);

    // 偏置初始化为小正值
    for (int j = 0; j < b.cols; ++j)
        b.at(0, j) = 0.01f;
}

Matrix Linear::forward(const Matrix& input) {
    X_cached = input;
    Matrix output = input * W;
    output.add_bias(b);
    return output;
}

static inline float clipf(float x, float c) {
    return std::max(-c, std::min(c, x));
}

Matrix Linear::backward(const Matrix& dY, const AdamConfig& opt) {
    // dY: 1×out_features

    // dW: in×out
    Matrix dW(X_cached.cols, dY.cols);
    for (int i = 0; i < X_cached.cols; ++i)
        for (int j = 0; j < dY.cols; ++j)
            dW.at(i, j) = X_cached.at(0, i) * dY.at(0, j);

    Matrix db = dY;

    // clip grads
    for (auto& v : dW.data) v = clipf(v, opt.grad_clip);
    for (auto& v : db.data) v = clipf(v, opt.grad_clip);

    // dX = dY * W^T (用“旧 W”或“更新后 W”差异很小；这里用更新前 W 更标准)
    Matrix dX(1, W.rows);
    for (int i = 0; i < W.rows; ++i) {
        float s = 0.0f;
        for (int j = 0; j < W.cols; ++j)
            s += dY.at(0, j) * W.at(i, j);
        dX.at(0, i) = s;
    }

    // Adam update
    t += 1;
    const float b1 = opt.beta1;
    const float b2 = opt.beta2;
    const float one_minus_b1 = 1.0f - b1;
    const float one_minus_b2 = 1.0f - b2;

    const float b1t = 1.0f - std::pow(b1, (float)t);
    const float b2t = 1.0f - std::pow(b2, (float)t);

    // update W
    for (int i = 0; i < W.rows; ++i) {
        for (int j = 0; j < W.cols; ++j) {
            float g = dW.at(i, j);

            float& m = mW.at(i, j);
            float& v = vW.at(i, j);

            m = b1 * m + one_minus_b1 * g;
            v = b2 * v + one_minus_b2 * g * g;

            float mhat = m / b1t;
            float vhat = v / b2t;

            W.at(i, j) -= opt.lr * mhat / (std::sqrt(vhat) + opt.eps);
        }
    }

    // update b
    for (int j = 0; j < b.cols; ++j) {
        float g = db.at(0, j);

        float& m = mb.at(0, j);
        float& v = vb.at(0, j);

        m = b1 * m + one_minus_b1 * g;
        v = b2 * v + one_minus_b2 * g * g;

        float mhat = m / b1t;
        float vhat = v / b2t;

        b.at(0, j) -= opt.lr * mhat / (std::sqrt(vhat) + opt.eps);
    }

    return dX;
}

// ================= Network 实现 =================
Network::Network()
    : layer1(648, 1024),
      layer2(1024, 1024),
      policy_head(1024, 81),
      value_head(1024, 1),
      // 默认稳定配置（你也可以在训练入口改）
      policy_smooth_eps(0.02f),
      value_weight(0.5f)
{}

void Network::forward(const Matrix& input, Matrix& policy, float& value) {
    // 第一隐藏层
    Matrix h1 = layer1.forward(input);
    h1_pre = h1;
    h1.relu();
    Matrix h1_post = h1;

    // 第二隐藏层
    Matrix h2 = layer2.forward(h1_post);
    h2_pre = h2;
    h2.relu();
    Matrix h2_post = h2;

    // 策略头
    Matrix p = policy_head.forward(h2_post);
    policy_raw = p;
    policy = p;
    policy.softmax();

    // 价值头
    Matrix v = value_head.forward(h2_post);
    value_raw = v;
    Matrix v_tanh = v;
    v_tanh.tanh_();
    value = v_tanh.at(0, 0);
}

// 可选：允许外部调参
void Network::set_policy_smoothing(float eps) {
    policy_smooth_eps = std::max(0.0f, std::min(eps, 0.2f));
}
void Network::set_value_weight(float w) {
    value_weight = std::max(0.0f, std::min(w, 5.0f));
}

float Network::train(const Matrix& input,
                     const std::vector<float>& pi,
                     float z,
                     const AdamConfig& opt)
{
    // ---- 0) label smoothing ----
    // 注：最严格的做法是只在合法动作上均匀，这里用全 81 均匀是“最小改动版”
    std::vector<float> pi2 = pi;
    if (policy_smooth_eps > 0.0f) {
        float eps = policy_smooth_eps;
        float uniform = 1.0f / 81.0f;
        for (int i = 0; i < 81; ++i) {
            pi2[i] = (1.0f - eps) * pi2[i] + eps * uniform;
        }
    }

    // ---- 1) forward ----
    Matrix policy;
    float value;
    forward(input, policy, value);

    // ---- 2) policy grad: p - pi ----
    Matrix pi_mat(1, 81);
    for (int i = 0; i < 81; ++i) pi_mat.at(0, i) = pi2[i];
    Matrix d_policy_raw = policy_loss_gradient(policy, pi_mat); // (1×81)

    // ---- 3) value grad (weighted) ----
    float v = value;                 // tanh 后
    float dv = 2.0f * (v - z);
    float tanh_deriv = 1.0f - v * v;
    float d_value_raw_val = value_weight * dv * tanh_deriv;

    Matrix d_value_raw(1, 1);
    d_value_raw.at(0, 0) = d_value_raw_val;

    // ---- 4) heads backward ----
    Matrix d_h2_from_policy = policy_head.backward(d_policy_raw, opt);
    Matrix d_h2_from_value  = value_head.backward(d_value_raw, opt);

    Matrix d_h2_post(1, 1024);
    for (int j = 0; j < 1024; ++j)
        d_h2_post.at(0, j) = d_h2_from_policy.at(0, j) + d_h2_from_value.at(0, j);

    // ---- 5) back through ReLU2 ----
    Matrix d_h2_pre = d_h2_post;
    d_h2_pre.relu_backward(h2_pre);

    // ---- 6) layer2 backward ----
    Matrix d_h1_post = layer2.backward(d_h2_pre, opt);

    // ---- 7) back through ReLU1 ----
    Matrix d_h1_pre = d_h1_post;
    d_h1_pre.relu_backward(h1_pre);

    // ---- 8) layer1 backward ----
    (void)layer1.backward(d_h1_pre, opt);

    // ---- 9) compute loss (policy CE + weighted value MSE) ----
    float loss_policy = 0.0f;
    for (int i = 0; i < 81; ++i) {
        float p = std::max(1e-7f, std::min(1.0f - 1e-7f, policy.at(0, i)));
        loss_policy -= pi2[i] * std::log(p);
    }
    float loss_value = value_weight * (v - z) * (v - z);

    return loss_policy + loss_value;
}

// ================= 持久化：Matrix =================
void Matrix::save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

Matrix Matrix::load(std::ifstream& in) {
    int r, c;
    in.read(reinterpret_cast<char*>(&r), sizeof(r));
    in.read(reinterpret_cast<char*>(&c), sizeof(c));
    Matrix m(r, c);
    in.read(reinterpret_cast<char*>(m.data.data()), r * c * sizeof(float));
    return m;
}

// ================= 持久化：Linear =================
// 注意：只保存权重 W,b；Adam 的 m/v 与 t 不保存（训练重启会重置优化器状态）
void Linear::save(std::ofstream& out) const {
    W.save(out);
    b.save(out);
}

Linear Linear::load(std::ifstream& in) {
    Linear tmp; // 需要你在 lilith.h 里保留一个可用的默认构造（或改成静态工厂）
    tmp.W = Matrix::load(in);
    tmp.b = Matrix::load(in);

    // 重置 Adam 状态
    tmp.mW = Matrix(tmp.W.rows, tmp.W.cols);
    tmp.vW = Matrix(tmp.W.rows, tmp.W.cols);
    tmp.mb = Matrix(tmp.b.rows, tmp.b.cols);
    tmp.vb = Matrix(tmp.b.rows, tmp.b.cols);
    tmp.t = 0;

    return tmp;
}

// ================= 持久化：Network =================
void Network::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for saving");
    layer1.save(out);
    layer2.save(out);
    policy_head.save(out);
    value_head.save(out);

    // 额外保存两个超参（不保存也行，但保存了更一致）
    out.write(reinterpret_cast<const char*>(&policy_smooth_eps), sizeof(policy_smooth_eps));
    out.write(reinterpret_cast<const char*>(&value_weight), sizeof(value_weight));
}

Network Network::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for loading");
    Network net;
    net.layer1      = Linear::load(in);
    net.layer2      = Linear::load(in);
    net.policy_head = Linear::load(in);
    net.value_head  = Linear::load(in);

    // 兼容老模型：如果文件里没有这两个字段，读会失败；这里做一个温和处理
    // 简化起见：用 try/catch 包一下读取（如果失败就保持默认）
    try {
        in.read(reinterpret_cast<char*>(&net.policy_smooth_eps), sizeof(net.policy_smooth_eps));
        in.read(reinterpret_cast<char*>(&net.value_weight), sizeof(net.value_weight));
    } catch (...) {
        // 保持默认
    }

    return net;
}