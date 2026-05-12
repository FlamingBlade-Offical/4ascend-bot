// lilith.cpp
#include "lilith.h"
#include <cmath>
#include <cstdlib>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

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
    // forward_input 是正向传播时ReLU之前的输入
    // 当前矩阵（*this）是上游传回的梯度 dY
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (forward_input.at(i, j) <= 0.0f)
                at(i, j) = 0.0f;
}

Matrix Matrix::transpose() const {
    Matrix res(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            res.at(j, i) = at(i, j);
    return res;
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
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; ++i)
        data[i] *= inv_sum;
}

void Matrix::tanh_() {
    for (auto& v : data)
        v = std::tanh(v);
}

// 计算策略损失关于 softmax 前原始分数的梯度 (p - π)
Matrix policy_loss_gradient(const Matrix& p, const Matrix& pi) {
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

    // 通道0：我的棋子
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

    // 通道2：魔力植物
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
        features[k] = 0.0f;  // 默认0
    if (state.ascend_turn != 0) {
        int attacker = state.ascend_turn;
        for (auto& [cx, cy] : state.ascend_player[attacker].slots)
            features[offset + cx * 9 + cy] = 1.0f;
    }
    return features;
}

// ================= Linear 实现 =================
Linear::Linear(int in_features, int out_features)
    : W(in_features, out_features), b(1, out_features)
{
    // Xavier 初始化权重
    float limit = std::sqrt(6.0f / (in_features + out_features));
    // 简单的随机数生成（之后你可以换成 std::mt19937）
    for (int i = 0; i < W.rows; ++i)
        for (int j = 0; j < W.cols; ++j){
            std::uniform_real_distribution<float> init_dist(-limit, limit);
            W.at(i, j) = init_dist(rng);
        }

    // 偏置初始化为小的正值（或零）
    for (int j = 0; j < b.cols; ++j)
        b.at(0, j) = 0.01f;
}

Matrix Linear::forward(const Matrix& input) {
    X_cached = input;   // 保存输入，供反向使用
    // 1. 矩阵乘法
    Matrix output = input * W;
    // 2. 加偏置
    output.add_bias(b);
    return output;
}

// 反向传播
Matrix Linear::backward(const Matrix& dY, float lr) {
    // dY: 1×out_features
    // dW: in_features×out_features
    Matrix dW(X_cached.cols, dY.cols);
    for (int i = 0; i < X_cached.cols; ++i)      // 遍历输入特征
        for (int j = 0; j < dY.cols; ++j)        // 遍历输出特征
            dW.at(i, j) = X_cached.at(0, i) * dY.at(0, j);
    for (auto& v : dW.data) v = std::max(-1.0f, std::min(1.0f, v));
    // db 直接拷贝 dY (1×out_features)
    Matrix db = dY;

    // 更新参数（梯度下降）
    for (int i = 0; i < W.rows; ++i)
        for (int j = 0; j < W.cols; ++j)
            W.at(i, j) -= lr * dW.at(i, j);
    for (int j = 0; j < b.cols; ++j)
        b.at(0, j) -= lr * db.at(0, j);

    // 计算并返回 dX = dY * W^T
    Matrix dX(1, W.rows);
    for (int i = 0; i < W.rows; ++i) {
        float s = 0.0f;
        for (int j = 0; j < W.cols; ++j)
            s += dY.at(0, j) * W.at(i, j);
        dX.at(0, i) = s;
    }
    return dX;
}

Network::Network() : layer1(648, 768), layer2(768, 768),
                     policy_head(768, 81), value_head(768, 1) {}

void Network::forward(const Matrix& input, Matrix& policy, float& value) {
    // 第一隐藏层
    Matrix h1 = layer1.forward(input);
    h1_pre = h1;                  // 缓存 ReLU 之前的输出
    h1.relu();
    Matrix h1_post = h1;          // ReLU 之后的输出（会自动被 layer2 的 X_cached 保存）

    // 第二隐藏层
    Matrix h2 = layer2.forward(h1_post);
    h2_pre = h2;                  // 缓存 ReLU 之前的输出
    h2.relu();
    Matrix h2_post = h2;          // ReLU 之后

    // 策略头
    Matrix p = policy_head.forward(h2_post);
    policy_raw = p;               // 缓存 softmax 之前的分数
    policy = p;
    policy.softmax();

    // 价值头
    Matrix v = value_head.forward(h2_post);
    value_raw = v;                // 缓存 tanh 之前的分数
    Matrix v_tanh = v;
    v_tanh.tanh_();
    value = v_tanh.at(0, 0);
}
float Network::train(const Matrix& input, const std::vector<float>& pi, float z, float lr) {
    // ---- 1. 前向传播并缓存所有中间值 ----
    Matrix policy;
    float value;
    forward(input, policy, value);  // 这会把 h1_pre, h2_pre, policy_raw, value_raw 都缓存好

    // ---- 2. 策略损失反向：dL/d(policy_raw) = p - π ----
    Matrix pi_mat(1, 81);
    for (int i = 0; i < 81; ++i) pi_mat.at(0, i) = pi[i];
    Matrix d_policy_raw = policy_loss_gradient(policy, pi_mat);  // (1×81)

    // ---- 3. 价值损失反向：dL/d(value_raw) ----
    float v = value;                             // 已经是 tanh 后的值
    float dL_dv = 2.0f * (v - z);                // 对 value 的梯度
    float tanh_deriv = 1.0f - v * v;             // tanh 的导数
    float d_value_raw_val = dL_dv * tanh_deriv;  // 对 value_raw 的梯度
    Matrix d_value_raw(1, 1);
    d_value_raw.at(0, 0) = d_value_raw_val;

    // ---- 4. 两个输出头分别反向传播 ----
    // policy_head 反向，得到对 h2_post 的梯度
    Matrix d_h2_from_policy = policy_head.backward(d_policy_raw, lr);  // (1×768)
    // value_head 反向，得到对 h2_post 的梯度
    Matrix d_h2_from_value = value_head.backward(d_value_raw, lr);     // (1×768)

    // 合并两个梯度形成对 h2_post 的总梯度
    Matrix d_h2_post(1, 768);
    for (int j = 0; j < 768; ++j)
        d_h2_post.at(0, j) = d_h2_from_policy.at(0, j) + d_h2_from_value.at(0, j);

    // ---- 5. 反向通过第二个 ReLU ----
    Matrix d_h2_pre = d_h2_post;          // 上游梯度复制
    d_h2_pre.relu_backward(h2_pre);       // 依据 h2_pre 的正负屏蔽梯度

    // ---- 6. 第二隐藏层的线性反向 ----
    Matrix d_h1_post = layer2.backward(d_h2_pre, lr);  // (1×768)

    // ---- 7. 反向通过第一个 ReLU ----
    Matrix d_h1_pre = d_h1_post;
    d_h1_pre.relu_backward(h1_pre);       // 依据 h1_pre 的正负屏蔽梯度

    // ---- 8. 第一隐藏层的线性反向 ----
    Matrix d_input = layer1.backward(d_h1_pre, lr);  // (1×648) 此值可以忽略

    // ---- 9. 计算并返回总损失（用于监控） ----
    float loss_policy = 0.0f;
    for (int i = 0; i < 81; ++i) {
        //if (pi[i] > 1e-12f)   // 避免 log(0)
        //    loss_policy -= pi[i] * std::log(policy.at(0, i) + 1e-12f);
        float p = std::max(1e-7f, std::min(1.0f - 1e-7f, policy.at(0, i)));
        loss_policy -= pi[i] * log(p);
    }
    float loss_value = (v - z) * (v - z);
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
void Linear::save(std::ofstream& out) const {
    W.save(out);
    b.save(out);
}

Linear Linear::load(std::ifstream& in) {
    Linear tmp;        // 默认构造（空）
    tmp.W = Matrix::load(in);
    tmp.b = Matrix::load(in);
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
}

Network Network::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for loading");
    Network net;
    net.layer1      = Linear::load(in);
    net.layer2      = Linear::load(in);
    net.policy_head = Linear::load(in);
    net.value_head  = Linear::load(in);
    return net;
}