#include "game_state.h"
#include "lilith.h"
#include <vector>
#include <cmath>
#include <algorithm>

// MCTS 节点结构体（在头文件或 mcts.cpp 中定义）
struct MCTSNode {
    GameState state;
    int player;                     // 当前局面轮到谁走 (1 白, 2 黑)
    std::pair<int,int> move;        // 从父节点走到此节点的落子坐标
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int N;                          // 访问次数
    float W;                        // 累积价值
    float P;                        // 先验概率（由神经网络策略头给出）
    bool expanded;

    MCTSNode(const GameState& s, int p, std::pair<int,int> m = std :: make_pair(-1,-1), MCTSNode* par = nullptr)
        : state(s), player(p), move(m), parent(par), N(0), W(0.0f), P(0.0f), expanded(false) {}

    ~MCTSNode() {
        for (auto* child : children) delete child;
    }
};

// 选择：从 root 开始，根据 UCB 公式递归选择子节点，直到到达未完全扩展的节点
MCTSNode* select(MCTSNode* root, float c_puct) {
    MCTSNode* node = root;
    while (node->expanded) {
        if (node->children.empty()) break;  // 终局或无法继续
        float best_ucb = -1e9;
        MCTSNode* best_child = nullptr;
        for (MCTSNode* child : node->children) {
            float Q = (child->N > 0) ? -(child->W / child->N) : 0.0f;
            float ucb = Q + c_puct * child->P * std::sqrt(node->N + 1.0f) / (1.0f + child->N);
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = child;
            }
        }
        if (best_child == nullptr) break;
        node = best_child;
    }
    return node;
}

// 回传：将叶节点价值 v 沿路径传回，每次向上层翻转符号（视角切换）
void backup(MCTSNode* leaf, float v) {
    MCTSNode* node = leaf;
    while (node != nullptr) {
        node->N++;
        v = std::max(-1.0f, std::min(1.0f, v));
        node->W += v;
        v = -v;              // 切换到对手视角
        node = node->parent;
    }
}

// 扩展：对叶节点用网络评估，创建所有合法子节点，并将先验概率填入子节点，然后立即回传网络价值
void expand(MCTSNode* leaf, Network& net) {

    // 获取合法走法
    auto moves = leaf->state.get_legal_moves();
    if (moves.empty() || leaf->state.game_end_check().first) {
        leaf->expanded = true;
        int winner = leaf->state.game_end_check().second;
        float final_v = (winner == leaf->player) ? 1.0f : -1.0f;
        backup(leaf, final_v);
        return;
    }

    // 编码当前局面
    std::vector<float> features = encode(leaf->state);
    Matrix input(1, 648);
    for (int i = 0; i < 648; ++i) input.at(0, i) = features[i];

    // 网络前向，获取策略和胜率
    Matrix policy;
    float value;
    net.forward(input, policy, value);

    // 扩展节点
    leaf->expanded = true;
    for (auto& m : moves) {
        GameState next = leaf->state.clone();
        if (!next.apply_move(m.first, m.second)) // ★ 必须检查
            continue;
        int next_player = next.player_turn;
        MCTSNode* child = new MCTSNode(next, next_player, m, leaf);
        int idx = m.first * 9 + m.second;
        child->P = policy.at(0, idx);
        leaf->children.push_back(child);
    }

    float sum_P = 0;
    for (auto* ch : leaf->children) sum_P += ch->P;
    if (sum_P > 0)
        for (auto* ch : leaf->children) ch->P /= sum_P;

    // 回传网络评估的价值
    backup(leaf, value);
}

// MCTS 主搜索：返回一个 81 维的概率分布 π，表示当前局面下每个落子位置的搜索概率
std::vector<float> mcts_search(const GameState& root_state, int root_player, Network& net,
                               int sims = 1600, float c_puct = 2.0f,bool add_noise = true) {
    MCTSNode* root = new MCTSNode(root_state, root_player);

     // ========== 新增：扩展根节点并添加探索噪声 ==========
    if (!root->expanded) {
        expand(root, net);
    }
    if (add_noise && !root->children.empty()) {
        float epsilon = 0.5f;          // 噪声混合比例
        float alpha   = 0.03f;          // 值越小，噪声越集中
        std::vector<float> noise(root->children.size());
        std::gamma_distribution<float> gamma(alpha, 1.0f);
        float noise_sum = 0.0f;
        for (auto& n : noise) {
            n = gamma(rng);             // rng 是全局梅森旋转引擎
            noise_sum += n;
        }
        for (size_t i = 0; i < root->children.size(); ++i) {
            float& original_P = root->children[i]->P;
            original_P = (1.0f - epsilon) * original_P + epsilon * (noise[i] / noise_sum);
        }
    }
    // ==========================================

    for (int i = 0; i < sims; ++i) {
        MCTSNode* leaf = select(root, c_puct);

        // 如果到达终局，直接用真实胜负价值回传
        if (leaf->state.game_end_check().first) {
            int winner = leaf->state.game_end_check().second;
            float final_v = (winner == leaf->player) ? 1.0f : -1.0f;
            backup(leaf, final_v);
            continue;
        }

        // 正常扩展（回传网络价值）
        expand(leaf, net);
    }


    // 根据根节点各子节点的访问次数构建 π
    std::vector<float> pi(81, 0.0f);
    float total_N = 0.0f;
    for (MCTSNode* child : root->children) {
        int idx = child->move.first * 9 + child->move.second;
        pi[idx] = static_cast<float>(child->N);
        total_N += child->N;
    }
    if (total_N > 0) {
        for (float& p : pi) p /= total_N;
    } else {
        // 如果没有子节点或都没有被访问（比如开局无合法走法），返回均匀分布
        float uniform = 1.0f / 81.0f;
        for (float& p : pi) p = uniform;
    }

    delete root;  // 递归删除所有节点
    return pi;
}