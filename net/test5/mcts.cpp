#include "game_state.h"
#include "lilith.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// =======================
// 外部随机数引擎
// =======================
extern thread_local std::mt19937 rng;

// =======================
// MCTS 节点
// =======================
struct MCTSNode {
    GameState state;

    // 当前局面轮到谁走
    int player;

    // 从父节点到此节点的落子
    std::pair<int, int> move;

    MCTSNode* parent;

    std::vector<MCTSNode*> children;

    // visit count
    int N;

    // total value
    float W;

    // prior probability
    float P;

    bool expanded;

    MCTSNode(
        const GameState& s,
        int p,
        std::pair<int,int> m = std::make_pair(-1,-1),
        MCTSNode* par = nullptr
    )
        : state(s),
          player(p),
          move(m),
          parent(par),
          N(0),
          W(0.0f),
          P(0.0f),
          expanded(false)
    {}

    ~MCTSNode() {
        for (auto* child : children) {
            delete child;
        }
    }
};

// =======================
// Backup
// =======================
void backup(MCTSNode* leaf, float v) {

    MCTSNode* node = leaf;

    while (node != nullptr) {

        node->N++;

        v = std::max(-1.0f, std::min(1.0f, v));

        node->W += v;

        // 切换视角
        v = -v;

        node = node->parent;
    }
}

// =======================
// Select
// =======================
MCTSNode* select(MCTSNode* root, float c_puct) {

    MCTSNode* node = root;

    while (node->expanded) {

        if (node->children.empty()) {
            break;
        }

        float best_ucb = -1e30f;

        MCTSNode* best_child = nullptr;

        for (auto* child : node->children) {

            float Q;

            // =======================
            // FPU
            // =======================
            if (child->N > 0) {

                Q = -(child->W / child->N);

                Q = std::max(-1.0f, std::min(1.0f, Q));

            } else {

                // First Play Urgency
                Q = -0.2f;
            }

            float U =
                c_puct *
                child->P *
                std::sqrt((float)std::max(1, node->N)) /
                (1.0f + child->N);

            float ucb = Q + U;

            if (ucb > best_ucb) {

                best_ucb = ucb;

                best_child = child;
            }
        }

        if (best_child == nullptr) {
            break;
        }

        node = best_child;
    }

    return node;
}

// =======================
// Expand
// =======================
void expand(MCTSNode* leaf, Network& net) {

    // =======================
    // 终局检测
    // =======================
    auto end = leaf->state.game_end_check();

    if (end.first) {

        int winner = end.second;

        // 当前节点 player 视角
        float final_v =
            (winner == leaf->state.player_turn)
            ? 1.0f
            : -1.0f;

        leaf->expanded = true;

        backup(leaf, final_v);

        return;
    }

    // =======================
    // 获取合法走法
    // =======================
    auto moves = leaf->state.get_legal_moves();

    if (moves.empty()) {

        leaf->expanded = true;

        backup(leaf, -1.0f);

        return;
    }

    // =======================
    // 编码局面
    // =======================
    std::vector<float> features = encode(leaf->state);

    Matrix input(1, 648);

    for (int i = 0; i < 648; ++i) {
        input.at(0, i) = features[i];
    }

    // =======================
    // 网络前向
    // =======================
    Matrix policy;

    float value;

    net.forward(input, policy, value);

    value = std::max(-1.0f, std::min(1.0f, value));

    // =======================
    // 扩展节点
    // =======================
    leaf->expanded = true;

    float sum_P = 0.0f;

    for (auto& m : moves) {

        GameState next = leaf->state.clone();

        if (!next.apply_move(m.first, m.second)) {
            continue;
        }

        int next_player = next.player_turn;

        auto* child =
            new MCTSNode(
                next,
                next_player,
                m,
                leaf
            );

        int idx = m.first * 9 + m.second;

        float p = policy.at(0, idx);

        p = std::max(0.0f, p);

        child->P = p;

        sum_P += p;

        leaf->children.push_back(child);
    }

    // =======================
    // normalize prior
    // =======================
    if (sum_P > 1e-8f) {

        for (auto* child : leaf->children) {
            child->P /= sum_P;
        }

    } else {

        // fallback uniform
        float uniform =
            1.0f / std::max(1, (int)leaf->children.size());

        for (auto* child : leaf->children) {
            child->P = uniform;
        }
    }

    // =======================
    // 回传 value
    // =======================
    backup(leaf, value);
}

// =======================
// MCTS 主搜索
// =======================
std::vector<float> mcts_search(
    const GameState& root_state,
    int root_player,
    Network& net,
    int sims = 1600,
    float c_puct = 2.0f,
    bool add_noise = true
) {

    MCTSNode* root =
        new MCTSNode(root_state, root_player);

    // =======================
    // root expand
    // =======================
    expand(root, net);

    // =======================
    // Dirichlet Noise
    // =======================
    if (
        add_noise &&
        !root->children.empty()
    ) {

        const float epsilon = 0.1f;

        const float alpha = 0.3f;

        std::gamma_distribution<float> gamma(alpha, 1.0f);

        std::vector<float> noise(root->children.size());

        float noise_sum = 0.0f;

        for (float& n : noise) {

            n = gamma(rng);

            noise_sum += n;
        }

        if (noise_sum > 1e-8f) {

            for (size_t i = 0; i < root->children.size(); ++i) {

                float dirichlet =
                    noise[i] / noise_sum;

                root->children[i]->P =
                    (1.0f - epsilon) * root->children[i]->P +
                    epsilon * dirichlet;
            }
        }
    }

    // =======================
    // simulations
    // =======================
    for (int i = 0; i < sims; ++i) {

        MCTSNode* leaf =
            select(root, c_puct);

        auto end =
            leaf->state.game_end_check();

        // =======================
        // terminal
        // =======================
        if (end.first) {

            int winner = end.second;

            float final_v =
                (winner == leaf->state.player_turn)
                ? 1.0f
                : -1.0f;

            backup(leaf, final_v);

            continue;
        }

        // =======================
        // expand
        // =======================
        expand(leaf, net);
    }

    // =======================
    // 构建 π
    // =======================
    std::vector<float> pi(81, 0.0f);

    float total_N = 0.0f;

    for (auto* child : root->children) {

        int idx =
            child->move.first * 9 +
            child->move.second;

        pi[idx] = (float)child->N;

        total_N += child->N;
    }

    // normalize
    if (total_N > 1e-8f) {

        for (float& p : pi) {
            p /= total_N;
        }

    } else {

        // fallback uniform legal
        auto legal =
            root_state.get_legal_moves();

        if (!legal.empty()) {

            float uniform =
                1.0f / legal.size();

            for (auto& m : legal) {

                pi[m.first * 9 + m.second] =
                    uniform;
            }
        }
    }

    delete root;

    return pi;
}