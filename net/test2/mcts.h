// mcts.h
#ifndef MCTS_H
#define MCTS_H

#include "game_state.h"
#include "lilith.h"
#include <vector>

struct MCTSNode {
    GameState state;
    int player;
    std::pair<int,int> move;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int N;
    float W;
    float P;
    bool expanded;

    MCTSNode(const GameState& s, int p, std::pair<int,int> m = std::make_pair(-1,-1), MCTSNode* par = nullptr);
    ~MCTSNode();
};

// 主搜索函数
std::vector<float> mcts_search(const GameState& root_state, int root_player, Network& net,
                               int sims = 1600, float c_puct = 2.0f,bool add_noise = true);

#endif