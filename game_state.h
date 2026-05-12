// game_state.h
#ifndef GAME_STATE_H
#define GAME_STATE_H

#include <vector>
#include <utility>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <numeric>

const int N = 9;

extern thread_local std::mt19937 rng;
extern thread_local std::uniform_int_distribution<int> dist;

struct GameState {
    // ========== 基本棋盘 ==========
    int board[N][N] = {};        // 0空 1白 2黑
    int plants[N][N] = {};       // 0无植物 1~2层植物
    int hp[3] = {0, 6, 6};       // 1白 2黑
    int cnt[3] = {0, 0, 0};      // 1白 2黑 棋子数
    int player_turn = 1;          // 1白 2黑
    int ascend_turn = 0;          // 0正常，否则为先攻方编号
    int turn_number = 0;

    struct Ascend {
        int attack = 0;
        std::vector<std::pair<int, int>> slots;
        void init() { attack = 0; slots.clear(); }
    } ascend_player[3];

    // ========== 官方植物系统状态（索引统一为1白2黑） ==========
    int grow_count = 11;
    int unascend_charge = 25;
    int unascend_chargef[3] = {0, 9, 9};   // [1]白 [2]黑 （索引0闲置）
    bool just_unascend = false;
    int over_status = 0;                    // 0/1/2
    int ascend_status = 0;                  // 0/1/2
    int turn_pos[2] = {0, 0};               // [0]=x [1]=y (与玩家无关)
    int turn_count_plant = 0;
    int plant_stone_count = 0;

    // 植物评分用随机数发生器（与官方完全一致）
    struct RandAIC {
        uint32_t s[4];
        RandAIC(uint32_t seed = 12345) {
            s[0] = seed; s[1] = seed + 123456789;
            s[2] = seed + 362436069; s[3] = seed + 521288629;
            for (int i = 0; i < 132; i++) next();
        }
        uint32_t next() {
            uint32_t t = s[0] ^ (s[0] << 11);
            s[0] = s[1]; s[1] = s[2]; s[2] = s[3];
            s[3] = s[3] ^ (s[3] >> 19) ^ (t ^ (t >> 8));
            return s[3];
        }
        int nextInt(int max) { return next() % max; }
    } rand_aic;

    struct RandUnity {
        uint32_t s[4];
        RandUnity(uint32_t seed = 67890) {
            s[0] = seed; s[1] = seed * 1812433253 + 1;
            s[2] = s[1] * 1812433253 + 1; s[3] = s[2] * 1812433253 + 1;
        }
        uint32_t next() {
            uint32_t t = s[0] ^ (s[0] << 11);
            s[0] = s[1]; s[1] = s[2]; s[2] = s[3];
            s[3] = s[3] ^ (s[3] >> 19) ^ (t ^ (t >> 8));
            return s[3];
        }
        int range(int min, int max) {
            if (min == max) return min;
            int d = max - min;
            return min + (next() % d);
        }
        void shuffle(std::vector<int>& arr) {
            int n = arr.size();
            for (int i = 0; i < (n - 1) * 23; i++)
                std::swap(arr[range(0, n)], arr[range(0, n)]);
        }
    } rand_unity;

    // 方向数组（官方顺序）
    int plant_dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    int plant_dy[8] = {-1, 1, 0, 0, -1, 1, 1, -1};

    // ========== 方法 ==========
    void init();
    void init_plant();
    void print_board();
    int coords_check(int x, int y);
    int ascend_check(int x, int y, int op);
    std::pair<int, int> game_end_check();
    void generate_plant();
    void game();

    GameState clone() const;
    std::vector<std::pair<int, int>> get_legal_moves() const;
    bool apply_move(int x, int y);

private:
    void get_align(int x, int y, int stone, int& max_align, int& max_align_total);
    std::vector<std::pair<int, int>> p_scan_pos(bool flag3, bool flag4);
    std::vector<std::pair<int, int>> p_scan_score(
        const std::vector<std::pair<int, int>>& points, bool atk, int count,
        bool flag, bool flag2, bool flag4);
    void p_sort_pos(std::vector<std::pair<int, int>>& pts);
    int get_player_index() const { return player_turn; }   // 返回1或2
};

#endif