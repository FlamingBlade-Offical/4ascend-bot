#include "lilith.h"
#include "mcts.h"
#include <iostream>
#include <string>
using namespace std;
// 全局随机数定义
 // 修正了缺少参数的编译错误
thread_local std::mt19937 rng(std::random_device{}());
thread_local std::uniform_int_distribution<int> dist(0, 1000000000);
// ================= 原有方法实现 =================
void GameState::init() {
    hp[1] = hp[2] = 6;
    player_turn = 1;
    cnt[1] = cnt[2] = 0;

    ascend_turn = 0;
    ascend_player[1].init();
    ascend_player[2].init();

    plant_top = plant_down = plant_left = plant_right = 4;
    plant_range = 2;
    soil_fertility = 0.0;
    max_fertility = 10.0;
    growth_fertility = 1.4;
    fertility_recharge_rate = 0.83;
}

void GameState::print_board() {
    // ========== 状态摘要 ==========
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    cout << "Turn: " << (player_turn == 1 ? "White (○)" : "Black (●)") << "\n";
    cout << "HP  → White: " << hp[1] << "   Black: " << hp[2] << "\n";

    if (ascend_turn != 0) {
        cout << "════ ASCEND Phase ════\n";
        cout << "Attacker: " << (ascend_turn == 1 ? "White" : "Black")
             << "   Attack Power: " << ascend_player[ascend_turn].attack << "\n";
        // 可选：显示进攻方槽位坐标
        // cout << "Slots: ";
        // for (auto& [cx, cy] : ascend_player[ascend_turn].slots)
        //     cout << "(" << cx << "," << cy << ") ";
        // cout << "\n";
    }
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // ========== 棋盘（带坐标） ==========
    // 列号
    cout << "   ";
    for (int j = 0; j < N; ++j) cout << j << " ";
    cout << "       ";  // 植物图例的间隔
    for (int j = 0; j < N; ++j) cout << j << " ";
    cout << "\n";

    for (int i = 0; i < N; ++i) {
        // 左侧：棋盘
        cout << i << "  ";
        for (int j = 0; j < N; ++j) {
            // 判断是否在进攻方的 ASCEND 槽位（棋子被拿起的位置）
            bool is_ascend_slot = false;
            if (ascend_turn != 0) {
                for (auto& [cx, cy] : ascend_player[ascend_turn].slots) {
                    if (cx == i && cy == j) { is_ascend_slot = true; break; }
                }
            }

            bool has_plant = (plants[i][j] > 0);
            int piece = board[i][j];   // 0空,1白,2黑

            // 根据 (ascend槽位, 棋子颜色, 植物) 组合字符
            if (is_ascend_slot) {
                if (has_plant) cout << "A*";
                else           cout << "A ";
            } else if (piece == 1) {
                if (has_plant) cout << "W*";
                else           cout << "W ";
            } else if (piece == 2) {
                if (has_plant) cout << "B*";
                else           cout << "B ";
            } else { // 空格
                if (has_plant) cout << "* ";
                else           cout << ". ";
            }
        }

        // 中间分隔
        cout << "    ";

        // 右侧：植物层独立显示（可选，为了更清晰）
        for (int j = 0; j < N; ++j) {
            if (plants[i][j]) cout << "* ";
            else              cout << ". ";
        }
        cout << "\n";
    }
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

int GameState::coords_check(int x, int y) {
    return (x >= 0 && y >= 0 && x < N && y < N) ? 1 : 0;
}

int GameState::ascend_check(int x, int y, int op) {
    for (int i = 0; i < 4; ++i) {
        int positive_cnt = 0, negative_cnt = 0;
        while (coords_check(x + positive_cnt * dx[i], y + positive_cnt * dy[i])
            && board[x + positive_cnt * dx[i]][y + positive_cnt * dy[i]] == op)
            ++positive_cnt;
        while (coords_check(x - negative_cnt * dx[i], y - negative_cnt * dy[i])
            && board[x - negative_cnt * dx[i]][y - negative_cnt * dy[i]] == op)
            ++negative_cnt;

        if (positive_cnt + negative_cnt >= 5) {
            for (int j = 1; j < positive_cnt; ++j) {
                ascend_player[op].attack += 1 + plants[x + j * dx[i]][y + j * dy[i]];
                ascend_player[op].slots.push_back(make_pair(x + j * dx[i], y + j * dy[i]));
            }
            for (int j = 1; j < negative_cnt; ++j) {
                ascend_player[op].attack += 1 + plants[x - j * dx[i]][y - j * dy[i]];
                ascend_player[op].slots.push_back(make_pair(x - j * dx[i], y - j * dy[i]));
            }
        }
    }
    if (ascend_player[op].attack != 0) {
        ascend_player[op].attack += 1 + plants[x][y];
        ascend_player[op].slots.push_back(make_pair(x, y));
        for (auto &[cx, cy] : ascend_player[op].slots)
            board[cx][cy] = 0;  // 攻方棋子拿起
        return 1;
    }
    return 0;
}

pair<int,int> GameState::game_end_check() {//第一个数传是否结束游戏，第二个数传谁赢了
    if (hp[1] <= 0) return make_pair(1,2);
    if (hp[2] <= 0) return make_pair(1,1);
    if (cnt[1] + cnt[2] == N * N) {
        if (hp[1] > hp[2]) return make_pair(1,1);
        else if (hp[2] > hp[1]) return make_pair(1,2);
        else {
            if (cnt[1] > cnt[2]) return make_pair(1,1);
            else if (cnt[2] > cnt[1]) return make_pair(1,2);
        }
    }
    return make_pair(0,0);
}

void GameState::generate_plant() {
    plant_down = plant_right = 4;
    plant_left = plant_top = 4;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (board[i][j] || plants[i][j]) {
                plant_down = max(i, plant_down);
                plant_top = min(i, plant_top);
                plant_right = max(j, plant_right);
                plant_left = min(j, plant_left);
            }
    plant_down = min(N - 1, plant_down + plant_range);
    plant_right = min(N - 1, plant_right + plant_range);
    plant_left = max(0, plant_left - plant_range);
    plant_top = max(0, plant_top - plant_range);

    int max_grow = (int)(soil_fertility / growth_fertility);
    int min_grow = (int)(0.3 * soil_fertility / growth_fertility);
    int grow_number = min_grow + dist(rng) % (max_grow - min_grow + 1);

    vector<pair<int, int> > blank_space;
    for (int i = plant_top; i <= plant_down; ++i)
        for (int j = plant_left; j <= plant_right; ++j)
            if (board[i][j] == 0 && plants[i][j] == 0)
                blank_space.push_back(make_pair(i, j));

    int siz = (int)blank_space.size() - 1;
    grow_number = min(grow_number, siz + 1);
    for (int i = 0; i < grow_number; ++i) {
        int j = i + dist(rng) % (siz - i + 1);
        plants[blank_space[j].first][blank_space[j].second] = 1;
        swap(blank_space[i], blank_space[j]);
    }
    soil_fertility -= growth_fertility * grow_number;
}

void GameState::game() {
    init();
    while (true) {
        //print_board();
        int px, py;
        cin >> px >> py;
        if (!coords_check(px, py) || board[px][py]) {
            cout << "input is invalid" << endl;
            continue;
        }
        apply_move(px, py);
        if (game_end_check().first)
            break;
    }
}

// ================= 新增供 AI 调用的方法 =================

GameState GameState::clone() const {
    return *this;  // 默认逐成员拷贝（vector 等会正确复制）
}

vector<pair<int, int> > GameState::get_legal_moves() const {
    vector<pair<int, int> > moves;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (board[i][j] == 0)
                moves.push_back(make_pair(i, j));
    return moves;
}

bool GameState::apply_move(int x, int y) {
    if (!coords_check(x, y) || board[x][y] != 0)
        return false;  // 非法走法

    board[x][y] = player_turn;
    cnt[player_turn]++;

    if (ascend_check(x, y, player_turn)) {
        if (ascend_turn == 0) {
            ascend_turn = player_turn;
            player_turn = 3 - player_turn;
            return true;  // 进入 ASCEND 回应阶段
        }
    }

    if (ascend_turn) {
        int decrease = 0;
        for (auto &[cx, cy] : ascend_player[ascend_turn].slots) {
            if (cx == x && cy == y) {
                decrease = 1 + (plants[cx][cy] > 0 ? 1 : 0);
                break;
            }
        }
        ascend_player[ascend_turn].attack -= decrease;

        int atk1 = ascend_player[1].attack, atk2 = ascend_player[2].attack;
        if (atk1 > atk2)
            hp[2] -= atk1 - atk2;
        else if (atk2 > atk1)
            hp[1] -= atk2 - atk1;

        for (int i = 1; i <= 2; ++i)
            for (auto &[cx, cy] : ascend_player[i].slots)
                plants[cx][cy] = 0;
        cnt[1] -= ascend_player[1].slots.size();
        cnt[2] -= ascend_player[2].slots.size();

        ascend_player[1].init();
        ascend_player[2].init();

        generate_plant();
        ascend_turn = 0;
    } else {
        soil_fertility += fertility_recharge_rate;
        if (soil_fertility >= max_fertility) {
            soil_fertility = max_fertility;
            generate_plant();
        }
    }
    player_turn = 3 - player_turn;
    return true;  // 游戏继续
}
std::vector<float> encode(const GameState& state);
struct TrainingSample {
    std::vector<float> features;
    std::vector<float> pi;
    float z;
};

int main() {
    // 加载训练好的网络
    Network ai;
    try {
        ai = Network::load("best_net.bin");
        cout << "已加载网络权重\n";
    } catch (...) {
        cout << "未找到 best_net.bin，使用随机网络\n";
    }

    // 选择先后手
    cout << "请选择你的颜色 (输入 white 或 black): ";
    string choice;
    cin >> choice;
    int human_player = 0;
    bool human_is_white = false;
    if (choice == "white" || choice == "w") {
        human_player = 1;
        human_is_white = true;
        cout << "你执白（先手）\n";
    } else {
        human_player = 2;
        human_is_white = false;
        cout << "你执黑（后手）\n";
    }

    GameState game;
    game.init();
    game.print_board();
    int flag = 0;
    while (game.game_end_check().first == 0) {
        if (game.player_turn == human_player) {
            // 人类回合
            cout << "你的回合，请输入坐标 (x y): ";
            int x, y;
            cin >> x >> y;
            if (!game.coords_check(x, y) || game.board[x][y] != 0) {
                cout << "无效走法，请重试\n";
                continue;
            }
            game.apply_move(x, y);
        } else {
            // AI 回合
            cout << "AI 思考中...\n";
            // 搜索次数可以调小一点让响应更快（如 200）
            auto pi = mcts_search(game, game.player_turn, ai, 1600, 2.0,false);
            int best_idx = 0;
            float best_p = pi[0];
            for (int i = 1; i < 81; ++i) {
                if (pi[i] > best_p) {
                    best_p = pi[i];
                    best_idx = i;
                }
            }
            int ax = best_idx / 9;
            int ay = best_idx % 9;
            cout << "AI 走棋 (" << ax << ", " << ay << ")\n";
            game.apply_move(ax, ay);
        }
        game.print_board();
    }

    cout << "游戏结束！";
    auto result = game.game_end_check();
    if (result.second == 1)
        cout << "白方胜利！\n";
    else
        cout << "黑方胜利！\n";
    return 0;
}