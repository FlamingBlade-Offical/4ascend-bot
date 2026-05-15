#include "lilith.h"
#include "mcts.h"
#include "game_state.h"
#include <iostream>
#include <string>
using namespace std;
// 全局随机数定义
 // 修正了缺少参数的编译错误
thread_local std::mt19937 rng(std::random_device{}());
thread_local std::uniform_int_distribution<int> dist(0, 1000000000);
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
    cout << "    ";  // 植物图例的间隔
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

            bool has_plant = plants[i][j];
            int piece = board[i][j];   // 0空,1白,2黑

            // 根据 (ascend槽位, 棋子颜色, 植物) 组合字符
            if (is_ascend_slot) {
                if (has_plant) cout << "A ";
                else           cout << "A ";
            } else if (piece == 1) {
                if (has_plant) cout << "W ";
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
            cout << plants[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

// ================= 基本规则 =================
void GameState::init() {
    hp[1] = hp[2] = 6;
    player_turn = 1;
    cnt[1] = cnt[2] = 0;
    turn_number = 0;
    ascend_turn = 0;
    ascend_player[1].init();
    ascend_player[2].init();
    init_plant();

    uint32_t seed_aic = rng();
    uint32_t seed_unity = rng();
    rand_aic = RandAIC(seed_aic);
    rand_unity = RandUnity(seed_unity);
}

void GameState::init_plant() {
    grow_count = 11;
    unascend_charge = 25;
    unascend_chargef[1] = unascend_chargef[2] = 9;
    unascend_chargef[0] = 0;
    just_unascend = false;
    over_status = 0;
    ascend_status = 0;
    turn_pos[0] = turn_pos[1] = 0;
    turn_count_plant = 0;
    plant_stone_count = 0;
}

// ================= AI 接口 =================
GameState GameState::clone() const { return *this; }

std::vector<std::pair<int, int>> GameState::get_legal_moves() const {
    std::vector<std::pair<int, int>> moves;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (board[i][j] == 0)
                moves.push_back({i, j});
    return moves;
}

int GameState::coords_check(int x, int y) { return x >= 0 && x < 9 && y >= 0 && y < 9; }

int GameState::ascend_check(int x, int y, int op) {
    int dx[4] = {0, 1, 1, 1};
    int dy[4] = {1, 1, 0, -1};
    for (int i = 0; i < 4; ++i) {
        int pos = 0, neg = 0;
        while (coords_check(x + pos * dx[i], y + pos * dy[i]) && board[x + pos * dx[i]][y + pos * dy[i]] == op) pos++;
        while (coords_check(x - neg * dx[i], y - neg * dy[i]) && board[x - neg * dx[i]][y - neg * dy[i]] == op) neg++;
        if (pos + neg >= 5) {
            for (int j = 1; j < pos; ++j) {
                ascend_player[op].attack += 1 + plants[x + j * dx[i]][y + j * dy[i]];
                ascend_player[op].slots.push_back({x + j * dx[i], y + j * dy[i]});
            }
            for (int j = 1; j < neg; ++j) {
                ascend_player[op].attack += 1 + plants[x - j * dx[i]][y - j * dy[i]];
                ascend_player[op].slots.push_back({x - j * dx[i], y - j * dy[i]});
            }
        }
    }
    if (ascend_player[op].attack != 0) {
        ascend_player[op].attack += 1 + plants[x][y];
        ascend_player[op].slots.push_back({x, y});
        for (auto& [cx, cy] : ascend_player[op].slots) board[cx][cy] = 0;
        return 1;
    }
    return 0;
}

pair<int, int> GameState::game_end_check() {
    if (hp[1] <= 0) return {1, 2};
    if (hp[2] <= 0) return {1, 1};
    if (cnt[1] + cnt[2] == 81) {
        if (hp[1] > hp[2]) return {1, 1};
        if (hp[2] > hp[1]) return {1, 2};
        return {1, (cnt[1] > cnt[2]) ? 1 : 2};
    }
    return {0, 0};
}

// ================= 官方植物系统（玩家索引统一为1/2） =================
void GameState::get_align(int x, int y, int stone, int& max_align, int& max_align_total) {
    int counts[8];
    for (int d = 0; d < 8; d++) {
        int cnt = 1;
        int nx = x, ny = y;
        while (true) {
            nx += plant_dx[d]; ny += plant_dy[d];
            if (!coords_check(nx, ny) || board[nx][ny] != stone) break;
            cnt++;
        }
        counts[d] = cnt;
    }
    max_align = 0;
    for (int i = 0; i < 8; i += 2) max_align = max(max_align, counts[i] + counts[i + 1] - 1);
    max_align_total = -7;
    for (int i = 0; i < 8; i++) max_align_total += counts[i];
}

void GameState::p_sort_pos(vector<pair<int, int>>& pts) {
    sort(pts.begin(), pts.end(), [](auto& a, auto& b) {
        return a.second != b.second ? a.second < b.second : a.first < b.first;
    });
}

vector<pair<int, int>> GameState::p_scan_pos(bool flag3, bool flag4) {
    vector<int> tst;
    if (unascend_chargef[1] == 0) tst.push_back(1);
    if (unascend_chargef[2] == 0) tst.push_back(2);
    if (tst.empty()) { tst.push_back(1); tst.push_back(2); }

    vector<pair<int, int>> result;
    auto add = [&](int x, int y) {
        if (find(result.begin(), result.end(), make_pair(x, y)) == result.end())
            result.push_back({x, y});
    };

    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            int stone = board[x][y];
            if (stone == 0) continue;
            bool filled = true;
            bool a = false;
            if (flag3) {
                if (find(tst.begin(), tst.end(), stone) != tst.end() || (filled && (flag4 || over_status != 0))) a = true;
            } else a = filled;
            if (!a) continue;

            if (flag3) add(x, y);
            for (int d = 0; d < 8; d++) {
                int nx = x + plant_dx[d], ny = y + plant_dy[d];
                if (coords_check(nx, ny) && board[nx][ny] == 0 && plants[nx][ny] < 2)
                    add(nx, ny);
            }
        }
    }
    if (result.empty() && flag3) {
        tst = {1, 2};
        return p_scan_pos(false, flag4);
    }
    p_sort_pos(result);
    return result;
}

// 修复：增加 self / foe 参数，不再从 player_turn 推断
vector<pair<int, int>> GameState::p_scan_score(
    const vector<pair<int, int>>& points, bool atk, int count,
    bool flag, bool flag2, bool flag4, int self, int foe)
{
    int tst[4] = {0}; int tst_len = 0;
    if (unascend_chargef[1] == 0) tst[tst_len++] = 1;
    if (unascend_chargef[2] == 0) tst[tst_len++] = 2;
    if (tst_len == 0) { tst[0]=1; tst[1]=2; tst_len=2; }

    struct Scored { int x, y, score, bit; };
    vector<Scored> result, buf;
    int byte_ = 0;

    for (auto& [x, y] : points) {
        int score = 500;
        bool flag5 = false, flag6 = atk;
        int abyte = 0;
        int self_stone = self;
        int foe_stone = foe;
        int self_align, self_align_t, foe_align, foe_align_t;
        get_align(x, y, self_stone, self_align, self_align_t);
        get_align(x, y, foe_stone, foe_align, foe_align_t);

        if (self_align >= 4 || foe_align >= 4) {
            if (over_status == 2) {
                int b = 0;
                if (self_align < 4 && !flag2) b |= 2;
                if (foe_align < 4 && !flag) b |= 1;
                if (b == 1 || b == 2) {
                    score += 450;
                    if (plants[x][y] == 0) score += 100;
                    if (b == 1) flag6 = true;
                    abyte = b;
                }
            }
            score -= 450;
        }

        if (flag6) {
            if (self_align_t < foe_align_t) score -= abs(self_align_t - foe_align_t) * 3;
            if (self_align < foe_align) score -= abs(self_align - foe_align) * 15;
            if (self_align == 3 && foe_align <= 1) {
                if (flag4) flag5 = true;
                else score += static_cast<int>(std::sin((float)(25 - unascend_charge) / 25.0f * 1.5707963f) * 120);
            }
        } else {
            if (self_align_t < foe_align_t && flag && !flag2) score += abs(self_align_t - foe_align_t) * 6;
            else if (self_align_t > foe_align_t && !flag && flag2) score += abs(self_align_t - foe_align_t) * 6;
            else if (self_align_t != foe_align_t) score -= abs(self_align_t - foe_align_t) * 3;
            if (self_align != foe_align) score -= abs(self_align - foe_align) * 15;
        }

        if (plants[x][y] != 0 && !flag4) score -= 30;
        else if (atk) {
            bool in_tst = false;
            for (int i = 0; i < tst_len; i++) if (tst[i] == self) in_tst = true;
            if (in_tst) score += 50;
        }
        score += rand_aic.nextInt(20);

        if (abyte > 0) {
            byte_ |= abyte;
            buf.push_back({x, y, score, abyte});
        } else {
            result.push_back({x, y, score, 0});
            if (flag5) {
                count++;
                result.push_back({x, y, score + 30 - rand_aic.nextInt(90), 0});
            }
        }
    }

    if (byte_ > 0) {
        vector<int> indices(buf.size());
        iota(indices.begin(), indices.end(), 0);
        rand_unity.shuffle(indices);
        for (int idx = 0; idx < (int)buf.size() && byte_ > 0; idx++) {
            int i = indices[idx];
            int bit = buf[i].bit;
            if ((byte_ & bit) != 0) {
                byte_ &= ~bit;
                count++;
                int score2 = buf[i].score + 40 - rand_aic.nextInt(120);
                result.push_back({buf[i].x, buf[i].y, score2, 0});
                result.push_back(buf[i]);
            }
        }
    }

    vector<int> indices2(result.size());
    iota(indices2.begin(), indices2.end(), 0);
    rand_unity.shuffle(indices2);
    sort(indices2.begin(), indices2.end(), [&](int a, int b) { return result[a].score > result[b].score; });

    vector<pair<int, int>> selected;
    for (int idx : indices2) {
        if ((int)selected.size() >= count) break;
        int x = result[idx].x, y = result[idx].y;
        if (plants[x][y] < 2) {
            selected.push_back({x, y});
            int bit = result[idx].bit;
            if (bit & 1) unascend_chargef[self] = max(unascend_chargef[self], 4);
            if (bit & 2) unascend_chargef[foe] = max(unascend_chargef[foe], 4);
        }
    }
    return selected;
}

void GameState::generate_plant() {
    turn_count_plant++;
    // ★ 核心修复：self 应为植物生成后即将行动的一方（3 - player_turn）
    int self = 3 - player_turn;   // 即将落子方
    int foe  = player_turn;       // 刚行动完的一方

    if (ascend_status == 1) return;
    bool atk = (ascend_status == 2);

    if (!atk) {
        if (unascend_charge > 0) unascend_charge--;
        if (unascend_chargef[1] > 0) unascend_chargef[1]--;
        if (unascend_chargef[2] > 0) unascend_chargef[2]--;
        just_unascend = false;
    }

    grow_count--;
    if (grow_count > 0 && !(atk && !just_unascend)) return;

    plant_stone_count = 0;
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) if (board[i][j]) plant_stone_count++;

    int plant_count = (turn_count_plant >= 65) ? 3 : 2;
    if (plant_stone_count >= 44 && over_status == 0) over_status = 1;
    if (over_status == 1 && hp[self] <= hp[foe]) over_status = 2; // 此处也改用 self/foe 而非 player_turn
    if (over_status != 0 && plant_stone_count < 22) over_status = 0;

    bool flag3 = atk;
    bool flag4 = atk && unascend_charge <= 0;
    bool flag = unascend_chargef[self] > 0;      // 修正：用 self 而非 player_turn
    bool flag2 = unascend_chargef[foe] > 0;       // 修正：用 foe
    if (unascend_chargef[self] > 0) atk = false;  // 修正

    auto cand = p_scan_pos(flag3, flag4);
    if (!cand.empty()) {
        if (over_status == 2) plant_count++;
        plant_count = min(plant_count, (int)cand.size());
        // 传入 self 和 foe
        auto selected = p_scan_score(cand, atk, plant_count, flag, flag2, flag4, self, foe);
        for (auto& [x, y] : selected) plants[x][y]++;
    }

    int base = max(7, 11 - (turn_count_plant / 22) * 2);
    grow_count = base;
    if (over_status != 0) {
        grow_count = max(1, grow_count / 2);
        if (grow_count % 2 == 0) grow_count--;
    }

    if (atk) {
        float val = 12.5f + unascend_charge + (25 - unascend_charge) * 0.4f;
        val = max(12.5f, min(val, 25.0f));
        unascend_charge = (int)ceil(val);
        // ★ 充能始终加给 self（进攻方，即即将落子方），符合官方逻辑
        unascend_chargef[self] = max(unascend_chargef[self], (over_status != 0) ? 4 : 9);
        just_unascend = true;
    }
}
bool GameState::apply_move(int x, int y) {
    if (!coords_check(x, y) || board[x][y] != 0) return false;
    board[x][y] = player_turn;
    turn_number++;
    cnt[player_turn]++;
    turn_pos[0] = x; turn_pos[1] = y;

    if (ascend_check(x, y, player_turn)) {
        if (ascend_turn == 0) {
            ascend_turn = player_turn;
            player_turn = 3 - player_turn;
            ascend_status = 1; // START
            return true;
        }
    }

    if (ascend_turn) {
        int decrease = 0;
        for (auto& [cx, cy] : ascend_player[ascend_turn].slots) {
            if (cx == x && cy == y) {
                decrease = 1 + plants[cx][cy]; // ★ 多层植物修正
                break;
            }
        }
        ascend_player[ascend_turn].attack -= decrease;
        int a1 = ascend_player[1].attack, a2 = ascend_player[2].attack;
        if (a1 > a2) hp[2] -= a1 - a2;
        else if (a2 > a1) hp[1] -= a2 - a1;
        for (int i = 1; i <= 2; i++) for (auto& [cx, cy] : ascend_player[i].slots) plants[cx][cy] = 0;
        cnt[1] -= ascend_player[1].slots.size();
        cnt[2] -= ascend_player[2].slots.size();
        ascend_player[1].init(); ascend_player[2].init();
        ascend_status = 2; // END
        generate_plant();          // ★ 此处 plant 生成时已使用修正后的 self
        ascend_status = 0;
        ascend_turn = 0;
    } else {
        generate_plant();          // 普通回合植物生成
    }
    player_turn = 3 - player_turn;
    return true;
}

// 打印策略分布（81维），显示访问次数最多的前K个动作
void print_policy(const std::vector<float>& pi) {
    std::vector<std::pair<float, int>> sorted;
    for (int i = 0; i < 81; ++i) {
        sorted.push_back({pi[i], i});
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    std::cout << "Policy top 10:\n";
    for (int i = 0; i < 10 && i < sorted.size(); ++i) {
        int idx = sorted[i].second;
        int x = idx / 9, y = idx % 9;
        std::cout << "  (" << x << "," << y << "): " << std::fixed << std::setprecision(4) << sorted[i].first << "\n";
    }

    // 计算熵
    float entropy = 0.0f;
    for (float p : pi) {
        if (p > 1e-9f) entropy -= p * std::log(p);
    }
    std::cout << "Policy entropy: " << entropy << "\n";
}
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
            cout << "Lilith 思考中...\n";
            // 搜索次数可以调小一点让响应更快（如 200）
            auto pi = mcts_search(game, game.player_turn, ai, 1600, 2.0,false);
            int best_idx = int(std::max_element(pi.begin(), pi.end()) - pi.begin());
            print_policy(pi);
            int ax = best_idx / 9;
            int ay = best_idx % 9;
            cout << "Lilith 走棋 (" << ax << ", " << ay << ")\n";
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