// game.cpp
#include "game_state.h"
#include "lilith.h"
#include "mcts.h"
#include <algorithm>
#include <ctime>
#include <random>
#include <future>
#include <fstream>
#include <cmath>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <iomanip>
#include <numeric>
#include <iostream>

using namespace std;

thread_local std::mt19937 rng(std::random_device{}());
thread_local std::uniform_int_distribution<int> dist(0, 1000000000);

std::atomic<int> active_tasks{0};
const int MAX_CONCURRENT = 6;   // 留两个核给系统

template<typename F>
auto launch_limited(F&& f) {
    while (active_tasks >= MAX_CONCURRENT)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    active_tasks++;
    return std::async(std::launch::async, [f = std::forward<F>(f)]() mutable {
        // 防异常卡死：确保 active_tasks-- 一定执行
        struct Guard {
            std::atomic<int>& a;
            ~Guard() { a--; }
        } guard{active_tasks};

        return f();
    });
}

// ================= 进度条 =================
std::atomic<int> games_done{0};
std::atomic<int> games_total{0};
std::atomic<int> evals_done{0};
std::atomic<int> evals_total{0};
std::mutex progress_mutex;

void print_progress(const char* label, int done, int total) {
    if (total <= 0) return;
    std::lock_guard<std::mutex> lock(progress_mutex);
    int bar_width = 30;
    float ratio = (float)done / total;
    int pos = int(bar_width * ratio);
    std::cout << "\r" << label << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(ratio * 100) << "% ("
              << done << "/" << total << ")" << std::flush;
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

void GameState::print_board() {
    cout << ascend_player[1].attack << " " << ascend_player[2].attack << endl;
    cout << hp[1] << " " << hp[2] << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) cout << board[i][j] << " ";
        for (int j = 0; j < N; ++j) cout << (plants[i][j] ? "* " : ". ");
        cout << endl;
    }
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
    int self = 3 - player_turn;
    int foe  = player_turn;

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
    if (over_status == 1 && hp[self] <= hp[foe]) over_status = 2;
    if (over_status != 0 && plant_stone_count < 22) over_status = 0;

    bool flag3 = atk;
    bool flag4 = atk && unascend_charge <= 0;
    bool flag = unascend_chargef[self] > 0;
    bool flag2 = unascend_chargef[foe] > 0;
    if (unascend_chargef[self] > 0) atk = false;

    auto cand = p_scan_pos(flag3, flag4);
    if (!cand.empty()) {
        if (over_status == 2) plant_count++;
        plant_count = min(plant_count, (int)cand.size());
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
            ascend_status = 1;
            return true;
        }
    }

    if (ascend_turn) {
        int decrease = 0;
        for (auto& [cx, cy] : ascend_player[ascend_turn].slots) {
            if (cx == x && cy == y) {
                decrease = 1 + plants[cx][cy];
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
        ascend_status = 2;
        generate_plant();
        ascend_status = 0;
        ascend_turn = 0;
    } else {
        generate_plant();
    }
    player_turn = 3 - player_turn;
    return true;
}
// ================= 训练模块 =================
struct TrainingSample {
    std::vector<float> features;
    std::vector<float> pi;
    float z;
};

std::vector<TrainingSample> self_play_one_game(Network& net, int sims = 1600) {
    GameState game;
    game.init();
    std::vector<std::pair<GameState, std::vector<float>>> history;

    while (!game.game_end_check().first) {
        auto pi = mcts_search(game, game.player_turn, net, sims);   // add_noise 默认 true
        history.push_back({game.clone(), pi});

        int sampled_idx;
        if (game.turn_number < 20) {
            std::discrete_distribution<int> dist_pi(pi.begin(), pi.end());
            sampled_idx = dist_pi(rng);
        } else {
            sampled_idx = int(std::max_element(pi.begin(), pi.end()) - pi.begin());
        }
        int x = sampled_idx / 9;
        int y = sampled_idx % 9;
        game.apply_move(x, y);
    }

    int winner = game.game_end_check().second;
    std::vector<TrainingSample> samples;
    samples.reserve(history.size());
    for (auto& [state, pi] : history) {
        float z = (state.player_turn == winner) ? 1.0f : -1.0f;
        samples.push_back({encode(state), pi, z});
    }
    return samples;
}

int play_one_game(Network& net1, Network& net2, int sims = 1200) {
    GameState game;
    game.init();
    while (!game.game_end_check().first) {
        auto pi = (game.player_turn == 1)
            ? mcts_search(game, 1, net1, sims, 2.0f, false)
            : mcts_search(game, 2, net2, sims, 2.0f, false);

        int best_idx = int(std::max_element(pi.begin(), pi.end()) - pi.begin());
        game.apply_move(best_idx / 9, best_idx % 9);
    }
    return game.game_end_check().second; // 1 or 2
}

void apply_transform(std::vector<float>& feat, std::vector<float>& pi, int rot, bool mirror) {
    if (rot == 0 && !mirror) return;

    auto transform81 = [rot, mirror](std::vector<float>& arr) {
        std::vector<float> temp(81);
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                int ni = i, nj = j;
                if (mirror) nj = 8 - j;
                int idx_new;
                if (rot == 0)      idx_new = ni * 9 + nj;
                else if (rot == 1) idx_new = nj * 9 + (8 - ni);
                else if (rot == 2) idx_new = (8 - ni) * 9 + (8 - nj);
                else               idx_new = (8 - nj) * 9 + ni;
                temp[idx_new] = arr[i * 9 + j];
            }
        }
        arr = temp;
    };

    for (int c = 0; c < 8; ++c) {
        std::vector<float> channel(feat.begin() + c * 81, feat.begin() + (c + 1) * 81);
        transform81(channel);
        std::copy(channel.begin(), channel.end(), feat.begin() + c * 81);
    }
    transform81(pi);
}

std::vector<TrainingSample> replay_buffer;
const int replay_capacity = 150000;

int main() {
    Network best_net;
    try {
        best_net = Network::load("best_net.bin");
        std::cout << "Loaded existing network.\n";
    } catch (...) {
        std::cout << "No saved network, starting from scratch.\n";
    }

    const int games_per_iter = 240;
    const int eval_games = 200;     // 每边 200，总 400
    const int epochs = 3;
    const int warmup_iterations = 0;

    for (int iter = 0; ; ++iter) {
        std::cout << "Best net initial weight: " << best_net.layer1.W.at(0,0) << "\n";

        // lr schedule (你选的稳健版)
        double lr;
        if (iter < 30) lr = 2e-4;
        else lr = 2e-4 * std::pow(0.95, (iter - 30) / 5.0);

        // ====== 自对弈 ======
        games_total = games_per_iter;
        games_done = 0;
        std::vector<std::future<std::vector<TrainingSample>>> self_play_futures;
        self_play_futures.reserve(games_per_iter);

        for (int g = 0; g < games_per_iter; ++g) {
            self_play_futures.emplace_back(
                launch_limited([best = best_net]() mutable {
                    // 这里 sims 你现在用 2000；如果你想更均衡算力，可改 1200
                    auto res = self_play_one_game(best, 1600);
                    games_done++;
                    print_progress("Self-Play", games_done.load(), games_total.load());
                    return res;
                })
            );
        }

        std::vector<TrainingSample> all_data;
        for (auto& fut : self_play_futures) {
            auto game_data = fut.get();
            for (auto& sample : game_data) {
                if (iter < warmup_iterations) sample.z = 0.0f;
                all_data.push_back(sample);
            }
        }
        std::cout << "\n";

        // push into replay
        for (auto& s : all_data) replay_buffer.push_back(std::move(s));
        while ((int)replay_buffer.size() > replay_capacity) {
            replay_buffer.erase(replay_buffer.begin()); // O(n)，后续建议换 deque
        }

        // ====== 训练 ======
        Network new_net = best_net;

        // 这两个是“稳定器”：早期建议保持
        new_net.set_value_weight(0.25f);
        new_net.set_policy_smoothing(0.02f);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            int batch_size = std::min(8192, (int)replay_buffer.size());
            std::vector<int> indices(replay_buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);

            float total_loss = 0.0f;
            AdamConfig opt((float)lr, 0.9f, 0.999f, 1e-8f, 1.0f);

            for (int i = 0; i < batch_size; i++) {
                auto& sample = replay_buffer[indices[i]];
                for (int rot = 0; rot < 4; ++rot) {
                    for (int mirror = 0; mirror < 2; ++mirror) {
                        TrainingSample aug = sample;
                        apply_transform(aug.features, aug.pi, rot, mirror);
                        Matrix input(1, 648);
                        for (int j = 0; j < 648; ++j) input.at(0, j) = sample.features[j];
                        total_loss += new_net.train(input, aug.pi, aug.z, opt);
                    }
                }
            }

            std::cout << "Iter " << iter << " Epoch " << epoch
                      << " avg loss: " << total_loss / (batch_size * 8) << "\n";
        }

        std::cout << "Sample weight: " << new_net.layer1.W.at(0,0) << "\n";

        // entropy diagnostic on empty board
        {
            GameState empty_state;
            empty_state.init();
            auto feat = encode(empty_state);
            Matrix input(1, 648);
            for (int i = 0; i < 648; ++i) input.at(0, i) = feat[i];

            Matrix pol; float val;
            new_net.forward(input, pol, val);

            float entropy = 0.0f;
            for (int i = 0; i < 81; ++i) {
                float p = pol.at(0, i);
                if (p > 1e-9f) entropy -= p * std::log(p);
            }
            std::cout << "New net policy entropy: " << entropy << "\n";

            best_net.forward(input, pol, val);
            float entropy_best = 0.0f;
            for (int i = 0; i < 81; ++i) {
                float p = pol.at(0, i);
                if (p > 1e-9f) entropy_best -= p * std::log(p);
            }
            std::cout << "Best net policy entropy: " << entropy_best << "\n";
        }

        if (iter < warmup_iterations) {
            best_net = new_net;
            std::cout << "Warmup iteration " << iter << ": best_net forced updated (no eval).\n";
            replay_buffer.clear();
            continue;
        }

        // ====== 评估 ======
        evals_total = eval_games * 2;
        evals_done = 0;

        int wins_black = 0; // new_net as player2
        int wins_white = 0; // new_net as player1

        std::vector<std::future<int>> futures_black, futures_white;
        futures_black.reserve(eval_games);
        futures_white.reserve(eval_games);

        for (int g = 0; g < eval_games; ++g) {
            futures_black.emplace_back(
                launch_limited([best = best_net, neu = new_net]() mutable {
                    int res = play_one_game(best, neu, 1600);
                    evals_done++;
                    print_progress("Eval      ", evals_done.load(), evals_total.load());
                    return res;
                })
            );
            futures_white.emplace_back(
                launch_limited([best = best_net, neu = new_net]() mutable {
                    int res = play_one_game(neu, best, 1600);
                    evals_done++;
                    print_progress("Eval      ", evals_done.load(), evals_total.load());
                    return res;
                })
            );
        }

        for (auto& fut : futures_black) if (fut.get() == 2) wins_black++;
        for (auto& fut : futures_white) if (fut.get() == 1) wins_white++;

        std::cout << "\n";

        float black_win_rate = (float)wins_black / eval_games;
        float white_win_rate = (float)wins_white / eval_games;

        std::cout << "New net (Black) win rate: " << black_win_rate
                  << ", (White) win rate: " << white_win_rate << "\n";

        float threshold = 0.55f;
        float floor_side = 0.45f;

        float overall = 0.5f * (black_win_rate + white_win_rate);
        float min_side = std::min(black_win_rate, white_win_rate);

        std::cout << "Overall: " << overall
                  << "  MinSide: " << min_side
                  << "  Threshold: " << threshold
                  << "  FloorSide: " << floor_side << "\n";

        if (overall >= threshold && min_side >= floor_side) {
            best_net = new_net;
            try {
                best_net.save("best_net.bin");
                std::cout << "Network saved to best_net.bin\n";
                system("mkdir -p net");
                try {
                    std::string backup_name = "net/best_net_iter_" + std::to_string(iter) + ".bin";
                    std::ifstream src("best_net.bin", std::ios::binary);
                    std::ofstream dst(backup_name, std::ios::binary);
                    dst << src.rdbuf();
                    std::cout << "Backup saved to " << backup_name << "\n";
                } catch (...) {
                    std::cout << "Backup failed.\n";
                }
            } catch (...) {
                std::cout << "Save failed.\n";
            }
        } else {
            std::cout << "Network rejected.\n";
        }
    }

    return 0;
}